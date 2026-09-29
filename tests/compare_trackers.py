"""Quantitative comparison of the two F0 trackers on real and synthetic audio.

Run directly for a readable table:

    python -m tests.compare_trackers
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from swaras._essentia import essentia_available
from swaras.audio_io import load_file
from swaras.cents import cents_between
from swaras.pitch import PitchContour, get_detector

SR = 22_050
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def frame_error_cents(contour: PitchContour, truth_hz: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Cents error of a contour against a ground-truth F0 track.

    Args:
        contour: Estimated contour.
        truth_hz: Ground-truth instantaneous frequency, sampled on ``times``.
        times: Times in seconds, aligned with ``truth_hz``.

    Returns:
        Signed error in cents for each frame (nan where unvoiced).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(cents_between(truth_hz, contour.f0))


def sweep_case(name: str = "sweep") -> dict:
    """Compare trackers on a synthetic sweep with a known F0 track.

    Args:
        name: Filename stem under ``samples/``.

    Returns:
        Dict of per-detector error statistics in cents.
    """
    from tests import synthetic

    audio = load_file(SAMPLES / f"{name}.wav")
    truth = np.geomspace(80.0, 900.0, audio.samples.size)
    times = np.arange(audio.samples.size) / audio.sample_rate
    out = {}
    for det in ("melodia", "pyin"):
        if det == "melodia" and not essentia_available():
            continue
        contour = get_detector(det).estimate(audio.samples, audio.sample_rate)
        err = frame_error_cents(contour, np.interp(contour.times, times, truth), contour.times)
        err = np.abs(err[contour.voiced & np.isfinite(err)])
        err = err[err.size // 10 : -(err.size // 10)]
        out[det] = {
            "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "voiced": contour.voiced_fraction,
        }
    return out


def real_case(name: str) -> dict:
    """Summarise each tracker on a real recording.

    There is no ground truth for speech, so this reports the F0 distribution
    and how much the two trackers disagree with each other, which is the
    useful signal when choosing a default.

    Args:
        name: Filename stem under ``samples/``.

    Returns:
        Dict with voiced fraction, F0 percentiles, and the median absolute
        disagreement between trackers in cents.
    """
    audio = load_file(SAMPLES / f"{name}.wav")
    rows: dict[str, dict] = {}
    for det in ("melodia", "pyin"):
        if det == "melodia" and not essentia_available():
            continue
        contour = get_detector(det).estimate(audio.samples, audio.sample_rate)
        vf = contour.voiced_f0()
        rows[det] = {
            "voiced": contour.voiced_fraction,
            "median": float(np.median(vf)) if vf.size else float("nan"),
            "p5": float(np.percentile(vf, 5)) if vf.size else float("nan"),
            "p95": float(np.percentile(vf, 95)) if vf.size else float("nan"),
        }
    if "melodia" in rows and "pyin" in rows:
        m = get_detector("melodia").estimate(audio.samples, audio.sample_rate)
        p = get_detector("pyin").estimate(audio.samples, audio.sample_rate)
        both = m.voiced & p.voiced
        if both.any():
            d = np.abs(np.asarray(cents_between(m.f0[both], p.f0[both])))
            d = d[np.isfinite(d)]
            rows["disagreement_median_cents"] = float(np.median(d))
            rows["disagreement_within_30c_pct"] = float((d < 30).mean())
    return rows


def main() -> int:
    """Print the comparison tables.

    Returns:
        Process exit code.
    """
    print("=" * 78)
    print("Synthetic octave sweep 80 -> 900 Hz (ground truth known)")
    print("=" * 78)
    for det, s in sweep_case().items():
        print(
            f"  {det:8s} median {s['median']:6.1f}c  p90 {s['p90']:6.1f}c  "
            f"voiced {s['voiced']:5.1%}"
        )

    for name in ("phrase_c4", "libri1", "libri2", "libri3"):
        path = SAMPLES / f"{name}.wav"
        if not path.is_file():
            continue
        print()
        print("=" * 78)
        print(f"{name}  (real recording, no ground truth)")
        print("=" * 78)
        for key, s in real_case(name).items():
            if not isinstance(s, dict):
                print(f"  {key:28s} {s:.1f}")
                continue
            print(
                f"  {key:28s} voiced {s['voiced']:5.1%}  median {s['median']:7.1f} Hz  "
                f"p5 {s['p5']:6.1f}  p95 {s['p95']:7.1f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
