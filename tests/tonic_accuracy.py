"""Quantitative Stage 2 report: tonic-detection accuracy.

Run directly for a readable table:

    python -m tests.tonic_accuracy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from swaras._essentia import essentia_available
from swaras.audio_io import load_file
from swaras.cents import cents_between
from swaras.config import TonicParams
from swaras.pitch import get_detector
from swaras.tonic import detect_tonic, detect_tonic_histogram
from tests import synthetic

SR = 22_050
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

#: Tonics the task asks us to cover.
TONICS = [220.0, 261.63, 293.66]

#: Phrase shapes a khayal performance might use.
PHRASES = {
    "Sa Re Ga Pa": [0, 200, 400, 700],
    "Sa Pa Sa' Ni'": [0, 700, 1200, 1900],
    "full scale up+down": [0, 200, 400, 500, 700, 900, 1100, 900, 700, 500, 400, 200, 0],
    "Pa Sa Re Ga": [700, 0, 200, 400],
    "Sa Ga Ma Pa Dha Ni": [0, 400, 500, 700, 900, 1100],
}


def render(tonic: float, cents: list[int], repeat: int = 3) -> np.ndarray:
    """Render a repeated phrase at a tonic."""
    freqs = [synthetic.cents_to_hz(c, tonic) for c in cents] * repeat
    return synthetic.sequence(freqs, note_duration_s=0.4, vibrato_cents=0.0)


def run(detector: str) -> list[dict]:
    """Detect the tonic for every phrase at every tonic.

    Args:
        detector: Pitch detector name.

    Returns:
        One result dict per (phrase, tonic) combination.
    """
    rows = []
    for phrase, cents in PHRASES.items():
        for tonic in TONICS:
            y = render(tonic, cents)
            contour = get_detector(detector).estimate(y, SR)
            result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
            err = (
                float(cents_between(tonic, result.hz)) if result.is_valid else float("nan")
            )
            rows.append(
                {
                    "phrase": phrase,
                    "tonic": tonic,
                    "detected": result.hz,
                    "error": err,
                    "conf": result.confidence,
                }
            )
    return rows


def transposition_sweep(detector: str, phrase: list[int]) -> list[dict]:
    """Detect the tonic across a wide range of transposition.

    Args:
        detector: Pitch detector name.
        phrase: Phrase in cents.

    Returns:
        One result dict per tonic.
    """
    rows = []
    for tonic in [130.81, 146.83, 174.61, 196.0, 220.0, 246.94, 261.63, 293.66, 329.63, 392.0, 440.0, 493.88]:
        y = render(tonic, phrase, repeat=2)
        contour = get_detector(detector).estimate(y, SR)
        result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
        err = float(cents_between(tonic, result.hz)) if result.is_valid else float("nan")
        rows.append({"tonic": tonic, "detected": result.hz, "error": err})
    return rows


def main() -> int:
    """Print the Stage 2 accuracy report.

    Returns:
        Process exit code.
    """
    detectors = ["pyin"] + (["melodia"] if essentia_available() else [])

    for det in detectors:
        rows = run(det)
        errs = np.array([r["error"] for r in rows], dtype=np.float64)
        ok = np.abs(errs) <= 15.0
        print("=" * 78)
        print(f"Stage 2 tonic accuracy, detector = {det}, tolerance = 15 cents")
        print("=" * 78)
        print(f"{'phrase':22s} " + " ".join(f"{t:>10.2f}" for t in TONICS))
        for phrase in PHRASES:
            cells = []
            for r in rows:
                if r["phrase"] != phrase:
                    continue
                e = r["error"]
                mark = " " if abs(e) <= 15 else "!"
                cells.append(f"{e:+9.1f}{mark}")
            print(f"{phrase:22s} " + " ".join(f"{c:>10s}" for c in cells))
        print("-" * 78)
        print(
            f"  within 15 cents: {int(ok.sum())}/{len(rows)} "
            f"({ok.mean():.0%})   median |error| {np.nanmedian(np.abs(errs)):.1f}c   "
            f"max {np.nanmax(np.abs(errs)):.1f}c"
        )
        print()

    print("=" * 78)
    print("Transposition sweep (Sa-Re-Ga-Pa, pyin)")
    print("=" * 78)
    for r in transposition_sweep("pyin", PHRASES["Sa Re Ga Pa"]):
        e = r["error"]
        mark = " " if abs(e) <= 15 else "!"
        print(f"  {r['tonic']:8.2f} Hz -> {r['detected']:8.2f} Hz  {e:+8.1f}c{mark}")
    print()

    for name in ("phrase_c4", "libri1", "libri2", "libri3"):
        path = SAMPLES / f"{name}.wav"
        if not path.is_file():
            continue
        audio = load_file(path)
        contour = get_detector("melodia" if essentia_available() else "pyin").estimate(
            audio.samples, audio.sample_rate
        )
        result = detect_tonic(contour, audio.samples, audio.sample_rate)
        best = result.candidates[0] if result.candidates else None
        print(f"{name:12s} Sa = {result.hz if result.hz is None else round(result.hz, 1)} Hz"
              f"  conf {result.confidence:.2f}  method {result.method}")
        if best is not None:
            print(f"{'':12s}   sa_peak {best.sa_peak:.2f}  pa_peak {best.pa_peak:.2f}"
                  f"  octave_peak {best.octave_peak:.2f}")
        for c in result.candidates[1:3]:
            print(f"{'':12s}   alt: {c.hz:8.2f} Hz  conf {c.confidence:.2f}")
        if result.warning:
            print(f"{'':12s}   warning: {result.warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
