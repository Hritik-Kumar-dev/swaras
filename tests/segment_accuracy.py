"""Quantitative Stage 3 report: round-trip transcription accuracy.

Synthesises a known swara sequence, transcribes it, and compares. Run as:

    python -m tests.segment_accuracy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from swaras._essentia import essentia_available
from swaras.cents import cents_between
from swaras.config import SegmentParams
from swaras.normalize import contour_to_cents, Octave
from swaras.pitch import get_detector
from swaras.segment import segment_contour
from swaras.swar import cents_to_swar
from swaras.tonic import detect_tonic
from swaras.tuning import load_shruti_table
from tests import synthetic

SR = 22_050
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TABLE = load_shruti_table()

#: (name, cents, expected short symbol) triples.
PHRASES = {
    "Sa Re Ga Pa": [(0, "S"), (200, "R"), (400, "G"), (700, "P")],
    "ascending scale": [
        (0, "S"), (200, "R"), (305, "g"), (400, "G"), (508, "M"),
        (700, "P"), (900, "D"), (1099, "N"),
    ],
    "Sa Sa' Pa Sa": [(0, "S"), (1200, "S'"), (700, "P"), (0, "S")],
    "with mandra": [(-1200, "S."), (-498, "P."), (0, "S"), (700, "P")],
    "komal and tivra": [(0, "S"), (90, "r"), (200, "R"), (601, "Mt"), (1007, "n"), (1100, "N")],
}
TONICS = [220.0, 261.63, 293.66]


def _render(freqs: list[float], repeat: int) -> np.ndarray:
    """Render a phrase, repeated, with a rest between repeats.

    A rest matters: a phrase ending on the same pitch it starts on, repeated
    back to back, is genuinely one sustained note, and the pipeline correctly
    merges it. Without the rest the expected output would have to contain a
    note split that no performer would sing.

    Args:
        freqs: Note frequencies in Hz.
        repeat: How many times to repeat.

    Returns:
        The rendered audio.
    """
    one = synthetic.sequence(freqs, note_duration_s=0.45, vibrato_cents=0.0)
    rest = np.zeros(int(0.25 * SR))
    parts = [one]
    for _ in range(repeat - 1):
        parts.extend([rest, one])
    # Trailing silence, so the last note is not clipped by the analysis window.
    parts.append(np.zeros(int(0.3 * SR)))
    return np.concatenate(parts)


def run_case(
    detector: str,
    tonic: float,
    phrase: list[tuple[int, str]],
    repeat: int = 2,
    sa_hz: float | None = None,
) -> dict:
    """Transcribe one synthetic phrase and compare with the expected notation.

    Args:
        detector: Pitch tracker name.
        tonic: Sa in Hz, the true tonic of the phrase.
        phrase: ``(cents, expected symbol)`` pairs, relative to Sa.
        repeat: How many times to repeat the phrase.
        sa_hz: Sa to use instead of detecting it. Supplying it isolates
            segmentation and mapping from the octave ambiguity in tonic
            detection, which is measured separately.

    Returns:
        Dict with the expected and got notation, the match rate, and the
        per-note cent errors.
    """
    freqs = [synthetic.cents_to_hz(c, tonic) for c, _ in phrase]
    y = _render(freqs, repeat)
    contour = get_detector(detector).estimate(y, SR)
    if sa_hz is None:
        result = detect_tonic(contour, y, SR)
        used_sa = result.hz
    else:
        used_sa = sa_hz
    cents = contour_to_cents(contour, used_sa)
    seg = segment_contour(cents, SegmentParams())
    notes = [cents_to_swar(n.cents, TABLE, n.confidence, n.start_s, n.end_s) for n in seg.notes]
    got = [n.label for n in notes]
    expected = [sym for _, sym in phrase] * repeat
    return {
        "expected": expected,
        "got": got,
        "sa_hz": used_sa,
        "notes": notes,
        "exact": got == expected,
        "match": _match_rate(expected, got),
        "seg": seg,
    }


def _match_rate(expected: list[str], got: list[str]) -> float:
    """Longest-common-subsequence ratio between two note lists.

    Using LCS rather than a positional comparison means an inserted or dropped
    note costs only its own length, instead of shifting every later note and
    reporting a cascade of errors.

    Args:
        expected: The reference notation.
        got: The transcribed notation.

    Returns:
        Fraction of the expected notes matched, in ``[0, 1]``.
    """
    if not expected:
        return 0.0
    m, n = len(expected), len(got)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if expected[i] == got[j] else max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0] / m


def main() -> int:
    """Print the Stage 3 round-trip report.

    Returns:
        Process exit code.
    """
    detectors = ["pyin"] + (["melodia"] if essentia_available() else [])

    for det in detectors:
        print("=" * 84)
        print(f"Stage 3 round trip, detector = {det}, Sa supplied (isolates mapping)")
        print("=" * 84)
        for name, phrase in PHRASES.items():
            cells = []
            exact = 0
            for tonic in TONICS:
                r = run_case(det, tonic, phrase, sa_hz=tonic)
                exact += int(r["exact"])
                cells.append("=" if r["exact"] else f"{r['match']:.0%}")
            status = "exact" if exact == len(TONICS) else f"{exact}/{len(TONICS)} exact"
            print(f"  {name:20s} " + "  ".join(f"{c:>7s}" for c in cells) + f"   {status}")
        print()

        print(f"  same phrases, Sa auto-detected (includes tonic errors)")
        print("-" * 84)
        for name, phrase in PHRASES.items():
            cells = []
            for tonic in TONICS:
                r = run_case(det, tonic, phrase)
                cells.append("=" if r["exact"] else f"{r['match']:.0%}")
            print(f"  {name:20s} " + "  ".join(f"{c:>7s}" for c in cells))
        print()

    print("=" * 84)
    print("Octave marking (Sa = 261.63 Hz, pYIN, Sa supplied)")
    print("=" * 84)
    phrase = [(-1200, "S."), (-498, "P."), (0, "S"), (700, "P"), (1200, "S'"), (2299, "N'")]
    r = run_case("pyin", 261.63, phrase, repeat=1, sa_hz=261.63)
    print(f"  expected {r['expected']}")
    print(f"  got      {r['got']}")
    print()

    print("=" * 84)
    print("Transcription of the bundled phrase_c4 sample (Sa = 261.63 Hz)")
    print("=" * 84)
    from swaras.audio_io import load_file

    audio = load_file(SAMPLES / "phrase_c4.wav")
    det = "melodia" if essentia_available() else "pyin"
    contour = get_detector(det).estimate(audio.samples, audio.sample_rate)
    t = detect_tonic(contour, audio.samples, audio.sample_rate)
    cents = contour_to_cents(contour, t.hz)
    seg = segment_contour(cents, SegmentParams())
    notes = [cents_to_swar(n.cents, TABLE) for n in seg.notes]
    print(f"  Sa = {t.hz:.2f} Hz   notes: {len(notes)}   transitions: {seg.diagnostics.get('transitions')}")
    print(f"  notation: {' '.join(n.label for n in notes)}")
    print(f"  expected: S R G P D N S R G P D N")
    for i, n in enumerate(notes, 1):
        print(
            f"    {i:>2}  {n.label:<4} {n.absolute_cents:7.1f}c  dev {n.deviation_cents:+6.1f}c"
            f"  {n.duration_s:.3f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
