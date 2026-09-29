"""Quantitative Stage 5 report: the 22-shruti layer.

Run directly for a readable table:

    python -m tests.shruti_accuracy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from swaras._essentia import essentia_available
from swaras.audio_io import load_file
from swaras.config import SegmentParams
from swaras.normalize import contour_to_cents
from swaras.pitch import get_detector
from swaras.segment import segment_contour
from swaras.shruti import cents_to_shruti, shruti_statistics
from swaras.swar import segment_to_swaras
from swaras.tonic import detect_tonic
from swaras.tuning import load_shruti_table
from tests import synthetic

SR = 22_050
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TABLE = load_shruti_table()


def grid_coverage() -> None:
    """Report every shruti, its neighbour gaps and its clarity margin."""
    print("=" * 78)
    print("The 22-shruti grid: spacing and how far a match can be trusted")
    print("=" * 78)
    print(f"{'#':>3} {'cents':>7} {'swar':<10} {'var':<8} {'gap below':>10} {'gap above':>10} {'clarity':>8}")
    c = TABLE.shruti_cents
    n = c.size
    for i, s in enumerate(TABLE.shrutis):
        lo = c[i - 1] if i > 0 else c[n - 1] - 1200.0
        hi = c[i + 1] if i < n - 1 else c[0] + 1200.0
        clarity = min(c[i] - lo, hi - c[i]) / 2.0
        print(
            f"{i:>3} {c[i]:>7.0f} {s.swar:<10} {s.variant:<8} "
            f"{c[i] - lo:>10.0f} {hi - c[i]:>10.0f} {clarity:>8.1f}"
        )
    print()


def round_trip_shrutis(tonic: float, phrase: list[tuple[int, str]], detector: str = "pyin") -> dict:
    """Transcribe a phrase and report its shruti placements.

    Args:
        tonic: Sa in Hz.
        phrase: ``(cents, expected label)`` pairs relative to Sa.
        detector: Pitch tracker name.

    Returns:
        Dict with per-note shruti labels, deviations and statistics.
    """
    freqs = [synthetic.cents_to_hz(c, tonic) for c, _ in phrase]
    y = synthetic.sequence(freqs, note_duration_s=0.45, vibrato_cents=0.0)
    y = np.concatenate([y, np.zeros(int(0.3 * SR))])
    contour = get_detector(detector).estimate(y, SR)
    seg = segment_contour(contour_to_cents(contour, tonic), SegmentParams())
    notes = segment_to_swaras(seg.notes, TABLE)
    shrutis = [cents_to_shruti(n.absolute_cents, TABLE) for n in notes]
    return {"shrutis": shrutis, "stats": shruti_statistics(shrutis), "notes": notes}


def main() -> int:
    """Print the Stage 5 report.

    Returns:
        Process exit code.
    """
    grid_coverage()

    print("=" * 78)
    print("Round trip: synthesized phrase -> shruti placement (pYIN)")
    print("=" * 78)
    # Synthesized on equal-temperament-ish pitches, which is what a test tone
    # gives. The point is that the shruti layer reports the offset rather than
    # snapping the note onto a position it was not sung at.
    phrase = [(0, "S"), (200, "R"), (400, "G"), (700, "P"), (900, "D"), (1100, "N")]
    for tonic in (220.0, 261.63, 293.66):
        r = round_trip_shrutis(tonic, phrase)
        print(f"  tonic {tonic:.2f} Hz")
        for n in r["shrutis"]:
            print(
                f"    sung {n.absolute_cents:7.1f}c -> {n.label:<5} "
                f"dev {n.deviation_cents:+6.1f}c  conf {n.confidence:.2f}  "
                f"clarity {n.clarity_cents:.0f}c"
            )
        s = r["stats"]
        print(
            f"    mean |dev| {s['mean_abs_deviation_cents']}c   "
            f"worst {s['max_abs_deviation_cents']}c   "
            f"unclear {s['unclear_count']}/{s['count']}"
        )
        print()

    print("=" * 78)
    print("Deliberate deviations: are they reported faithfully?")
    print("=" * 78)
    print(f"  {'sung':>8}  {'shruti':<6} {'dev':>7} {'conf':>5}  note")
    cases = [
        (0.0, "exactly on Sa"),
        (8.0, "8 cents sharp of Sa"),
        (-8.0, "8 cents flat of Sa"),
        (100.0, "between the two Re komal shrutis (90/112)"),
        (397.0, "exactly midway between the two Ga shrutis (386/408)"),
        (702.0, "exactly on Pa"),
        (745.0, "43 sharp of Pa, still nearer Pa than Dha komal (792)"),
        (780.0, "78 sharp of Pa, now nearer Dha komal (792)"),
    ]
    for cents, why in cases:
        n = cents_to_shruti(cents, TABLE)
        print(f"  {cents:>8.0f}  {n.label:<6} {n.deviation_cents:>+7.1f} {n.confidence:>5.2f}  {why}")
    print()

    print("=" * 78)
    print("Deviation recovered from a real recording")
    print("=" * 78)
    for name in ("phrase_c4", "libri1", "libri2", "libri3"):
        path = SAMPLES / f"{name}.wav"
        if not path.is_file():
            continue
        audio = load_file(path)
        det = "melodia" if essentia_available() else "pyin"
        contour = get_detector(det).estimate(audio.samples, audio.sample_rate)
        t = detect_tonic(contour, audio.samples, audio.sample_rate)
        if not t.is_valid:
            print(f"  {name:10s} no Sa detected")
            continue
        seg = segment_contour(contour_to_cents(contour, t.hz), SegmentParams())
        notes = segment_to_swaras(seg.notes, TABLE)
        shrutis = [cents_to_shruti(n.absolute_cents, TABLE) for n in notes]
        s = shruti_statistics(shrutis)
        print(
            f"  {name:10s} Sa {t.hz:7.2f} Hz  {s['count']:>3} notes  "
            f"mean |dev| {s['mean_abs_deviation_cents']}c  worst {s['max_abs_deviation_cents']}c  "
            f"unclear {s['unclear_count']}"
        )
        if shrutis:
            print(f"             {' '.join(n.label_with_deviation for n in shrutis[:14])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
