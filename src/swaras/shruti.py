"""Stage 5: cents to the nearest of the 22 shrutis, with the deviation.

The shruti layer is deliberately not a second label. A shruti is a pitch
position, and a performance sits *near* one rather than on it, so every result
carries the signed distance in cents from the position it was matched to. That
is the honest answer, and it is the one the task asks for: real performance is
context-dependent, so "nearest shruti plus deviation" says something true where
a hard name would not.

The layer sits beside :mod:`swaras.swar` rather than replacing it. The two
answer different questions: ``swar`` asks "which of the 12 positions is this",
``shruti`` asks "how finely can you place it, and how far off are you".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cents import OCTAVE_CENTS
from .normalize import Octave, fold_to_octave
from .swar import SwaraNote, octave_for
from .tuning import Shruti, ShrutiTable, load_shruti_table


@dataclass(frozen=True)
class ShrutiNote:
    """A note placed on the 22-shruti grid.

    Attributes:
        index: Index of the matched shruti in the table.
        swara: Name of the swara the shruti belongs to.
        variant: ``"low"`` or ``"high"`` within a 22-cent pair, or
            ``"shuddha"`` for a singleton such as Sa or Pa.
        deviation_cents: Signed distance from the shruti. Negative means the
            note is sung flatter than the position. This is the measurement;
            everything else here is a label attached to it.
        cents: Measured pitch, folded into one octave.
        absolute_cents: The unfolded measurement.
        octave: Register relative to the detected Sa.
        confidence: How unambiguously the nearest shruti was chosen, in
            ``[0, 1]``. It is 1.0 on a shruti and falls to 0.0 exactly halfway
            to the next one, so a note in the middle of a 70-cent gap is
            reported as an uncertain choice rather than a confident one.
        clarity_cents: Signed distance to the midpoint between this shruti and
            its neighbour, in cents. The distance at which the match would
            flip to a different shruti.
        start_s: Note start time.
        end_s: Note end time.
    """

    index: int
    swara: str
    variant: str
    deviation_cents: float
    cents: float
    absolute_cents: float
    octave: Octave
    confidence: float
    clarity_cents: float
    start_s: float = 0.0
    end_s: float = 0.0

    @property
    def duration_s(self) -> float:
        """Note length in seconds."""
        return max(0.0, self.end_s - self.start_s)

    @property
    def label(self) -> str:
        """Short text label, e.g. ``"S0"`` or ``"R4h"``."""
        return f"{_swar_short(self.swara)}{self.index}{_variant_suffix(self.variant)}"

    @property
    def label_with_deviation(self) -> str:
        """Label with the signed deviation, e.g. ``"S0 -8c"``."""
        return f"{self.label} {self.deviation_cents:+.0f}c"

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "shruti_index": self.index,
            "swar": self.swara,
            "variant": self.variant,
            "label": self.label,
            "deviation_cents": round(self.deviation_cents, 1),
            "cents": round(self.cents, 1),
            "absolute_cents": round(self.absolute_cents, 1),
            "octave": int(self.octave),
            "octave_name": self.octave.name_short,
            "confidence": round(self.confidence, 3),
            "clarity_cents": round(self.clarity_cents, 1),
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
        }


#: The short symbols used by the labels, keyed by swara name.
_SHORT = {
    "Sa": "S", "Re komal": "R", "Re": "R", "Ga komal": "G", "Ga": "G",
    "Ma": "M", "Ma tivra": "M", "Pa": "P", "Dha komal": "D", "Dha": "D",
    "Ni komal": "N", "Ni": "N",
}


def _swar_short(swara: str) -> str:
    """Short symbol for a swara name, falling back to its first letter."""
    return _SHORT.get(swara, swara[:1].upper())


def _variant_suffix(variant: str) -> str:
    """Suffix distinguishing the two shrutis of an altered swara."""
    return {"low": "l", "high": "h"}.get(variant, "")


def _extended_targets(table: ShrutiTable) -> tuple[np.ndarray, np.ndarray]:
    """Shruti positions repeated across three octaves.

    Searching a repeated array makes the circular case fall out for free: a
    note at 1190 cents finds the Sa' entry at 1200 rather than the code having
    to compare against a wrapped copy by hand.

    Args:
        table: The tuning table.

    Returns:
        Tuple ``(cents, indices)`` of equal length, where ``indices`` maps each
        entry back to its shruti index in the table.
    """
    base = table.shruti_cents
    cents = np.concatenate([base - OCTAVE_CENTS, base, base + OCTAVE_CENTS])
    indices = np.tile(np.arange(base.size), 3)
    return cents, indices


def nearest_shruti(
    cents: np.ndarray | float,
    table: ShrutiTable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the nearest shruti for each cents value.

    Args:
        cents: Cents above Sa, any magnitude; ``nan`` passes through.
        table: Tuning table; defaults to the packaged one.

    Returns:
        Tuple ``(indices, deviations, confidences)``. The deviation is signed
        and measured the short way round the octave, so a note eight cents flat
        reports ``-8`` and not ``+1192``. The confidence is 1.0 on a shruti and
        falls to 0.0 exactly at the midpoint to the next one.
    """
    tbl = table or load_shruti_table()
    targets, target_index = _extended_targets(tbl)
    c = np.atleast_1d(np.asarray(cents, dtype=np.float64))
    finite = np.isfinite(c)
    folded, _ = fold_to_octave(np.where(finite, c, 0.0))
    safe = np.where(finite, folded, 0.0)

    distances = np.abs(safe[:, None] - targets[None, :])
    rows = np.arange(safe.size)
    winner = np.argmin(distances, axis=1)
    nearest = distances[rows, winner]

    # Second nearest, with the winner pushed out of reach so it cannot win
    # twice.
    runner_up = distances.copy()
    runner_up[rows, winner] = np.inf
    second = runner_up.min(axis=1)

    idx = target_index[winner]
    raw = safe - targets[winner]
    # Wrap the deviation the short way round. The nearest shruti is at most half
    # an octave away, so anything larger is the wrapped copy of the same
    # position and has to be pulled back.
    deviation = np.where(np.abs(raw) > OCTAVE_CENTS / 2, raw - np.sign(raw) * OCTAVE_CENTS, raw)

    boundary = (nearest + second) / 2.0
    confidence = np.where(boundary > 0, 1.0 - nearest / boundary, 0.0)
    confidence = np.clip(confidence, 0.0, 1.0)

    return (
        np.where(finite, idx, -1),
        np.where(finite, deviation, np.nan),
        np.where(finite, confidence, np.nan),
    )


def cents_to_shruti(
    cents: float,
    table: ShrutiTable | None = None,
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> ShrutiNote:
    """Place one measured pitch on the shruti grid.

    Args:
        cents: Signed cents above Sa, unfolded, so the octave is recoverable.
        table: Tuning table; defaults to the packaged one.
        start_s: Note start time.
        end_s: Note end time.

    Returns:
        The nearest shruti with its signed deviation and the confidence of the
        match.

    Raises:
        ValueError: If ``cents`` is not finite.
    """
    if not np.isfinite(cents):
        raise ValueError(f"cents must be finite, got {cents!r}")
    tbl = table or load_shruti_table()
    idx, deviation, confidence = nearest_shruti(cents, tbl)
    i = int(idx[0])
    position: Shruti = tbl.shrutis[i]
    folded, _ = fold_to_octave(np.array([cents]))
    neighbours = _neighbour_gap(tbl, i)
    return ShrutiNote(
        index=i,
        swara=position.swar,
        variant=position.variant,
        deviation_cents=float(deviation[0]),
        cents=float(folded[0]),
        absolute_cents=float(cents),
        octave=octave_for(cents, tbl),
        confidence=float(confidence[0]),
        clarity_cents=neighbours,
        start_s=float(start_s),
        end_s=float(end_s),
    )


def _neighbour_gap(table: ShrutiTable, index: int) -> float:
    """Distance from a shruti to the nearest point where its match would flip.

    Args:
        table: The tuning table.
        index: Index of the shruti.

    Returns:
        Half the smaller of the gaps to the neighbours on either side, in
        cents. Half, because the match flips at the midpoint of a gap, and the
        *smaller* gap, because that is the one that flips first: Ga-high sits
        22 cents above Ga-low but 90 below Ma, so a note 11 cents below it is
        already ambiguous while one 45 below is not.
    """
    c = table.shruti_cents
    n = c.size
    if n < 2:
        return float(OCTAVE_CENTS / 2)
    lo = c[index - 1] if index > 0 else c[n - 1] - OCTAVE_CENTS
    hi = c[index + 1] if index < n - 1 else c[0] + OCTAVE_CENTS
    here = c[index]
    return float(min(here - lo, hi - here) / 2.0)


def notes_to_shrutis(
    notes: list[SwaraNote],
    table: ShrutiTable | None = None,
) -> list[ShrutiNote]:
    """Re-place swara notes on the shruti grid.

    Args:
        notes: The swara notes, in time order.
        table: Tuning table; defaults to the packaged one.

    Returns:
        The same notes, each on its nearest shruti with its deviation.
    """
    tbl = table or load_shruti_table()
    return [cents_to_shruti(n.absolute_cents, tbl, n.start_s, n.end_s) for n in notes]


def format_shruti_line(notes: list[ShrutiNote], with_deviation: bool = True) -> str:
    """Render shruti matches as a single line of text.

    Args:
        notes: The matches, in time order.
        with_deviation: Whether to include the signed deviation.

    Returns:
        The line, empty if there are no notes.
    """
    if with_deviation:
        return " ".join(n.label_with_deviation for n in notes)
    return " ".join(n.label for n in notes)


def shruti_statistics(notes: list[ShrutiNote]) -> dict:
    """Summarise a set of shruti matches.

    Args:
        notes: The matches, in time order.

    Returns:
        Dict with the mean and worst absolute deviation, the number of matches
        below a given clarity, and the per-shruti histogram.
    """
    if not notes:
        return {
            "count": 0, "mean_abs_deviation_cents": None, "max_abs_deviation_cents": None,
            "unclear_count": 0, "histogram": {},
        }
    deviations = np.array([abs(n.deviation_cents) for n in notes], dtype=np.float64)
    confidences = np.array([n.confidence for n in notes], dtype=np.float64)
    histogram: dict[str, int] = {}
    for n in notes:
        histogram[n.label] = histogram.get(n.label, 0) + 1
    return {
        "count": len(notes),
        "mean_abs_deviation_cents": round(float(deviations.mean()), 2),
        "max_abs_deviation_cents": round(float(deviations.max()), 2),
        "unclear_count": int(np.count_nonzero(confidences < 0.5)),
        "histogram": dict(sorted(histogram.items())),
    }
