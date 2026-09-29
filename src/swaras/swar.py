"""Stage 3b: cents to one of the 12 swara positions (Level A).

The 12 positions are the ones named in the specification -- Sa, komal Re, Re,
komal Ga, Ga, Ma, tivra Ma, Pa, komal Dha, Dha, komal Ni, Ni -- and their
nominal cents come from the tuning table, so the musicology stays in data.

This module never claims a note is a hard label. Every mapping returns how far
the note sits from the position it was assigned, because a sung pitch is a
measurement and 22-shruti practice does not have the note exactly on a
position anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cents import OCTAVE_CENTS
from .normalize import fold_cents
from .normalize import Octave
from .tuning import ShrutiTable, Swara, load_shruti_table


@dataclass(frozen=True)
class SwaraNote:
    """A note assigned to one of the 12 swara positions.

    Attributes:
        swara: The swara position name, e.g. ``"Re komal"``.
        short: Compact symbol, e.g. ``"r"``.
        octave: Register relative to the detected Sa.
        cents: Measured pitch in cents above Sa, folded into one octave.
        deviation_cents: Distance from the position's nominal cents, signed.
            Negative means the note is sung flatter than nominal.
        absolute_cents: The unfolded measurement, used for the shruti layer.
        confidence: Mean frame confidence over the note.
        start_s: Note start time.
        end_s: Note end time.
    """

    swara: str
    short: str
    octave: Octave
    cents: float
    deviation_cents: float
    absolute_cents: float
    confidence: float
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Note length in seconds."""
        return max(0.0, self.end_s - self.start_s)

    @property
    def label(self) -> str:
        """Text notation with the octave marker, e.g. ``"Ni."`` or ``"Sa'"``."""
        return f"{self.short}{self.octave.marker}"

    @property
    def label_long(self) -> str:
        """Readable name with the register, e.g. ``"Ni (taar)"``."""
        name = self.swara if self.octave == Octave.MADHYA else f"{self.swara} ({self.octave.name_short})"
        return name

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "swar": self.swara,
            "short": self.short,
            "octave": int(self.octave),
            "octave_name": self.octave.name_short,
            "label": self.label,
            "cents": round(self.cents, 1),
            "absolute_cents": round(self.absolute_cents, 1),
            "deviation_cents": round(self.deviation_cents, 1),
            "confidence": round(self.confidence, 3),
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
        }


def nearest_swar(
    cents: np.ndarray | float,
    table: ShrutiTable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest of the 12 swara positions for each cents value.

    The search is circular, so a note at 1195 cents is Sa one cent below, not
    Ni 1095 cents away. Getting that wrong would put every Sa' in the wrong
    octave.

    Args:
        cents: Cents above Sa, any magnitude; ``nan`` passes through.
        table: Tuning table; defaults to the packaged one.

    Returns:
        Tuple ``(indices, deviations)`` into the table's swara list, and the
        signed cents from each position.
    """
    tbl = table or load_shruti_table()
    targets = tbl.swara_cents
    c = np.atleast_1d(np.asarray(cents, dtype=np.float64))
    folded = fold_cents(c)
    finite = np.isfinite(folded)

    safe = np.where(finite, folded, 0.0)
    # Circular distance: compare against every position and its octave
    # neighbours, then take the closest.
    deltas = np.abs(safe[:, None] - targets[None, :])
    for shift in (-OCTAVE_CENTS, OCTAVE_CENTS):
        deltas = np.minimum(deltas, np.abs(safe[:, None] - (targets[None, :] + shift)))
    idx = np.argmin(deltas, axis=1)
    # The signed deviation must be measured the short way round the octave. A
    # note sung 8 cents *below* Sa folds to 1192 cents, which really is 1192
    # cents above the Sa position, but reporting that reads as a wildly sharp Sa
    # when the performer was flat. Wrapping gives the -8 cents actually sung.
    deviation = safe - targets[idx]
    deviation = np.where(deviation > OCTAVE_CENTS / 2, deviation - OCTAVE_CENTS, deviation)
    deviation = np.where(deviation < -OCTAVE_CENTS / 2, deviation + OCTAVE_CENTS, deviation)
    return np.where(finite, idx, -1), np.where(finite, deviation, np.nan)


def octave_for(cents: float, table: ShrutiTable | None = None) -> Octave:
    """Determine the register of an unfolded cents value relative to Sa.

    The naive rule, rounding to the nearest octave boundary, is wrong here
    because the 12 swara positions are not evenly spaced. Ni sits at 1099
    cents, only 101 below the octave, so ``floor(1100 / 1200 + 0.5)`` reports
    madhya Ni as taar Sa'. A pitch therefore belongs to a higher octave only
    when the *nearest swara position* is Sa itself; every other position fixes
    the octave, because Sa is the only swara that occurs at both ends of the
    octave.

    Args:
        cents: Signed cents above Sa, unfolded.
        table: Tuning table; defaults to the packaged one.

    Returns:
        The register.
    """
    if not np.isfinite(cents):
        return Octave.MADHYA
    tbl = table or load_shruti_table()
    idx, _ = nearest_swar(cents, tbl)
    sa_name = tbl.swaras[0].name
    if tbl.swaras[int(idx[0])].name == sa_name:
        # A Sa: its register is how many octaves above the reference Sa.
        value = int(np.floor(cents / OCTAVE_CENTS + 0.5))
    else:
        value = int(np.floor(cents / OCTAVE_CENTS))
    # Clamp rather than raise. Octave names two octaves down and two up, but a
    # pitch five octaves from Sa should still be reportable: the shruti layer
    # is a measurement, and crashing on out-of-range input loses it.
    return Octave(max(int(Octave.MANDRA_2), min(int(Octave.TAAR_2), value)))


def cents_to_swar(
    cents: float,
    table: ShrutiTable | None = None,
    confidence: float = 1.0,
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> SwaraNote:
    """Map one measured pitch to a swara position.

    Args:
        cents: Signed cents above Sa, unfolded, so the octave is recoverable.
        table: Tuning table; defaults to the packaged one.
        confidence: Note confidence, carried through unchanged.
        start_s: Note start time.
        end_s: Note end time.

    Returns:
        The assigned swara with its deviation.

    Raises:
        ValueError: If ``cents`` is not finite.
    """
    if not np.isfinite(cents):
        raise ValueError(f"cents must be finite, got {cents!r}")
    tbl = table or load_shruti_table()
    idx, deviation = nearest_swar(cents, tbl)
    position = tbl.swaras[int(idx[0])]
    return SwaraNote(
        swara=position.name,
        short=position.short,
        octave=octave_for(cents, tbl),
        cents=float(fold_cents(np.array([cents]))[0]),
        deviation_cents=float(deviation[0]),
        absolute_cents=float(cents),
        confidence=float(confidence),
        start_s=float(start_s),
        end_s=float(end_s),
    )


def segment_to_swaras(
    segments: list,
    table: ShrutiTable | None = None,
    include_transitions: bool = False,
) -> list[SwaraNote]:
    """Map segmented notes to swaras.

    Transitions are skipped by default, which implements the rule that a meend
    is identified by its target, not sung as a note in its own right. With
    ``include_transitions`` they are mapped too, each using the pitch it
    arrives at.

    Args:
        segments: Segments from :mod:`swaras.segment`, in time order.
        table: Tuning table; defaults to the packaged one.
        include_transitions: Whether to map transition segments as well.

    Returns:
        The swara notes in time order.
    """
    tbl = table or load_shruti_table()
    notes: list[SwaraNote] = []
    for seg in segments:
        kind = getattr(seg.kind, "value", seg.kind)
        if kind == "note" or (include_transitions and kind == "transition"):
            if np.isfinite(seg.cents):
                notes.append(
                    cents_to_swar(
                        seg.cents, tbl, seg.confidence, seg.start_s, seg.end_s
                    )
                )
    return notes
