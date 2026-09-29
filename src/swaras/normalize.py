"""Stage 4a: express F0 as cents relative to Sa, keeping octave separate.

Two operations, deliberately kept apart:

* :func:`contour_to_cents` converts a whole contour to a signed cents track
  relative to Sa, *unfolded*. Negative values sit below Sa (mandra) and values
  above 1200 sit in the taar octave.
* :func:`fold_to_octave` reduces those values into one octave while returning
  the octave index, so the swar name and the register stay separable.

Nothing is quantised here. The output is still a continuous float track, which
is what lets the segmentation stage decide note boundaries on its own terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .cents import OCTAVE_CENTS, hz_to_cents
from .pitch import PitchContour


class Octave(IntEnum):
    """Register of a note relative to the detected Sa.

    The values are signed octave indices: ``MADHYA`` is 0, ``MANDRA`` is -1 and
    ``TAAR`` is 1. Using plain integers keeps arithmetic trivial; the names
    exist for display and for the JSON output.
    """

    MANDRA = -1
    MADHYA = 0
    TAAR = 1
    TAAR_2 = 2
    MANDRA_2 = -2

    @property
    def name_short(self) -> str:
        """Roman-numeral style name, e.g. ``"taar"``."""
        return {
            -2: "mandra-2",
            -1: "mandra",
            0: "madhya",
            1: "taar",
            2: "taar-2",
        }.get(int(self), f"octave{int(self):+d}")

    @property
    def marker(self) -> str:
        """Notation marker for the octave: a dot below, none, or a tick above.

        Returns:
            ``"Ni."``-style suffix, so ``""``, ``"."`` or ``"'"``.
        """
        value = int(self)
        if value < 0:
            return "." * (-value)
        if value > 0:
            return "'" * value
        return ""


@dataclass(frozen=True)
class CentsContour:
    """A pitch contour expressed in cents relative to Sa.

    Attributes:
        times: Frame times in seconds, copied from the source contour.
        cents: Signed interval above Sa in cents. Values are ``nan`` where the
            frame is unvoiced. Negative means below Sa.
        voiced: Boolean voiced mask.
        octave: Signed octave index per frame; ``0`` where unvoiced.
        sa_hz: The Sa the conversion used, in Hz.
        source: F0 track in Hz, kept so later stages can work in Hz again.
        confidence: Per-frame confidence, carried through from the detector.
            Defaults to ones, meaning "not reported" rather than "no confidence".
    """

    times: np.ndarray
    cents: np.ndarray
    voiced: np.ndarray
    octave: np.ndarray
    sa_hz: float
    source: np.ndarray
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.times.shape[0]
        if not (self.cents.shape == self.voiced.shape == self.octave.shape == (n,)):
            raise ValueError("times, cents, voiced and octave must have the same length")
        if self.confidence is None:
            # Default via object.__setattr__: the dataclass is frozen, and a
            # default in __init__ would need the same escape hatch anyway.
            object.__setattr__(self, "confidence", np.ones(n, dtype=np.float64))
        elif self.confidence.shape != (n,):
            object.__setattr__(self, "confidence", np.ones(n, dtype=np.float64))

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.times.shape[0])

    @property
    def folded_cents(self) -> np.ndarray:
        """Cents folded into ``[0, 1200)``, with the octave removed."""
        return np.mod(np.where(self.voiced, self.cents, np.nan), OCTAVE_CENTS)

    def voiced_cents(self) -> np.ndarray:
        """Unfolded cents of voiced frames only."""
        return self.cents[self.voiced]

    def to_hz(self) -> np.ndarray:
        """The F0 track this contour was built from, in Hz."""
        return self.source

    def voicing(self) -> np.ndarray:
        """Boolean voiced mask."""
        return self.voiced


def contour_to_cents(contour: PitchContour, sa_hz: float) -> CentsContour:
    """Convert a pitch contour to cents relative to Sa, unfolded.

    Args:
        contour: Source contour, with F0 in Hz.
        sa_hz: Sa in Hz.

    Returns:
        The cents track, with unvoiced frames set to ``nan``.

    Raises:
        ValueError: If ``sa_hz`` is not a positive frequency.
    """
    hz = float(sa_hz)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError(f"sa_hz must be a positive frequency, got {sa_hz!r}")
    voiced = contour.voiced & np.isfinite(contour.f0) & (contour.f0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = np.asarray(hz_to_cents(np.where(voiced, contour.f0, np.nan), hz), dtype=np.float64)
    cents = np.where(voiced, cents, np.nan)
    # Compute the floor on a nan-free copy: casting nan to an integer is
    # undefined behaviour and NumPy warns about it. The result is masked back
    # to 0, which is a placeholder; the voiced mask is the authority.
    safe = np.where(voiced, cents, 0.0)
    octave = np.where(voiced, np.floor(safe / OCTAVE_CENTS).astype(np.int64), 0)
    return CentsContour(
        times=contour.times,
        cents=cents,
        voiced=voiced,
        octave=octave,
        sa_hz=hz,
        source=contour.f0,
        confidence=contour.confidence,
    )


def fold_to_octave(cents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold cents into one octave and return the octave index.

    Args:
        cents: Signed interval above Sa, any magnitude.

    Returns:
        Tuple ``(folded, octave)`` where ``folded`` lies in ``[0, 1200)`` and
        ``octave`` is an integer array. ``nan`` inputs yield ``nan`` outputs and
        an octave of 0.
    """
    c = np.asarray(cents, dtype=np.float64)
    finite = np.isfinite(c)
    safe = np.where(finite, c, 0.0)
    octave = np.floor(safe / OCTAVE_CENTS).astype(np.int64)
    folded = np.mod(safe, OCTAVE_CENTS)
    return np.where(finite, folded, np.nan), np.where(finite, octave, 0)


def fold_cents(cents: np.ndarray) -> np.ndarray:
    """Fold cents into ``[0, 1200)`` without keeping the octave.

    Args:
        cents: Signed interval above Sa.

    Returns:
        The folded values, ``nan`` where the input is ``nan``.
    """
    folded, _ = fold_to_octave(cents)
    return folded


def cents_to_hz(cents: np.ndarray | float, sa_hz: float) -> np.ndarray | float:
    """Convert cents above Sa back to Hz.

    Args:
        cents: Signed interval above Sa.
        sa_hz: Sa in Hz.

    Returns:
        Frequency in Hz, ``nan`` where the input is ``nan``.
    """
    return float(sa_hz) * 2.0 ** (np.asarray(cents, dtype=np.float64) / 1200.0)
