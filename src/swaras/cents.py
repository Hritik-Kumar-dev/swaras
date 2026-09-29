"""Shared helpers for pitch math and cent arithmetic.

Kept separate from :mod:`swaras.pitch` so that later stages (normalize, swar,
shruti) can depend on the arithmetic without pulling in librosa/Essentia.
"""

from __future__ import annotations

import numpy as np

#: Cents in one octave.
OCTAVE_CENTS = 1200.0


def hz_to_cents(f: np.ndarray | float, ref_hz: float) -> np.ndarray | float:
    """Convert a frequency to cents relative to a reference.

    Args:
        f: Frequency in Hz (scalar or array).
        ref_hz: Reference frequency in Hz.

    Returns:
        Cents above ``ref_hz``; negative below it.
    """
    return 1200.0 * np.log2(np.asarray(f, dtype=np.float64) / float(ref_hz))


def cents_to_hz(c: np.ndarray | float, ref_hz: float) -> np.ndarray | float:
    """Convert cents relative to a reference back to Hz."""
    return float(ref_hz) * 2.0 ** (np.asarray(c, dtype=np.float64) / 1200.0)


def cents_between(f1: np.ndarray | float, f2: np.ndarray | float) -> np.ndarray | float:
    """Signed interval in cents from ``f1`` to ``f2``.

    Unvoiced frames (frequency ``0``) would produce ``log2(0)``, so they are
    passed through as ``nan``. Callers comparing contours should mask
    unvoiced frames rather than treating the result as a real interval.
    """
    a = np.asarray(f1, dtype=np.float64)
    b = np.asarray(f2, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(b / a)


def fold_octave(cents: np.ndarray | float) -> np.ndarray | float:
    """Fold a cents value into the half-open interval ``[0, 1200)``.

    Octave identity is deliberately *not* discarded: callers that need it
    should use :func:`fold_with_octave`, which returns the octave index
    alongside the folded value.

    Args:
        cents: Interval in cents, any magnitude.

    Returns:
        The same interval reduced into one octave.
    """
    return np.mod(cents, OCTAVE_CENTS)


def fold_with_octave(cents: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray]:
    """Fold cents into one octave and report the octave index.

    The octave index is the number of complete octaves the value sits above
    the reference, computed with ``floor`` so that it is stable and
    monotonic: ``-5`` cents gives ``(-0, -1)`` and ``1195`` gives ``(1195, 0)``.

    Args:
        cents: Interval in cents relative to a reference (typically Sa).

    Returns:
        Tuple ``(folded_cents, octave)`` where ``folded_cents`` lies in
        ``[0, 1200)`` and ``octave`` is an integer array.
    """
    c = np.asarray(cents, dtype=np.float64)
    octave = np.floor(c / OCTAVE_CENTS).astype(np.int64)
    return c - octave * OCTAVE_CENTS, octave


def nearest_index(values: np.ndarray | float, targets: np.ndarray) -> np.ndarray:
    """Index of the closest target for each value.

    Args:
        values: Query values.
        targets: Sorted or unsorted candidate values; all are considered.

    Returns:
        Integer array of indices into ``targets``, one per query value.
    """
    v = np.atleast_1d(np.asarray(values, dtype=np.float64))
    t = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    return np.argmin(np.abs(v[:, None] - t[None, :]), axis=1)


def signed_residual(values: np.ndarray | float, target: np.ndarray | float) -> np.ndarray:
    """Signed difference ``values - target``."""
    return np.asarray(values, dtype=np.float64) - np.asarray(target, dtype=np.float64)
