"""Tests for pure cents arithmetic (no audio, no heavy deps)."""

from __future__ import annotations

import numpy as np
import pytest

from swaras.cents import (
    cents_between,
    cents_to_hz,
    fold_octave,
    fold_with_octave,
    hz_to_cents,
    nearest_index,
    signed_residual,
)


def test_hz_cents_roundtrip() -> None:
    for hz in (55.0, 220.0, 261.63, 1046.5):
        assert float(cents_to_hz(hz_to_cents(hz, 261.63), 261.63)) == pytest.approx(hz, rel=1e-12)


def test_hz_to_cents_of_reference_is_zero() -> None:
    assert float(hz_to_cents(261.63, 261.63)) == pytest.approx(0.0)


def test_octave_is_1200_cents() -> None:
    # A perfect octave: C3 -> C4, i.e. exactly double the frequency.
    assert float(hz_to_cents(2 * 261.63, 261.63)) == pytest.approx(1200.0, rel=1e-9)


def test_cents_between_octave() -> None:
    assert float(cents_between(110.0, 440.0)) == pytest.approx(2400.0, rel=1e-9)


def test_fold_octave_into_unit_range() -> None:
    out = fold_octave(np.array([-5.0, 0.0, 700.0, 1200.0, 1250.0, 2400.0]))
    assert np.all(out >= 0.0)
    assert np.all(out < 1200.0)
    np.testing.assert_allclose(out, [1195.0, 0.0, 700.0, 0.0, 50.0, 0.0], atol=1e-9)


def test_fold_with_octave_tracks_index() -> None:
    folded, octave = fold_with_octave(np.array([-5.0, 0.0, 700.0, 1300.0, 2500.0]))
    np.testing.assert_allclose(folded, [1195.0, 0.0, 700.0, 100.0, 100.0], atol=1e-9)
    assert octave.tolist() == [-1, 0, 0, 1, 2]


def test_fold_with_octave_is_consistent_with_fold_octave() -> None:
    cents = np.linspace(-2400, 3600, 500)
    folded, octave = fold_with_octave(cents)
    np.testing.assert_allclose(folded, fold_octave(cents), atol=1e-9)
    np.testing.assert_allclose(folded + octave * 1200, cents, atol=1e-9)


def test_nearest_index() -> None:
    targets = np.array([0.0, 100.0, 200.0])
    assert nearest_index(np.array([-10.0, 90.0, 190.0, 250.0]), targets).tolist() == [0, 1, 2, 2]


def test_nearest_index_accepts_scalar() -> None:
    assert nearest_index(95.0, np.array([0.0, 100.0])).tolist() == [1]


def test_signed_residual() -> None:
    np.testing.assert_allclose(signed_residual(np.array([10.0, 20.0]), 15.0), [-5.0, 5.0])
