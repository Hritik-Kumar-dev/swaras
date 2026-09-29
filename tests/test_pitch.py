"""Tests for Stage 1: pitch tracking (pYIN and Essentia Melodia)."""

from __future__ import annotations

import numpy as np
import pytest

from swaras._essentia import essentia_available
from swaras.cents import cents_between
from swaras.config import PitchParams
from swaras.pitch import (
    PitchContour,
    PyinPitchDetector,
    correct_octave_jumps,
    get_detector,
    median_filter_f0,
)
from tests import synthetic

requires_essentia = pytest.mark.skipif(not essentia_available(), reason="essentia not installed")
SR = 22_050


def make_contour(f0: np.ndarray | list[float], voiced: list[bool] | None = None, hop: float = 0.023) -> PitchContour:
    """Build a PitchContour directly, for testing the pure helper functions."""
    f0 = np.asarray(f0, dtype=np.float64)
    n = f0.size
    voiced_arr = np.asarray(voiced, dtype=bool) if voiced is not None else (f0 > 0)
    return PitchContour(
        times=np.arange(n) * hop,
        f0=f0,
        voiced=voiced_arr,
        confidence=np.where(voiced_arr, 0.9, 0.0),
        hop_seconds=hop,
    )


# --- pure helper functions -------------------------------------------------


def test_median_filter_removes_single_spike() -> None:
    f0 = np.full(21, 220.0)
    f0[10] = 330.0  # a single rogue frame
    smoothed = median_filter_f0(f0, f0 > 0, width_frames=5)
    assert smoothed[10] == pytest.approx(220.0, rel=1e-6)
    # Neighbours should be untouched.
    assert smoothed[5] == pytest.approx(220.0, rel=1e-6)
    assert smoothed[15] == pytest.approx(220.0, rel=1e-6)


def test_median_filter_preserves_flat_track() -> None:
    f0 = np.full(11, 330.0)
    smoothed = median_filter_f0(f0, f0 > 0, width_frames=5)
    np.testing.assert_allclose(smoothed, f0, rtol=1e-9)


def test_median_filter_handles_unvoiced_frames() -> None:
    f0 = np.array([220.0, 0.0, 220.0, 0.0, 220.0, 0.0, 220.0])
    voiced = f0 > 0
    smoothed = median_filter_f0(f0, voiced, width_frames=5)
    # Unvoiced positions stay at 0; voiced ones are unaffected.
    assert np.all(smoothed[~voiced] == 0.0)
    np.testing.assert_allclose(smoothed[voiced], 220.0, rtol=1e-9)


def test_median_filter_shorter_than_window() -> None:
    f0 = np.array([220.0, 221.0, 222.0])
    smoothed = median_filter_f0(f0, f0 > 0, width_frames=5)
    np.testing.assert_allclose(smoothed, f0, rtol=1e-9)


def test_median_filter_scales_evenly_across_pitch() -> None:
    """A cents-domain filter should smooth a high note as much as a low one."""
    n = 21
    for base in (110.0, 880.0):
        f0 = np.full(n, base)
        f0[10] = base * 1.5
        smoothed = median_filter_f0(f0, f0 > 0, width_frames=5)
        assert abs(cents_between(smoothed[10], base)) < 1.0


def test_correct_octave_jumps_fixes_subharmonic_error() -> None:
    f0 = np.full(21, 220.0)
    f0[10] = 110.0  # octave below the true pitch
    fixed = correct_octave_jumps(f0, f0 > 0, octave_tolerance_cents=55.0)
    assert fixed[10] == pytest.approx(220.0, abs=1.0)


def test_correct_octave_jumps_fixes_double_octave_error() -> None:
    f0 = np.full(21, 220.0)
    f0[10] = 55.0  # two octaves below
    fixed = correct_octave_jumps(f0, f0 > 0, octave_tolerance_cents=55.0)
    assert abs(cents_between(fixed[10], 220.0)) < 55.0


def test_correct_octave_jumps_leaves_real_leaps_alone() -> None:
    """A genuine octave jump should survive, not be 'corrected' away."""
    f0 = np.concatenate([np.full(10, 220.0), np.full(10, 440.0)])
    fixed = correct_octave_jumps(f0, f0 > 0, octave_tolerance_cents=55.0)
    np.testing.assert_allclose(fixed, f0, rtol=1e-6)


def test_correct_octave_jumps_ignores_unvoiced() -> None:
    f0 = np.array([220.0, 220.0, 0.0, 220.0, 220.0])
    fixed = correct_octave_jumps(f0, f0 > 0)
    assert fixed[2] == 0.0


def test_contour_validates_shapes() -> None:
    with pytest.raises(ValueError):
        PitchContour(
            times=np.zeros(5), f0=np.zeros(4), voiced=np.zeros(5, bool),
            confidence=np.zeros(5), hop_seconds=0.01,
        )


def test_contour_set_f0_updates_voiced() -> None:
    c = make_contour([220.0, 0.0, 440.0])
    c.set_f0(np.array([100.0, 100.0, -5.0]))
    assert c.voiced.tolist() == [True, True, False]


def test_contour_accessors() -> None:
    c = make_contour([220.0, 0.0, 440.0, 0.0])
    assert c.n_frames == 4
    assert c.voiced_fraction == 0.5
    assert c.voiced_f0().tolist() == [220.0, 440.0]
    assert len(c.voiced_confidence()) == 2


def test_contour_smoothed_f0_is_pure() -> None:
    """smoothed_f0 must not mutate the original track."""
    c = make_contour([220.0] * 10 + [440.0] + [220.0] * 10)
    original = c.f0.copy()
    _ = c.smoothed_f0()
    np.testing.assert_allclose(c.f0, original)


# --- pYIN on synthetic audio ----------------------------------------------


def test_pyin_tracks_steady_tone() -> None:
    y = synthetic.sine_tone(np.full(SR, 220.0), SR)
    contour = PyinPitchDetector().estimate(y, SR)
    assert contour.n_frames > 0
    assert contour.voiced_fraction > 0.8
    assert np.median(contour.voiced_f0()) == pytest.approx(220.0, abs=2.0)
    assert contour.method == "pyin"


def test_pyin_tracks_a_sine_sweep() -> None:
    """A sweep should be followed closely, frame by frame."""
    duration = 4.0
    y = synthetic.sine_sweep(150.0, 800.0, duration, SR)
    contour = PyinPitchDetector().estimate(y, SR)
    n = min(y.size, contour.n_frames)
    expected = np.geomspace(150.0, 800.0, n)
    err = cents_between(expected, contour.f0[:n])
    # Ignore the very first frames (the window has not filled yet).
    central = err[10:-10]
    assert np.median(np.abs(central)) < 30.0
    assert contour.voiced_fraction > 0.9


def test_pyin_on_harmonic_tone_gives_fundamental_not_harmonic() -> None:
    y = synthetic.note(196.0, 1.0, SR, vibrato_rate_hz=0.0, n_harmonics=10)
    contour = PyinPitchDetector().estimate(y, SR)
    assert np.median(contour.voiced_f0()) == pytest.approx(196.0, abs=5.0)


def test_pyin_marks_silence_unvoiced() -> None:
    tone = synthetic.sine_tone(np.full(SR, 220.0), SR)
    y = np.concatenate([np.zeros(SR), tone, np.zeros(SR)])
    contour = PyinPitchDetector().estimate(y, SR)
    # The first and last frames sit inside the silence padding, the middle is
    # inside the steady tone.
    assert not contour.voiced[0]
    assert not contour.voiced[-1]
    assert contour.voiced[0:3].sum() == 0
    assert contour.voiced[-3:].sum() == 0
    assert contour.voiced[len(contour.voiced) // 2]


def test_pyin_handles_empty_input() -> None:
    contour = PyinPitchDetector().estimate(np.array([]), SR)
    assert contour.n_frames == 0
    assert contour.voiced_fraction == 0.0


def test_pyin_confidence_in_unit_range() -> None:
    y = synthetic.note(220.0, 0.5, SR)
    contour = PyinPitchDetector().estimate(y, SR)
    assert contour.confidence.min() >= 0.0
    assert contour.confidence.max() <= 1.0


def test_pyin_time_grid_is_frame_centres() -> None:
    y = synthetic.sine_tone(np.full(SR, 220.0), SR)
    params = PitchParams()
    contour = PyinPitchDetector(params).estimate(y, SR)
    assert contour.hop_seconds == pytest.approx(params.hop_length / SR)
    assert contour.times[0] == pytest.approx(params.frame_length / 2 / SR, rel=1e-6)


# --- detector registry -----------------------------------------------------


def test_get_detector_by_name() -> None:
    assert isinstance(get_detector("pyin"), PyinPitchDetector)
    with pytest.raises(ValueError, match="unknown pitch detector"):
        get_detector("nope")


@requires_essentia
def test_get_detector_melodia() -> None:
    from swaras.pitch import MelodiaPitchDetector

    assert isinstance(get_detector("melodia"), MelodiaPitchDetector)


# --- Essentia Melodia ------------------------------------------------------


@requires_essentia
def test_melodia_tracks_steady_tone() -> None:
    from swaras.pitch import MelodiaPitchDetector

    y = synthetic.sine_tone(np.full(SR, 220.0), SR)
    contour = MelodiaPitchDetector().estimate(y, SR)
    assert contour.method == "melodia"
    assert contour.voiced_fraction > 0.5
    assert np.median(contour.voiced_f0()) == pytest.approx(220.0, abs=3.0)


@requires_essentia
def test_melodia_tracks_sweep() -> None:
    from swaras.pitch import MelodiaPitchDetector

    y = synthetic.sine_sweep(200.0, 600.0, 4.0, SR)
    contour = MelodiaPitchDetector().estimate(y, SR)
    n = min(y.size, contour.n_frames)
    expected = np.geomspace(200.0, 600.0, n)
    err = np.abs(cents_between(expected, contour.f0[:n]))
    assert np.median(err[10:-10]) < 35.0


@requires_essentia
def test_melodia_frame_count_matches_pyin() -> None:
    """Both trackers must share a frame grid so contours can be compared."""
    from swaras.pitch import MelodiaPitchDetector

    y = synthetic.note(220.0, 1.0, SR)
    p = PyinPitchDetector().estimate(y, SR)
    m = MelodiaPitchDetector().estimate(y, SR)
    assert p.n_frames == m.n_frames
    np.testing.assert_allclose(p.times, m.times)


@requires_essentia
def test_melodia_handles_empty_input() -> None:
    from swaras.pitch import MelodiaPitchDetector

    contour = MelodiaPitchDetector().estimate(np.array([]), SR)
    assert contour.n_frames == 0


@requires_essentia
def test_melodia_finds_fundamental_of_harmonic_tone() -> None:
    from swaras.pitch import MelodiaPitchDetector

    y = synthetic.note(196.0, 1.0, SR, vibrato_rate_hz=0.0, n_harmonics=10)
    contour = MelodiaPitchDetector().estimate(y, SR)
    assert np.median(contour.voiced_f0()) == pytest.approx(196.0, abs=6.0)
