"""Quantitative Stage 1 accuracy: how closely do the trackers follow known F0?

These tests compare each detector against the ground-truth instantaneous
frequency of synthetic signals, and report the error distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from swaras._essentia import essentia_available
from swaras.cents import cents_between
from swaras.pitch import PyinPitchDetector, get_detector
from tests import synthetic

SR = 22_050
requires_essentia = pytest.mark.skipif(not essentia_available(), reason="essentia not installed")


def sweep_error(contour, y: np.ndarray, frame_length: int, hop: int) -> dict:
    """Compare a contour against the true sweep frequency, frame by frame.

    Both sweep endpoints must match ``synthetic.sine_sweep``'s defaults.

    Args:
        contour: Estimated contour.
        y: The signal, whose length defines the time axis.
        frame_length: Analysis window in samples.
        hop: Hop size in samples.

    Returns:
        Dict with median and 90th-percentile absolute error in cents, plus
        the voiced fraction.
    """
    n = min(y.size, contour.n_frames)
    # A frame reports the pitch near its centre, so sample the truth there
    # rather than at the frame start.
    centres = (np.arange(n) * hop + frame_length / 2.0) / SR
    truth = np.interp(
        centres,
        np.linspace(0.0, y.size / SR, y.size),
        np.geomspace(80.0, 900.0, y.size),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(cents_between(truth, contour.f0[:n]))
    voiced = contour.voiced[:n]
    err = err[voiced & np.isfinite(err)]
    return {
        "median": float(np.median(err)) if err.size else float("nan"),
        "p90": float(np.percentile(err, 90)) if err.size else float("nan"),
        "voiced": float(contour.voiced_fraction),
    }


def test_pyin_sweep_accuracy() -> None:
    y = synthetic.sine_sweep(80.0, 900.0, 6.0, SR)
    contour = PyinPitchDetector().estimate(y, SR)
    stats = sweep_error(contour, y, 1024, 512)
    assert stats["median"] < 20.0, stats
    assert stats["p90"] < 40.0, stats
    assert stats["voiced"] > 0.95, stats


@requires_essentia
def test_melodia_sweep_accuracy() -> None:
    y = synthetic.sine_sweep(80.0, 900.0, 6.0, SR)
    contour = get_detector("melodia").estimate(y, SR)
    stats = sweep_error(contour, y, 1024, 512)
    assert stats["median"] < 20.0, stats
    assert stats["p90"] < 50.0, stats


@pytest.mark.parametrize(
    "f0,tolerance",
    [
        (98.0, 12.0),      # pYIN bias peaks around +9.2 cents here
        (110.0, 12.0),
        (146.83, 12.0),
        (196.0, 12.0),
        (261.63, 7.0),     # bias falls through zero near 290 Hz
        (329.63, 4.0),
        (440.0, 4.0),
    ],
)
def test_pyin_finds_fundamental_across_the_range(f0: float, tolerance: float) -> None:
    """A voice-range sine must be tracked within pYIN's known bias budget.

    pYIN's error on a steady tone is systematic, not random: about +9 cents
    below 250 Hz and about -0.8 cents above 293 Hz. The tolerance is set just
    above the measured bias so the test fails if accuracy degrades, without
    demanding precision the algorithm does not have. Melodia is the accurate
    tracker here; see the Melodia test below.
    """
    y = synthetic.sine_tone(np.full(int(1.0 * SR), f0), SR)
    contour = PyinPitchDetector().estimate(y, SR)
    measured = float(np.median(contour.voiced_f0()))
    assert abs(float(cents_between(f0, measured))) < tolerance, (f0, measured)


@pytest.mark.parametrize("f0", [110.0, 220.0, 440.0])
@requires_essentia
def test_melodia_finds_fundamental_across_the_range(f0: float) -> None:
    y = synthetic.sine_tone(np.full(int(1.0 * SR), f0), SR)
    contour = get_detector("melodia").estimate(y, SR)
    measured = float(np.median(contour.voiced_f0()))
    assert abs(float(cents_between(f0, measured))) < 10.0, (f0, measured)


def test_pyin_survives_vibrato() -> None:
    """Vibrato must widen the instantaneous track but centre on the note."""
    for f0 in (196.0, 261.63, 392.0):
        y = synthetic.note(f0, 1.2, SR, vibrato_rate_hz=5.5, vibrato_cents=40.0)
        contour = PyinPitchDetector().estimate(y, SR)
        f0_track = contour.voiced_f0()
        assert abs(float(cents_between(f0, np.median(f0_track)))) < 20.0, f0
        # The contour should still move, i.e. vibrato is not smoothed away.
        assert float(np.ptp(cents_between(np.median(f0_track), f0_track))) > 15.0, f0


def test_pyin_tracks_a_glide() -> None:
    """A portamento from Sa to Sa' should be followed closely in cents.

    pYIN measures pitch over a finite window, so a fast glissando is reported
    with a lag that grows with speed. At 2048-sample windows this test showed
    ~58 cents of median error; the 1024-sample default in
    :data:`swaras.config.N_FFT` roughly halves it. The bound is deliberately
    loose, since the exact lag depends on glide speed.
    """
    n = int(2.0 * SR)
    f_track = synthetic.add_glide(n, 261.63, 523.25, SR, shape="exponential")
    y = synthetic.sine_tone(f_track, SR)
    contour = PyinPitchDetector().estimate(y, SR)
    # Compare against the truth at each frame's own time. Building the
    # expectation from the voiced values alone would misalign the two
    # sequences, because dropping unvoiced frames compresses the index.
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(
            np.asarray(cents_between(
                np.interp(contour.times, np.linspace(0.0, n / SR, n), f_track), contour.f0
            ))
        )
    m = contour.voiced & np.isfinite(err)
    err = err[m]
    err = err[err.size // 10 : -(err.size // 10)]  # ignore the window edges
    assert float(np.median(err)) < 40.0, float(np.median(err))
