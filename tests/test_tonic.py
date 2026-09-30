"""Tests for Stage 2: tonic (Sa) detection.

The headline requirement is that a synthetic phrase of known Sa/Re/Ga/Pa is
resolved to the right Sa within ~15 cents, at several different tonics.
"""

from __future__ import annotations

import numpy as np
import pytest

from swaras._essentia import essentia_available
from swaras.cents import cents_between, hz_to_cents
from swaras.config import TonicParams
from swaras.pitch import MelodiaPitchDetector, PitchContour, PyinPitchDetector, get_detector
from swaras.tonic import (
    TonicCandidate,
    TonicResult,
    cents_histogram,
    collapse_octave_equivalents,
    lowest_supported_cents,
    detect_tonic,
    detect_tonic_essentia,
    detect_tonic_histogram,
    local_maxima,
    manual_tonic,
    _octave_equivalent_delta,
    merge_essentia,
    sample_histogram,
    score_candidate,
)
from tests import synthetic

SR = 22_050
requires_essentia = pytest.mark.skipif(not essentia_available(), reason="essentia not installed")

#: The tonics the task asks us to test.
TEST_TONICS = [220.0, 261.63, 293.66]

#: Sa-Re-Ga-Pa in cents, the phrase whose tonic we expect to recover.
SA_RE_GA_PA = [0, 200, 400, 700]

#: Accuracy target from the task specification.
TOLERANCE_CENTS = 15.0


def make_contour(f0: list[float], voiced: list[bool] | None = None) -> PitchContour:
    """Build a contour from a list of Hz values, for pure-function tests."""
    arr = np.asarray(f0, dtype=np.float64)
    n = arr.size
    v = np.asarray(voiced, dtype=bool) if voiced is not None else arr > 0
    return PitchContour(
        times=np.arange(n) * 0.023,
        f0=arr,
        voiced=v,
        confidence=np.where(v, 0.9, 0.0),
        hop_seconds=0.023,
    )


def render_phrase(tonic_hz: float, cents: list[int], repeat: int = 3, **kw) -> np.ndarray:
    """Render a repeated phrase of scale degrees at a given tonic."""
    freqs = [synthetic.cents_to_hz(c, tonic_hz) for c in cents] * repeat
    return synthetic.sequence(freqs, note_duration_s=0.4, vibrato_cents=0.0, **kw)


# --- histogram construction ------------------------------------------------


def test_cents_histogram_peaks_at_the_sung_pitches() -> None:
    f0 = np.array([220.0] * 50 + [293.66] * 50)
    centers, density = cents_histogram(f0, np.ones(100, bool))
    # Cluster the bins above half maximum into peaks, then read off the two
    # sung pitches. Comparing "the top 200 bins" directly is not a peak test:
    # smoothing spreads each peak over many neighbouring bins.
    strong = np.flatnonzero(density > 0.5 * density.max())
    groups = np.split(strong, np.flatnonzero(np.diff(strong) > 1) + 1)
    found = [27.5 * 2 ** (float(centers[g[np.argmax(density[g])]]) / 1200.0) for g in groups]
    assert len(found) == 2, found
    assert abs(float(cents_between(found[0], found[1]))) == pytest.approx(500, abs=2)


def test_cents_histogram_range_covers_its_own_data() -> None:
    """The lowest sung pitch must fall inside the interpolated range.

    Regression test: np.histogram places the first bin centre half a bin above
    the lower edge, so a range built exactly from the data used to put the
    lowest note outside the range and score it zero.
    """
    f0 = np.array([261.63] * 100)
    centers, density = cents_histogram(f0, np.ones(100, bool))
    lo_cents = float(hz_to_cents(261.63, 27.5))
    assert centers[0] < lo_cents < centers[-1]
    assert sample_histogram(centers, density, [lo_cents])[0] > 0.9


def test_lowest_supported_cents_finds_the_lowest_sung_note() -> None:
    f0 = np.array([523.25] * 50 + [261.63] * 50 + [392.0] * 50)
    centers, density = cents_histogram(f0, np.ones(150, bool))
    lowest = lowest_supported_cents(centers, density, 0.3)
    assert abs(float(cents_between(261.63, 27.5 * 2 ** (lowest / 1200.0)))) < 5.0


def test_lowest_supported_cents_on_empty_histogram() -> None:
    assert np.isnan(lowest_supported_cents(np.zeros(0), np.zeros(0), 0.3))


def test_lowest_note_prior_breaks_scale_run_ties() -> None:
    """With the prior off, Sa and Re tie on a full scale run; on, Sa wins."""
    scale = [0, 200, 400, 500, 700, 900, 1100, 900, 700, 500, 400, 200, 0]
    freqs = [synthetic.cents_to_hz(c, 220.0) for c in scale] * 2
    y = synthetic.sequence(freqs, note_duration_s=0.4, vibrato_cents=0.0)
    contour = PyinPitchDetector().estimate(y, SR)

    without = detect_tonic_histogram(contour, TonicParams(use_essentia=False, w_lowest=0.0))
    with_prior = detect_tonic_histogram(contour, TonicParams(use_essentia=False, w_lowest=0.12))

    # Without the prior the peak terms genuinely cannot separate them.
    sa_err = abs(float(cents_between(220.0, without.hz)))
    assert sa_err > TOLERANCE_CENTS, "the no-prior case is expected to be wrong"
    # With it, the lowest note wins.
    assert abs(float(cents_between(220.0, with_prior.hz))) <= TOLERANCE_CENTS


def test_lowest_note_prior_does_not_disturb_a_plain_phrase() -> None:
    """The prior must not override clear Sa/Pa evidence."""
    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False, w_lowest=0.12))
    assert abs(float(cents_between(261.63, result.hz))) <= TOLERANCE_CENTS


def test_cents_histogram_is_normalised() -> None:
    f0 = np.concatenate([np.full(10, 220.0), np.full(500, 440.0)])
    _, density = cents_histogram(f0, np.ones(510, bool))
    assert density.max() == pytest.approx(1.0)


def test_cents_histogram_ignores_unvoiced() -> None:
    f0 = np.array([220.0] * 20 + [0.0] * 80 + [440.0] * 20)
    voiced = f0 > 0
    _, dense = cents_histogram(f0, voiced)
    _, all_frames = cents_histogram(f0, np.ones(120, bool))
    np.testing.assert_allclose(dense, all_frames)


def test_cents_histogram_empty_when_unvoiced() -> None:
    centers, density = cents_histogram(np.zeros(50), np.zeros(50, bool))
    assert centers.size == 0
    assert density.size == 0


def test_sample_histogram_zero_outside_range() -> None:
    centers, density = cents_histogram(np.full(100, 220.0), np.ones(100, bool))
    inside = sample_histogram(centers, density, [float(centers[len(centers) // 2])])
    below = sample_histogram(centers, density, [float(centers[0]) - 500.0])
    assert inside[0] > 0.5
    assert below[0] == 0.0


# --- scoring ---------------------------------------------------------------


def test_score_prefers_sa_over_a_pa_like_candidate() -> None:
    """A real tonic has a fifth above it; a scale degree usually does not."""
    # A phrase spanning Sa, Re, Ga, Pa.
    f0 = []
    for hz in (261.63, 293.66, 329.63, 392.0):
        f0 += [hz] * 200
    centers, density = cents_histogram(np.array(f0), np.ones(len(f0), bool))
    no_prior = TonicParams(w_lowest=0.0)

    sa = float(hz_to_cents(261.63, 27.5))
    re_ = float(hz_to_cents(293.66, 27.5))
    pa = float(hz_to_cents(392.0, 27.5))
    score_sa, _, pa_peak_sa, _ = score_candidate(sa, centers, density, no_prior)
    score_re, _, _, _ = score_candidate(re_, centers, density, no_prior)
    score_pa, _, _, _ = score_candidate(pa, centers, density, no_prior)

    assert pa_peak_sa > 0.5, "Sa should see the Pa peak"
    assert score_sa > score_re
    assert score_sa > score_pa


def test_score_of_silence_is_zero() -> None:
    centers, density = cents_histogram(np.full(200, 261.63), np.ones(200, bool))
    score, sa_peak, pa_peak, oct_peak = score_candidate(0.0, centers, density, TonicParams(w_lowest=0.0))
    assert sa_peak == 0.0 and pa_peak == 0.0 and oct_peak == 0.0 and score == 0.0


def test_score_weights_are_configurable() -> None:
    f0 = []
    for hz in (261.63, 392.0):
        f0 += [hz] * 200
    centers, density = cents_histogram(np.array(f0), np.ones(len(f0), bool))
    sa = float(hz_to_cents(261.63, 27.5))
    no_prior = TonicParams(w_lowest=0.0)
    heavy = score_candidate(sa, centers, density, TonicParams(w_pa=2.0, w_lowest=0.0))[0]
    light = score_candidate(sa, centers, density, TonicParams(w_pa=0.0, w_lowest=0.0))[0]
    assert heavy > light
    # The prior is additive and must move the score in a known direction.
    with_prior = score_candidate(sa, centers, density, TonicParams(w_lowest=0.5))[0]
    assert with_prior > light


# --- local maxima ----------------------------------------------------------


def test_local_maxima_finds_peaks() -> None:
    values = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    assert local_maxima(values).tolist() == [5, 3, 1]


def test_local_maxima_on_short_and_flat_arrays() -> None:
    assert local_maxima(np.zeros(0)).size == 0
    assert local_maxima(np.array([1.0])).tolist() == [0]
    assert local_maxima(np.ones(10)).tolist() == [0]  # arbitrary but non-empty


def test_local_maxima_respects_min_separation() -> None:
    values = np.array([0.0, 1.0, 0.9, 1.2, 0.0])
    # Peaks at indices 1 and 3, only 2 apart, so a separation of 3 must keep
    # just the stronger one while a separation of 2 keeps both.
    assert local_maxima(values, min_separation=3).tolist() == [3]
    assert sorted(local_maxima(values, min_separation=2).tolist()) == [1, 3]


# --- octave collapsing -----------------------------------------------------


def candidate(hz: float, score: float) -> TonicCandidate:
    """Build a bare candidate for collapse tests."""
    return TonicCandidate(
        hz=hz, cents=float(cents_between(27.5, hz)), score=score, confidence=score,
        sa_peak=1.0, pa_peak=0.0, octave_peak=0.0,
    )


def test_collapse_prefers_lower_octave() -> None:
    low = candidate(261.63, 1.0)
    high = candidate(523.25, 0.85)  # an octave up, within the collapse ratio
    kept = collapse_octave_equivalents([high, low], ratio=0.9)
    assert [round(c.hz) for c in kept] == [262]


def test_collapse_respects_the_ratio_boundary() -> None:
    """A candidate just inside the ratio is kept; just outside, collapsed."""
    low = candidate(261.63, 1.0)
    kept = collapse_octave_equivalents([candidate(523.25, 0.91), low], ratio=0.9)
    assert len(kept) == 2, "0.91 > 0.9 should survive as clearly stronger"
    kept = collapse_octave_equivalents([candidate(523.25, 0.89), low], ratio=0.9)
    assert len(kept) == 1, "0.89 <= 0.9 should collapse"


def test_collapse_keeps_clearly_stronger_upper_octave() -> None:
    low = candidate(261.63, 1.0)
    high = candidate(523.25, 1.5)
    kept = collapse_octave_equivalents([low, high], ratio=0.9)
    assert round(kept[0].hz) == 523


def test_collapse_leaves_unrelated_candidates() -> None:
    a = candidate(261.63, 1.0)
    b = candidate(392.0, 0.9)  # a fifth, not an octave
    assert len(collapse_octave_equivalents([a, b], ratio=0.9)) == 2


def test_collapse_on_empty_list() -> None:
    assert collapse_octave_equivalents([]) == []


# --- the headline requirement: synthetic phrases ---------------------------


@pytest.mark.parametrize("tonic", TEST_TONICS)
def test_tonic_recovered_within_15_cents_pyin(tonic: float) -> None:
    """Sa/Re/Ga/Pa at several tonics: Sa recovered within 15 cents."""
    y = render_phrase(tonic, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    assert result.is_valid, result
    err = float(cents_between(tonic, result.hz))
    assert abs(err) <= TOLERANCE_CENTS, f"tonic {tonic} -> {result.hz:.2f} Hz ({err:+.1f} cents)"


@pytest.mark.parametrize("tonic", TEST_TONICS)
@requires_essentia
def test_tonic_recovered_within_15_cents_melodia(tonic: float) -> None:
    y = render_phrase(tonic, SA_RE_GA_PA)
    contour = get_detector("melodia").estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    assert result.is_valid, result
    err = float(cents_between(tonic, result.hz))
    assert abs(err) <= TOLERANCE_CENTS, f"tonic {tonic} -> {result.hz:.2f} Hz ({err:+.1f} cents)"


def test_tonic_recovered_with_ascending_arpeggio() -> None:
    """A Sa-Pa-Sa'-Ni' shape, which strongly corroborates the tonic."""
    y = render_phrase(261.63, [0, 700, 1200, 1900], repeat=3)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    err = float(cents_between(261.63, result.hz))
    assert abs(err) <= TOLERANCE_CENTS, f"got {result.hz:.2f} Hz ({err:+.1f} cents)"


def test_tonic_recovered_with_dense_scale_run() -> None:
    """A full ascending and descending scale from Sa to Ni."""
    scale = [0, 200, 400, 500, 700, 900, 1100, 900, 700, 500, 400, 200, 0]
    y = render_phrase(220.0, scale, repeat=2)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    err = float(cents_between(220.0, result.hz))
    assert abs(err) <= TOLERANCE_CENTS, f"got {result.hz:.2f} Hz ({err:+.1f} cents)"


def test_tonic_tracks_a_real_transposition_range() -> None:
    """Sweep the tonic across an octave and check each is recovered."""
    for tonic in [130.81, 174.61, 196.0, 246.94, 329.63, 392.0, 493.88]:
        y = render_phrase(tonic, SA_RE_GA_PA, repeat=2)
        contour = PyinPitchDetector().estimate(y, SR)
        result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
        err = float(cents_between(tonic, result.hz))
        assert abs(err) <= TOLERANCE_CENTS, f"tonic {tonic} -> {result.hz:.2f} Hz ({err:+.1f} cents)"


# --- result plumbing -------------------------------------------------------


def test_alternatives_are_distinct_pitches_not_one_peak() -> None:
    """Reported alternatives must be different pitch classes.

    Without suppression the top 3 all land on the same broad peak a few cents
    apart, which tells the user nothing they could act on.
    """
    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False), top_k=3)
    hz = [c.hz for c in result.candidates]
    for i in range(len(hz)):
        for j in range(i + 1, len(hz)):
            gap = abs(float(cents_between(hz[i], hz[j])))
            assert gap > 45.0, f"candidates {hz[i]:.1f} and {hz[j]:.1f} Hz are only {gap:.0f}c apart"


def test_result_reports_top_k_candidates() -> None:
    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False), top_k=3)
    assert len(result.candidates) == 3
    assert result.confidence == 1.0
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    confs = [c.confidence for c in result.candidates]
    assert all(0 < x <= 1.0 for x in confs)


def test_result_confidence_decreases_down_the_list() -> None:
    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    confs = [c.confidence for c in result.candidates]
    assert confs[0] >= confs[1] >= confs[2]


def test_result_to_dict_is_serialisable() -> None:
    import json

    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    payload = json.dumps(result.to_dict())
    assert "candidates" in payload
    assert result.to_dict()["hz"] == pytest.approx(261.63, abs=1.0)


def test_unvoiced_contour_gives_empty_result() -> None:
    contour = make_contour([0.0] * 100)
    result = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    assert not result.is_valid
    assert result.hz is None
    assert "no voiced frames" in (result.warning or "")


def test_detect_tonic_without_audio_runs_histogram_only() -> None:
    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic(contour, samples=None, params=TonicParams(use_essentia=False))
    assert result.is_valid
    assert result.method == "histogram"


# --- Essentia integration --------------------------------------------------


@requires_essentia
def test_essentia_returns_something_on_a_phrase() -> None:
    y = render_phrase(261.63, SA_RE_GA_PA)
    hz = detect_tonic_essentia(y.astype(np.float32), SR)
    assert hz is None or hz > 0


@requires_essentia
def test_essentia_on_drone_less_audio_is_flagged_not_trusted() -> None:
    """Essentia clamps to its search floor without a drone; we must not trust it."""
    y = render_phrase(261.63, SA_RE_GA_PA)
    contour = PyinPitchDetector().estimate(y, SR)
    result = detect_tonic(contour, samples=y.astype(np.float32), sample_rate=SR)
    assert result.is_valid
    # Whatever Essentia said, the histogram answer must still be correct.
    err = float(cents_between(261.63, result.hz))
    assert abs(err) <= TOLERANCE_CENTS, f"{result.hz:.2f} Hz ({err:+.1f} cents)"
    if result.essentia_hz is not None:
        # It should either be flagged as clamped to the search floor, or
        # reported as a disagreement. Either way it must not win.
        assert result.warning is not None, f"essentia said {result.essentia_hz} Hz with no warning"
        at_edge = result.essentia_hz <= 55.0 + 5.0
        if not at_edge:
            assert abs(float(cents_between(261.63, result.essentia_hz))) > 50.0, (
                "if essentia actually agreed, no disagreement warning is expected"
            )


def test_merge_essentia_boosts_agreeing_candidate() -> None:
    c = TonicCandidate(
        hz=261.63, cents=float(hz_to_cents(261.63, 27.5)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    merged = merge_essentia(result, 261.63, TonicParams(essentia_agree_cents=50.0))
    assert "essentia" in merged.candidates[0].methods
    assert merged.candidates[0].score > c.score


def test_merge_essentia_resorts_when_boost_promotes_a_candidate() -> None:
    """A boost that reorders the list must move the new best to the front.

    TonicResult.hz reads candidates[0], so leaving a boosted candidate further
    down the list would make the reported Sa disagree with the highest
    confidence in the same payload.
    """
    best = TonicCandidate(
        hz=261.63, cents=float(hz_to_cents(261.63, 27.5)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    # A 5% gap, which the default 1.08 boost can cross. A wider gap is
    # deliberately out of reach -- see
    # test_essentia_support_confirms_but_does_not_override.
    runner_up = TonicCandidate(
        hz=196.0, cents=float(hz_to_cents(196.0, 27.5)), score=0.95, confidence=0.95,
        sa_peak=1.0, pa_peak=0.2, octave_peak=0.0,
    )
    result = TonicResult(candidates=(best, runner_up), method="histogram")
    # Essentia agrees with the runner-up, whose boost overtakes the leader.
    merged = merge_essentia(result, 196.0, TonicParams(essentia_agree_cents=50.0))
    assert merged.hz == pytest.approx(196.0)
    assert merged.candidates[0].confidence == pytest.approx(1.0)
    assert merged.candidates[0].score >= merged.candidates[1].score
    assert set(merged.candidates[0].methods) == {"histogram", "essentia"}


def test_merge_essentia_warns_on_disagreement() -> None:
    c = TonicCandidate(
        hz=261.63, cents=float(cents_between(27.5, 261.63)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    merged = merge_essentia(result, 330.0, TonicParams(essentia_agree_cents=50.0))
    assert "essentia" not in merged.candidates[0].methods
    assert "does not match" in (merged.warning or "")


def test_merge_essentia_rejects_search_edge() -> None:
    c = TonicCandidate(
        hz=261.63, cents=float(cents_between(27.5, 261.63)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    merged = merge_essentia(result, 55.0, TonicParams())
    assert "edge of its search range" in (merged.warning or "")
    assert merged.candidates[0].methods == ("histogram",)


def test_merge_essentia_credits_an_octave_error_as_pitch_class_support() -> None:
    """An octave-off answer is evidence about the note, not about the register.

    Measured on a tanpura-style drone, Essentia named the right pitch class in
    1 case out of 8 and the right octave in 1 of 8. Comparing raw frequencies
    threw the pitch-class evidence away, which is why the khayal sample
    reported "no match" from a model that had in fact found the right note.
    """
    c = TonicCandidate(
        hz=261.63, cents=float(cents_between(27.5, 261.63)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    merged = merge_essentia(result, 130.81, TonicParams(essentia_agree_cents=50.0))
    assert "essentia-octave" in merged.candidates[0].methods
    assert "essentia" not in merged.candidates[0].methods
    assert merged.candidates[0].score > c.score
    # The octave is wrong, so the user is told it rather than quietly trusting it.
    assert "octave" in (merged.warning or "")


def test_merge_essentia_ignores_an_octave_error_when_told_to() -> None:
    c = TonicCandidate(
        hz=261.63, cents=float(cents_between(27.5, 261.63)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    params = TonicParams(essentia_agree_cents=50.0, essentia_ignore_octave=False)
    merged = merge_essentia(result, 130.81, params)
    assert "essentia-octave" not in merged.candidates[0].methods
    assert "does not match" in (merged.warning or "")


def test_merge_essentia_does_not_credit_a_fifth_as_support() -> None:
    """A perfect fifth is a different note, even though it is only 702 cents.

    Folding to pitch class must not slide into folding to the nearest
    consonant interval, or a Pa/Ma confusion would be read as agreement.
    """
    c = TonicCandidate(
        hz=261.63, cents=float(cents_between(27.5, 261.63)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.5, octave_peak=0.5,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    merged = merge_essentia(result, 392.0, TonicParams(essentia_agree_cents=50.0))
    assert merged.candidates[0].methods == ("histogram",)


def test_octave_equivalent_delta_folds_to_pitch_class() -> None:
    assert _octave_equivalent_delta(261.63, 130.81) == pytest.approx(0.0, abs=1.0)
    assert _octave_equivalent_delta(261.63, 261.63) == pytest.approx(0.0)
    # Symmetric, and reported in [0, 600] however far apart the octaves are.
    assert _octave_equivalent_delta(130.81, 261.63) == pytest.approx(
        _octave_equivalent_delta(261.63, 130.81)
    )
    assert 0.0 <= _octave_equivalent_delta(261.63, 392.0) <= 600.0
    # A fifth is 701 cents raw, but pitch-class distance goes the shorter way
    # round the octave, so it reports ~500.
    assert _octave_equivalent_delta(261.63, 392.0) == pytest.approx(500.0, abs=1.0)


def test_essentia_support_confirms_but_does_not_override() -> None:
    """The boost must break a near-tie, not overrule a clear lead.

    At 1.15 the boost flipped a 10% score gap and moved a speech sample from
    76.93 Hz to 83.70 Hz, which is a coin toss dressed up as a correction.
    """
    lead = TonicCandidate(
        hz=76.93, cents=float(cents_between(27.5, 76.93)), score=1.263, confidence=1.0,
        sa_peak=1.0, pa_peak=0.4, octave_peak=0.4,
    )
    close = TonicCandidate(
        hz=83.70, cents=float(cents_between(27.5, 83.70)), score=1.141, confidence=0.90,
        sa_peak=0.8, pa_peak=0.4, octave_peak=0.4,
    )
    result = TonicResult(candidates=(lead, close), method="histogram")
    params = TonicParams()
    # Essentia supports the runner-up; the ranking must survive.
    assert params.essentia_agreement_boost < 1.263 / 1.141
    merged = merge_essentia(result, 83.70, params)
    assert merged.hz == pytest.approx(76.93)
    assert "essentia" in merged.candidates[1].methods


def test_merge_essentia_with_none_is_a_no_op() -> None:
    c = TonicCandidate(
        hz=261.63, cents=1200.0, score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.0, octave_peak=0.0,
    )
    result = TonicResult(candidates=(c,), method="histogram")
    assert merge_essentia(result, None).candidates == result.candidates


# --- the lowest-note prior -------------------------------------------------


def test_a_scale_run_resolves_to_its_lowest_note_not_its_strongest() -> None:
    """Sa must win a scale run, even when another degree has a taller peak.

    In a run that touches every degree, every candidate has an equally strong
    peak *and* a fifth above it -- there is a peak above Re too, it just is not
    a Pa. So the peak terms cannot separate Sa from Re at all, and the decision
    rests on the lowest-note prior. At the old weight of 0.12 the two came out
    0.2% apart in score and Re won whenever the peak jitter went its way.
    """
    from tests.synthetic import sequence

    from swaras.tonic import detect_tonic_histogram

    degrees = [0, 200, 400, 700, 900, 1100, 900, 700, 400, 200, 0]
    freqs = [261.63 * 2.0 ** (c / 1200.0) for c in degrees]
    contour = MelodiaPitchDetector().estimate(sequence(freqs, note_duration_s=0.40), 22050)
    got = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    assert float(cents_between(261.63, got.hz)) == pytest.approx(0.0, abs=15.0)


def test_steady_tones_can_still_defeat_the_prior_and_that_is_known() -> None:
    """Document the residual failure, so a change in it is noticed.

    At some note lengths the tracked contour dwells on one degree and gives it
    roughly double the histogram mass of every other. That is not jitter a
    prior can absorb: recovering from it needs a weight of 0.50, which was
    measured and rejected because it makes the tonic lock onto an octave-low
    mandra and flips a speech sample's octave. Perfectly sustained synthetic
    tones are the worst case; the realistic phrases in
    ``tests/test_performance.py``, which have vibrato, glides, an envelope and
    a drone, are exact at every note length tried.
    """
    from tests.synthetic import sequence

    from swaras.tonic import detect_tonic_histogram

    degrees = [0, 200, 400, 700, 900, 1100, 900, 700, 400, 200, 0]
    freqs = [261.63 * 2.0 ** (c / 1200.0) for c in degrees]
    contour = MelodiaPitchDetector().estimate(sequence(freqs, note_duration_s=0.55), 22050)
    got = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    assert abs(float(cents_between(261.63, got.hz))) > 15.0, (
        f"steady tones no longer defeat the prior (Sa now {got.hz:.1f} Hz); the "
        "rejected 0.50 weight and the README's known limitations are worth "
        "revisiting"
    )


def test_the_prior_does_not_lock_onto_an_octave_low_mandra() -> None:
    """A guard on the weight from the other direction.

    A mandra an octave below Sa is the lowest pitch in a phrase that dips to
    it, so a prior heavy enough to dominate the peak terms will call the
    mandra the tonic. That is the failure that ruled out 0.50, and it is much
    worse than being mildly confused: it is an octave error, which is the
    mistake this stage is most often wrong about in the first place.
    """
    from tests.synthetic import sequence

    from swaras.pipeline import transcribe

    sa = 261.63
    degrees = [-1200, 0, 200, 400, 700, 900, 1100, 900, 700, 400, 200, -1200]
    freqs = [sa * 2.0 ** (c / 1200.0) for c in degrees]
    y = sequence(freqs, note_duration_s=0.45)
    result = transcribe(y, 22050, detector="melodia")
    error = float(cents_between(sa, result.tonic.hz))
    # Confused, yes, but by a fifth-ish rather than locked an octave low.
    assert abs(error) < 600.0, f"tonic locked {error:.0f} cents from Sa"


def test_the_lowest_note_prior_is_strong_enough_to_break_a_tie() -> None:
    """A regression guard on the weight itself.

    This is the arithmetic that failed: the prior is the only term that can
    separate Sa from Re in a scale run, so if it drops much below the measured
    value the original bug returns.
    """
    params = TonicParams()
    assert params.w_lowest >= 0.25


def test_a_quiet_drone_does_not_become_the_tonic() -> None:
    """The prior must not simply mean "the lowest note in the audio".

    A tanpura an octave below the voice puts its Pa under the singer's Sa, and
    a dominant prior would call that Pa the tonic. ``min_peak_ratio`` is what
    saves it: a quiet drone's Pa does not count as sung, so the lowest
    supported pitch is the voice's own Sa.
    """
    from tests.synthetic import meend, sung_note, tanpura_drone

    from swaras.tonic import detect_tonic_histogram

    sa = 261.63
    degrees = [0, 200, 400, 700, 900, 1100, 900, 700, 400, 200, 0]
    parts: list[np.ndarray] = []
    for i, c in enumerate(degrees):
        f = sa * 2.0 ** (c / 1200.0)
        parts.append(sung_note(f, 0.55, 22050, vibrato_rate_hz=5.5, vibrato_cents=35.0, seed=i))
        if i + 1 < len(degrees):
            parts.append(meend(f, sa * 2.0 ** (degrees[i + 1] / 1200.0), 0.11, 22050))
    voice = np.concatenate(parts)
    voice = voice / max(np.max(np.abs(voice)), 1e-9) * 0.8
    # Sa and Pa an octave below the singer, so the drone's Pa is the lower pitch.
    drone = tanpura_drone(sa / 2, voice.size / 22050, 22050, fifth_hz=sa / 2 * 1.4983070768766815, seed=7)
    mix = voice + drone[: voice.size] * 0.08 / max(np.max(np.abs(drone)), 1e-9)
    contour = MelodiaPitchDetector().estimate(mix, 22050)
    got = detect_tonic_histogram(contour, TonicParams(use_essentia=False))
    assert float(cents_between(sa, got.hz)) == pytest.approx(0.0, abs=15.0)


# --- manual override -------------------------------------------------------


def test_manual_tonic_has_full_confidence() -> None:
    result = manual_tonic(261.63)
    assert result.method == "manual"
    assert result.hz == pytest.approx(261.63)
    assert result.confidence == 1.0
    assert result.candidates[0].methods == ("manual",)


@pytest.mark.parametrize("bad", [0.0, -100.0, float("nan"), float("inf")])
def test_manual_tonic_rejects_invalid(bad: float) -> None:
    with pytest.raises(ValueError):
        manual_tonic(bad)
