"""Tests for Stage 5: the 22-shruti layer.

The shruti layer's contract is that it reports a *measurement*, not a label:
the nearest shruti plus the signed distance from it. These tests hold it to
that, and check the confidence is meaningful rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from swaras.normalize import Octave
from swaras.shruti import (
    _extended_targets,
    cents_to_shruti,
    format_shruti_line,
    nearest_shruti,
    notes_to_shrutis,
    shruti_statistics,
)
from swaras.swar import cents_to_swar
from swaras.tuning import load_shruti_table

TABLE = load_shruti_table()


# --- the grid itself -------------------------------------------------------


def test_extended_targets_cover_three_octaves() -> None:
    cents, indices = _extended_targets(TABLE)
    assert cents.size == 3 * TABLE.n_shrutis
    np.testing.assert_array_equal(indices[: TABLE.n_shrutis], np.arange(TABLE.n_shrutis))
    # The middle third is the table itself.
    np.testing.assert_allclose(cents[TABLE.n_shrutis : 2 * TABLE.n_shrutis], TABLE.shruti_cents)


@pytest.mark.parametrize("cents", TABLE.shruti_cents.tolist())
def test_every_shruti_matches_itself(cents: float) -> None:
    """A pitch exactly on a shruti must match that shruti, with zero deviation."""
    idx, deviation, confidence = nearest_shruti(cents, TABLE)
    assert int(idx[0]) == TABLE.shrutis.index(
        next(s for s in TABLE.shrutis if s.cents == cents)
    )
    assert deviation[0] == pytest.approx(0.0, abs=1e-6)
    assert confidence[0] == pytest.approx(1.0, abs=1e-6)


def test_all_22_shrutis_are_reachable() -> None:
    got = sorted(int(i) for i in nearest_shruti(TABLE.shruti_cents, TABLE)[0])
    assert got == list(range(22))


# --- deviation -------------------------------------------------------------


def test_deviation_is_signed() -> None:
    above = cents_to_shruti(10.0, TABLE)
    below = cents_to_shruti(-10.0, TABLE)
    assert above.deviation_cents == pytest.approx(10.0, abs=1e-6)
    assert below.deviation_cents == pytest.approx(-10.0, abs=1e-6)
    assert above.swara == below.swara == "Sa"


def test_deviation_is_measured_the_short_way_round() -> None:
    """A note 8 cents below Sa must report -8, not +1192."""
    n = cents_to_shruti(-8.0, TABLE)
    assert n.swara == "Sa"
    assert n.deviation_cents == pytest.approx(-8.0, abs=0.1)
    assert abs(n.deviation_cents) < 100


def test_deviation_within_a_pair_prefers_the_closer_one() -> None:
    """The two shrutis of a pair are 22 cents apart, so this is a real choice."""
    lower = cents_to_shruti(95.0, TABLE)
    upper = cents_to_shruti(110.0, TABLE)
    assert (lower.index, lower.variant) == (1, "low")
    assert (upper.index, upper.variant) == (2, "high")
    assert lower.deviation_cents == pytest.approx(5.0, abs=0.1)
    assert upper.deviation_cents == pytest.approx(-2.0, abs=0.1)


# --- confidence ------------------------------------------------------------


def test_confidence_is_one_on_a_shruti_and_zero_at_a_midpoint() -> None:
    on = cents_to_shruti(408.0, TABLE)
    assert on.confidence == pytest.approx(1.0, abs=1e-6)
    # 397 is exactly halfway between the Ga pair at 386 and 408.
    between = cents_to_shruti(397.0, TABLE)
    assert between.confidence == pytest.approx(0.0, abs=1e-6)


def test_confidence_falls_to_zero_at_the_boundary() -> None:
    """Confidence is a V: 1.0 on a shruti, 0.0 midway to the next.

    It rises again past the boundary because the match has moved to the other
    shruti, so confidence measures the *current* match, not distance from some
    fixed reference.
    """
    values = [cents_to_shruti(float(c), TABLE).confidence for c in (408, 405, 400, 397, 392)]
    assert values[0] == pytest.approx(1.0, abs=1e-6)
    # Falling towards the boundary from the Ga-high side.
    assert values[0] > values[1] > values[2] > values[3]
    # The boundary itself is a coin flip, and just past it the other shruti
    # takes over with its own rising confidence.
    assert values[3] == pytest.approx(0.0, abs=1e-6)
    assert values[4] > values[3]


def test_clarity_marks_where_the_match_would_flip() -> None:
    """A note exactly ``clarity`` cents from a shruti should be a coin flip."""
    n = cents_to_shruti(0.0, TABLE)
    midpoint = cents_to_shruti(n.clarity_cents, TABLE)
    assert midpoint.confidence == pytest.approx(0.0, abs=0.02)
    assert int(midpoint.index) != 0 or midpoint.confidence < 0.05


def test_pa_singleton_has_wider_clarity_than_a_pair() -> None:
    """Pa is a single shruti, so it is unambiguous across a 90-cent gap."""
    pa = cents_to_shruti(702.0, TABLE)
    ga = cents_to_shruti(408.0, TABLE)
    # Pa is 90 cents from both neighbours, so the boundary is 45 away. Ga-high
    # is 22 above Ga-low and 90 below Ma, so the nearer boundary wins and the
    # clarity is 11.
    assert pa.clarity_cents == pytest.approx(45.0)
    assert ga.clarity_cents == pytest.approx(11.0)


# --- octave ----------------------------------------------------------------


def test_shruti_indices_repeat_across_octaves() -> None:
    low = cents_to_shruti(0.0, TABLE)
    high = cents_to_shruti(1200.0, TABLE)
    assert low.index == high.index
    assert low.octave == Octave.MADHYA
    assert high.octave == Octave.TAAR


def test_octave_below_sa() -> None:
    n = cents_to_shruti(-1200.0, TABLE)
    assert n.octave == Octave.MANDRA
    assert n.swara == "Sa"


# --- the two layers together ----------------------------------------------


@pytest.mark.parametrize("cents", [0, 90, 112, 182, 204, 294, 316, 386, 408, 498, 519, 590, 612, 702, 792, 814, 884, 906, 996, 1018, 1088, 1110])
def test_shruti_agrees_with_the_swar_it_belongs_to(cents: float) -> None:
    """Every shruti must sit inside the swara position it is assigned to."""
    shruti = cents_to_shruti(float(cents), TABLE)
    swara = cents_to_swar(float(cents), TABLE)
    assert shruti.swara == swara.swara


def test_notes_to_shrutis_preserves_order_and_timing() -> None:
    from swaras.swar import SwaraNote

    notes = [
        SwaraNote("Sa", "S", Octave.MADHYA, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5),
        SwaraNote("Ni", "N", Octave.MADHYA, 1099.0, 0.0, 1099.0, 1.0, 0.5, 1.0),
    ]
    out = notes_to_shrutis(notes, TABLE)
    assert [n.swara for n in out] == ["Sa", "Ni"]
    assert out[0].start_s == 0.0 and out[1].end_s == 1.0


# --- output ----------------------------------------------------------------


def test_labels_carry_the_variant() -> None:
    assert cents_to_shruti(90.0, TABLE).label == "R1l"
    assert cents_to_shruti(112.0, TABLE).label == "R2h"
    assert cents_to_shruti(0.0, TABLE).label == "S0"
    assert cents_to_shruti(702.0, TABLE).label == "P13"


def test_label_with_deviation_is_signed() -> None:
    assert cents_to_shruti(10.0, TABLE).label_with_deviation.endswith("+10c")
    assert cents_to_shruti(-10.0, TABLE).label_with_deviation.endswith("-10c")


def test_format_shruti_line() -> None:
    notes = [cents_to_shruti(0.0, TABLE), cents_to_shruti(10.0, TABLE)]
    assert format_shruti_line(notes) == "S0 +0c S0 +10c"
    assert format_shruti_line(notes, with_deviation=False) == "S0 S0"
    assert format_shruti_line([]) == ""


def test_shruti_statistics() -> None:
    notes = [cents_to_shruti(0.0, TABLE), cents_to_shruti(20.0, TABLE), cents_to_shruti(397.0, TABLE)]
    stats = shruti_statistics(notes)
    assert stats["count"] == 3
    assert stats["max_abs_deviation_cents"] == pytest.approx(20.0)
    # 397 is an exact tie, so it counts as unclear.
    assert stats["unclear_count"] == 1
    assert stats["histogram"]


def test_shruti_statistics_on_empty() -> None:
    stats = shruti_statistics([])
    assert stats["count"] == 0
    assert stats["mean_abs_deviation_cents"] is None


def test_shruti_note_to_dict() -> None:
    import json

    json.dumps(cents_to_shruti(10.0, TABLE).to_dict())


# --- robustness ------------------------------------------------------------


def test_nan_passes_through() -> None:
    idx, deviation, confidence = nearest_shruti([0.0, np.nan], TABLE)
    assert idx.tolist() == [0, -1]
    assert np.isnan(deviation[1])
    assert np.isnan(confidence[1])


def test_cents_to_shruti_rejects_nan() -> None:
    with pytest.raises(ValueError):
        cents_to_shruti(float("nan"), TABLE)


def test_far_out_of_range_pitches_wrap() -> None:
    """A pitch many octaves from Sa still lands on a real shruti.

    The register names stop at two octaves either way, so the octave is
    clamped rather than the measurement being discarded.
    """
    for cents in (5000.0, -5000.0, 12345.0):
        n = cents_to_shruti(cents, TABLE)
        assert 0 <= n.index < 22
        assert abs(n.deviation_cents) <= 600.0
        assert n.octave in (Octave.MANDRA_2, Octave.TAAR_2)
