"""Tests for Stages 3 and 4: normalization, segmentation, swara mapping,
octave marking and output formatting.

The headline requirement is the round trip: synthesise a known swara
sequence, transcribe it, and get the same notation back.
"""

from __future__ import annotations

import numpy as np
import pytest

from swaras._essentia import essentia_available
from swaras.config import SegmentParams
from swaras.format import (
    Transcription,
    format_detailed,
    format_summary,
    format_text,
    to_jsonable,
)
from swaras.normalize import Octave, contour_to_cents, fold_cents, fold_to_octave
from swaras.pitch import PitchContour
from swaras.pipeline import transcribe
from swaras.segment import (
    SegmentKind,
    _contiguous_runs,
    classify_transition,
    segment_contour,
    smooth_cents,
)
from swaras.swar import cents_to_swar, nearest_swar, octave_for, segment_to_swaras
from swaras.tonic import manual_tonic
from swaras.tuning import load_shruti_table
from tests import synthetic

SR = 22_050
TABLE = load_shruti_table()
FRAME_S = 0.023


def cents_contour(values, voiced=None, times=None, sa_hz=261.63):
    """Build a CentsContour directly from cents values."""
    from swaras.normalize import CentsContour

    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    v = np.asarray(voiced, dtype=bool) if voiced is not None else np.isfinite(arr)
    t = np.arange(n) * FRAME_S if times is None else np.asarray(times, dtype=np.float64)
    cents = np.where(v, arr, np.nan)
    octave = np.where(v, np.floor(np.nan_to_num(cents) / 1200.0).astype(np.int64), 0)
    return CentsContour(
        times=t, cents=cents, voiced=v, octave=octave, sa_hz=sa_hz,
        source=np.full(n, 261.63), confidence=np.ones(n),
    )


# --- normalize -------------------------------------------------------------


def test_contour_to_cents_maps_sa_to_zero() -> None:
    contour = PitchContour(
        times=np.arange(3) * FRAME_S, f0=np.array([261.63, 261.63, 0.0]),
        voiced=np.array([True, True, False]), confidence=np.ones(3), hop_seconds=FRAME_S,
    )
    out = contour_to_cents(contour, 261.63)
    assert out.cents[0] == pytest.approx(0.0, abs=0.01)
    assert out.cents[2] != out.cents[2]  # nan
    assert not out.voiced[2]


def test_contour_to_cents_rejects_bad_sa() -> None:
    contour = PitchContour(
        times=np.zeros(1), f0=np.array([220.0]), voiced=np.array([True]),
        confidence=np.array([1.0]), hop_seconds=FRAME_S,
    )
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            contour_to_cents(contour, bad)


def test_fold_to_octave_tracks_index() -> None:
    folded, octave = fold_to_octave(np.array([-100.0, 0.0, 1199.0, 1300.0]))
    np.testing.assert_allclose(folded, [1100.0, 0.0, 1199.0, 100.0], atol=1e-6)
    assert octave.tolist() == [-1, 0, 0, 1]


def test_fold_cents_handles_nan() -> None:
    out = fold_cents(np.array([0.0, np.nan, 1300.0]))
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "value,octave,marker",
    [(0, Octave.MADHYA, ""), (-1, Octave.MANDRA, "."), (1, Octave.TAAR, "'"),
     (2, Octave.TAAR_2, "''"), (-2, Octave.MANDRA_2, "..")],
)
def test_octave_enum_markers(value: int, octave: Octave, marker: str) -> None:
    assert Octave(value).marker == marker
    assert Octave(value).name_short


# --- swara mapping ---------------------------------------------------------


@pytest.mark.parametrize(
    "cents,expected",
    [
        (0, "Sa"), (1, "Sa"), (-1, "Sa"), (60, "Re komal"), (101, "Re komal"),
        (193, "Re"), (240, "Re"), (305, "Ga komal"), (397, "Ga"),
        (508, "Ma"), (601, "Ma tivra"), (702, "Pa"), (803, "Dha komal"),
        (895, "Dha"), (1007, "Ni komal"), (1099, "Ni"),
    ],
)
def test_nearest_swar_picks_the_right_position(cents: float, expected: str) -> None:
    note = cents_to_swar(cents, TABLE)
    assert note.swara == expected


def test_sa_wraps_across_the_octave_boundary() -> None:
    """A note just below Sa' is a sharp Sa', not a very flat Ni."""
    assert cents_to_swar(1199.0, TABLE).swara == "Sa"
    assert cents_to_swar(1201.0, TABLE).swara == "Sa"
    assert cents_to_swar(1099.0, TABLE).swara == "Ni"


def test_deviation_is_measured_the_short_way_round() -> None:
    """A note 8 cents flat must report -8, not +1192."""
    note = cents_to_swar(-8.0, TABLE)
    assert note.swara == "Sa"
    assert note.deviation_cents == pytest.approx(-8.0, abs=0.1)


def test_nearest_swar_handles_nan() -> None:
    idx, dev = nearest_swar(np.array([0.0, np.nan]), TABLE)
    assert idx[0] == 0
    assert idx[1] == -1
    assert np.isnan(dev[1])


def test_cents_to_swar_rejects_nan() -> None:
    with pytest.raises(ValueError):
        cents_to_swar(float("nan"), TABLE)


@pytest.mark.parametrize(
    "cents,octave",
    [
        (0, Octave.MADHYA), (1099, Octave.MADHYA), (1100, Octave.MADHYA),
        (1200, Octave.TAAR), (2299, Octave.TAAR), (2400, Octave.TAAR_2),
        (-1200, Octave.MANDRA), (-1201, Octave.MANDRA), (-2400, Octave.MANDRA_2),
    ],
)
def test_octave_for_is_derived_from_the_nearest_swar(cents: float, octave: Octave) -> None:
    """Ni at 1099 is madhya even though it is only 101 cents below the octave."""
    assert octave_for(cents, TABLE) == octave


def test_ni_is_not_misreported_as_taar_sa() -> None:
    """Regression: rounding to the nearest octave boundary put Ni in taar."""
    note = cents_to_swar(1099.0, TABLE)
    assert note.swara == "Ni"
    assert note.label == "N"


def test_swar_labels_carry_the_octave_marker() -> None:
    assert cents_to_swar(0.0, TABLE).label == "S"
    assert cents_to_swar(1200.0, TABLE).label == "S'"
    assert cents_to_swar(-1200.0, TABLE).label == "S."
    assert cents_to_swar(2299.0, TABLE).label == "N'"


def test_komal_and_tivra_are_distinct_positions() -> None:
    assert cents_to_swar(101.0, TABLE).swara == "Re komal"
    assert cents_to_swar(193.0, TABLE).swara == "Re"
    assert cents_to_swar(601.0, TABLE).swara == "Ma tivra"
    assert cents_to_swar(508.0, TABLE).swara == "Ma"


# --- segmentation ----------------------------------------------------------


def test_contiguous_runs_finds_runs() -> None:
    mask = np.array([False, True, True, False, False, True, False])
    assert _contiguous_runs(mask) == [(1, 3), (5, 6)]
    assert _contiguous_runs(np.zeros(5, bool)) == []


def test_smooth_cents_preserves_gaps() -> None:
    cents = np.array([0.0, np.nan, np.nan, 200.0, 200.0, np.nan])
    voiced = np.isfinite(cents)
    out = smooth_cents(cents, voiced, 5)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[0] == pytest.approx(0.0)
    assert out[4] == pytest.approx(200.0)


def test_smooth_cents_collapses_vibrato() -> None:
    """A 5 Hz vibrato must be flattened, since it would split a note.

    At a 43 Hz frame rate a 5 Hz wobble has a half-period of 4.3 frames, so a
    five-frame median cannot touch it: a 25-cent vibrato fragmented a phrase
    into 16 notes instead of 12. The default width of 9 covers a full period.
    """
    t = np.arange(200) * FRAME_S
    vib = 40.0 * np.sin(2 * np.pi * 5.0 * t)
    out = smooth_cents(vib, np.ones(200, bool), 9)
    # Ignore the last few frames, where edge padding leaves residual wobble.
    assert float(np.max(np.abs(out[:-4] - 0.0))) < 12.0


def test_two_separated_notes_become_two_notes() -> None:
    contour = cents_contour([0.0] * 20 + [np.nan] * 5 + [200.0] * 20)
    seg = segment_contour(contour, SegmentParams())
    assert len(seg.notes) == 2
    assert seg.notes[0].cents == pytest.approx(0.0, abs=1.0)
    assert seg.notes[1].cents == pytest.approx(200.0, abs=1.0)


def test_step_within_a_single_voiced_run_is_still_split() -> None:
    """A tracker rarely marks the gap between notes unvoiced.

    Regression: the whole six-note phrase arrived as one 171-frame run and the
    stability test rejected it, giving a single note for the whole phrase.
    """
    values = []
    for _ in range(3):
        values += [0.0] * 20 + [200.0] * 20 + [400.0] * 20
    seg = segment_contour(cents_contour(values), SegmentParams())
    assert len(seg.notes) == 9
    assert [round(n.cents) for n in seg.notes] == [0, 200, 400] * 3


@pytest.mark.parametrize("frames", [4, 8, 13, 20, 30, 40])
def test_meend_lengths_yield_two_notes(frames: int) -> None:
    """A glide must yield the note it leaves and the note it arrives at.

    A ramp is chopped into short regions, and each slice is locally flat
    enough to pass every magnitude test, so the slices are recognised as
    drift by their shape: they travel in one direction, so the distance the
    values cover is about the range they span. A note with vibrato covers its
    range several times over and is not drift.
    """
    ramp = list(np.linspace(0.0, 700.0, frames))
    seg = segment_contour(cents_contour([0.0] * 20 + ramp + [700.0] * 20), SegmentParams())
    assert [round(n.cents) for n in seg.notes] == [0, 700]


@pytest.mark.parametrize("depth", [25.0, 40.0, 70.0])
def test_deep_vibrato_is_one_note(depth: float) -> None:
    """Vibrato must collapse to the centre pitch, not split the note.

    A trained singer rocks a steady note by 50 to 100 cents several times a
    second, which is more than the 40 cents that separate two swara positions.
    """
    n = int(0.45 / FRAME_S)
    vib = [depth * np.sin(2 * np.pi * 5.0 * (i * FRAME_S)) for i in range(n)]
    seg = segment_contour(cents_contour([0.0] * 20 + vib + [700.0] * 20), SegmentParams())
    assert [round(x.cents) for x in seg.notes] == [0, 700]


def test_rest_separates_two_same_pitch_notes() -> None:
    """Two notes at the same pitch with a rest between them are two notes.

    A gap in voicing has to end a region, or the two are joined across the
    silence and read as one long note.
    """
    seg = segment_contour(cents_contour([0.0] * 20 + [np.nan] * 11 + [0.0] * 20), SegmentParams())
    assert len(seg.notes) == 2


def test_glide_is_a_transition_not_a_note() -> None:
    ramp = list(np.linspace(0.0, 700.0, 40))
    seg = segment_contour(cents_contour([0.0] * 20 + ramp + [700.0] * 20), SegmentParams())
    assert [round(n.cents) for n in seg.notes] == [0, 700]
    assert any(s.kind == SegmentKind.TRANSITION for s in seg.segments)


def test_octave_spike_does_not_become_a_note() -> None:
    """A 3-frame octave error is a tracking glitch, not a sung note."""
    values = [0.0] * 20 + [1250.0] * 3 + [0.0] * 20
    seg = segment_contour(cents_contour(values), SegmentParams())
    assert len(seg.notes) == 1
    assert seg.notes[0].cents == pytest.approx(0.0, abs=1.0)


def test_real_octave_leap_produces_three_notes() -> None:
    """A sustained octave is a real note, not a glitch to be folded away."""
    values = [0.0] * 20 + [1200.0] * 20 + [0.0] * 20
    seg = segment_contour(cents_contour(values), SegmentParams())
    assert [round(n.cents) for n in seg.notes] == [0, 1200, 0]


def test_short_blip_is_dropped() -> None:
    """A one-frame excursion inside a note is absorbed, leaving one note."""
    values = [0.0] * 20 + [300.0] + [0.0] * 20
    seg = segment_contour(cents_contour(values), SegmentParams())
    assert [round(n.cents) for n in seg.notes] == [0]


def test_repeated_same_pitch_is_one_note() -> None:
    """A pitch re-articulated across a short gap is one sustained note."""
    values = [0.0] * 20 + [np.nan] * 3 + [0.0] * 20
    seg = segment_contour(cents_contour(values), SegmentParams())
    assert len(seg.notes) == 1


def test_short_notes_are_ignored() -> None:
    """A single stray frame is not a sung note."""
    values = [0.0] * 20 + [700.0] + [0.0] * 20
    seg = segment_contour(cents_contour(values), SegmentParams())
    assert [round(n.cents) for n in seg.notes] == [0]


def test_vibrato_does_not_split_a_note() -> None:
    """A note with 40 cents of 5 Hz vibrato is one note, centred on its pitch."""
    n = int(0.45 / FRAME_S)
    vib = [40.0 * np.sin(2 * np.pi * 5.0 * (i * FRAME_S)) for i in range(n)]
    seg = segment_contour(cents_contour([0.0] * 20 + vib + [700.0] * 20), SegmentParams())
    assert [round(x.cents) for x in seg.notes] == [0, 700]


def test_segment_contour_on_empty_input() -> None:
    seg = segment_contour(cents_contour([np.nan] * 50), SegmentParams())
    assert seg.notes == [] and seg.segments == []
    assert "no voiced frames" in seg.diagnostics["reason"]


def test_segment_contour_with_no_stable_material() -> None:
    seg = segment_contour(cents_contour([0.0, 500.0, 1000.0]), SegmentParams())
    assert seg.notes == []


def test_classify_transition_distinguishes_glide_from_blip() -> None:
    p = SegmentParams()
    assert classify_transition(0.0, 400.0, 0.1, p) == SegmentKind.TRANSITION
    assert classify_transition(0.0, 10.0, 0.1, p) == SegmentKind.BLIP


def test_segmentation_to_dict_is_serialisable() -> None:
    import json

    seg = segment_contour(cents_contour([0.0] * 30 + [200.0] * 30), SegmentParams())
    json.dumps(to_jsonable(seg.to_dict()))


# --- segment -> swara ------------------------------------------------------


def test_segment_to_swaras_maps_notes_only_by_default() -> None:
    seg = segment_contour(cents_contour([0.0] * 20 + list(np.linspace(0, 700, 30)) + [700.0] * 20))
    notes = segment_to_swaras(seg.segments, TABLE)
    assert [n.swara for n in notes] == ["Sa", "Pa"]
    with_transitions = segment_to_swaras(seg.segments, TABLE, include_transitions=True)
    assert len(with_transitions) >= len(notes)


# --- round trip ------------------------------------------------------------

ROUND_TRIP_PHRASES = {
    "Sa Re Ga Pa": [(0, "S"), (200, "R"), (400, "G"), (700, "P")],
    "ascending scale": [
        (0, "S"), (200, "R"), (305, "g"), (400, "G"), (508, "M"),
        (700, "P"), (900, "D"), (1099, "N"),
    ],
    "Sa Sa' Pa Sa": [(0, "S"), (1200, "S'"), (700, "P"), (0, "S")],
    "with mandra": [(-1200, "S."), (-498, "P."), (0, "S"), (700, "P")],
    "komal and tivra": [
        (0, "S"), (90, "r"), (200, "R"), (601, "Mt"), (1007, "n"), (1100, "N"),
    ],
}
TONICS = [220.0, 261.63, 293.66]


def render(freqs: list[float], repeat: int = 2) -> np.ndarray:
    """Render a repeated phrase with rests, plus trailing silence."""
    one = synthetic.sequence(freqs, note_duration_s=0.45, vibrato_cents=0.0)
    rest = np.zeros(int(0.25 * SR))
    parts = [one]
    for _ in range(repeat - 1):
        parts.extend([rest, one])
    parts.append(np.zeros(int(0.3 * SR)))
    return np.concatenate(parts)


@pytest.mark.parametrize("detector", ["pyin", "melodia"])
@pytest.mark.parametrize("tonic", TONICS)
@pytest.mark.parametrize("name", list(ROUND_TRIP_PHRASES))
def test_round_trip_is_exact(name: str, tonic: float, detector: str) -> None:
    """Synthesise a known swara sequence and transcribe it back exactly.

    Sa is supplied, so this measures segmentation and mapping alone; tonic
    detection is covered separately in test_tonic.py.
    """
    from swaras.pitch import get_detector

    if detector == "melodia" and not essentia_available():
        pytest.skip("essentia not installed")
    phrase = ROUND_TRIP_PHRASES[name]
    freqs = [synthetic.cents_to_hz(c, tonic) for c, _ in phrase]
    y = render(freqs, repeat=1)
    contour = get_detector(detector).estimate(y, SR)
    seg = segment_contour(contour_to_cents(contour, tonic), SegmentParams())
    got = [n.label for n in segment_to_swaras(seg.notes, TABLE)]
    assert got == [sym for _, sym in phrase], f"{detector} @ {tonic}: {got}"


@pytest.mark.parametrize("tonic", TONICS)
def test_round_trip_full_pipeline_with_supplied_sa(tonic: float) -> None:
    """The same round trip through the whole pipeline, via --sa."""
    phrase = ROUND_TRIP_PHRASES["Sa Re Ga Pa"]
    freqs = [synthetic.cents_to_hz(c, tonic) for c, _ in phrase]
    y = render(freqs, repeat=1)
    result = transcribe(y, SR, sa_hz=tonic, detector="pyin")
    assert result.transcription.text == "S R G P"


def test_octave_markers_end_to_end() -> None:
    phrase = [(-1200, "S."), (-498, "P."), (0, "S"), (700, "P"), (1200, "S'"), (2299, "N'")]
    freqs = [synthetic.cents_to_hz(c, 261.63) for c, _ in phrase]
    y = render(freqs, repeat=1)
    result = transcribe(y, SR, sa_hz=261.63, detector="pyin")
    assert result.transcription.text == "S. P. S P S' N'"


# --- formatting ------------------------------------------------------------


def make_transcription(notes: list[SwaraNote]) -> Transcription:  # type: ignore[name-defined]
    """Build a Transcription from notes."""
    return Transcription(notes=notes, sa_hz=261.63)


def test_format_text_and_labels() -> None:
    notes = segment_to_swaras(
        segment_contour(cents_contour([0.0] * 20 + [1200.0] * 20 + [2299.0] * 20)).notes, TABLE
    )
    t = make_transcription(notes)
    assert t.text == "S S' N'"
    assert format_text(notes) == t.text
    assert "Sa (taar)" in t.text_long


def test_transcription_counts_ignore_octave() -> None:
    notes = segment_to_swaras(
        segment_contour(cents_contour([0.0] * 20 + [1200.0] * 20)).notes, TABLE
    )
    assert make_transcription(notes).counts == {"Sa": 2}


def test_transcription_to_json_round_trips() -> None:
    import json

    notes = segment_to_swaras(segment_contour(cents_contour([0.0] * 25)).notes, TABLE)
    payload = json.loads(make_transcription(notes).to_json())
    assert payload["notation"] == "S"
    assert payload["notes"][0]["swar"] == "Sa"
    assert payload["note_count"] == 1


def test_format_detailed_and_summary_render() -> None:
    notes = segment_to_swaras(segment_contour(cents_contour([0.0] * 25)).notes, TABLE)
    t = make_transcription(notes)
    assert "cents" in format_detailed(notes)
    assert "Sa" in format_summary(t)
    assert format_detailed([]) == "(no notes)"


def test_to_jsonable_handles_numpy_and_nan() -> None:
    out = to_jsonable({"a": np.int64(3), "b": np.float64("nan"), "c": [np.float32(1.5)]})
    assert out == {"a": 3, "b": None, "c": [1.5]}
