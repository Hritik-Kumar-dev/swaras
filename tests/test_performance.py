"""End-to-end tests on a *realistic* performance rather than clean tones.

Every other end-to-end test in this suite runs on steady sine tones with no
vibrato, no accompaniment, no glides and no dynamics. That is enough to prove
the arithmetic is right, and the headline accuracy figures were measured on
it, but it is much easier than singing. A real khayal line has all four of
the things the clean tests leave out:

- a tanpura drone underneath, which is what Essentia's tonic model is designed
  to find and what the clean tests never give it;
- vibrato that arrives *after* the note onset, as a singer's does;
- meends, the portamento between one note and the next;
- phrase-level dynamics, so a note is not the same loudness as its neighbour.

These tests exist because the honest gap was that this had never been
measured. Run ``python tests/performance_accuracy.py`` for the full report.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from swaras.cents import cents_between
from swaras.pipeline import transcribe_upload
from tests.synthetic import khayal_phrase

SA_HZ = 261.63  # C4


def _wav(y: np.ndarray, sample_rate: int = 22050) -> bytes:
    """Encode mono float samples as in-memory WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _transcribe(y: np.ndarray, **kwargs):
    return transcribe_upload(_wav(y), **kwargs)


# --- the headline: a drone-backed phrase, with everything switched on ------


def test_melodia_reads_a_full_khayal_phrase_exactly() -> None:
    """Drone, vibrato, meends and dynamics: the notation must still be exact.

    This is the case a clean-tone round trip cannot tell us anything about.
    """
    y, expected, _ = khayal_phrase(SA_HZ, with_drone=True, with_vibrato=True,
                                   with_dynamics=True, meend_s=0.11)
    result = _transcribe(y, detector="melodia")
    assert result.transcription.text == " ".join(expected)


def test_the_drone_does_not_cost_accuracy() -> None:
    """Adding a tanpura under the voice must not degrade the transcription.

    The drone is roughly a fifth below the voice for much of the phrase, so a
    tracker that locks onto the accompaniment instead of the singer produces
    plausible-looking nonsense. This is the specific failure the drone is here
    to expose.
    """
    clean, expected, _ = khayal_phrase(SA_HZ, with_drone=False)
    with_drone, _, _ = khayal_phrase(SA_HZ, with_drone=True)
    bare = _transcribe(clean, detector="melodia").transcription.text
    full = _transcribe(with_drone, detector="melodia").transcription.text
    assert bare == " ".join(expected)
    assert full == " ".join(expected)


def test_detecting_sa_works_on_a_drone_backed_phrase() -> None:
    """Sa must be found without being told, with an accompaniment present.

    A drone supplies a second, competing source of pitch, so a detector that
    only understands solo tones has an easy time here and a real one does not.
    """
    y, _, _ = khayal_phrase(SA_HZ, with_drone=True)
    result = _transcribe(y, detector="melodia")
    assert float(cents_between(SA_HZ, result.tonic.hz)) == pytest.approx(0.0, abs=15.0)


# --- each complicating factor on its own ----------------------------------


@pytest.mark.parametrize(
    ("kwargs", "what"),
    [
        ({"with_vibrato": True}, "vibrato"),
        ({"meend_s": 0.11}, "meends between every pair of notes"),
        ({"with_dynamics": True}, "phrase-level dynamics"),
    ],
)
def test_each_factor_alone_does_not_break_the_notation(kwargs: dict, what: str) -> None:
    y, expected, _ = khayal_phrase(SA_HZ, with_drone=False, **kwargs)
    result = _transcribe(y, detector="melodia")
    assert result.transcription.text == " ".join(expected), f"{what} broke it"


def test_vibrato_is_absorbed_into_the_note_rather_than_splitting_it() -> None:
    """35 cents of 5.5 Hz vibrato must not become spurious notes.

    Vibrato depth is well inside the stability window this stage is built
    around; if it were not, a single held note would come back as a run of
    short ones and the phrase length would change.
    """
    y, expected, _ = khayal_phrase(SA_HZ, with_drone=False, with_vibrato=True)
    result = _transcribe(y, detector="melodia")
    assert len(result.segmentation.notes) == len(expected)


def test_a_repeated_phrase_is_read_twice_over() -> None:
    """A phrase sung twice must be transcribed twice, not merged into one.

    The break between repetitions includes a breath, so this also covers the
    case where the same pitch sits either side of a silence.
    """
    once, expected, _ = khayal_phrase(SA_HZ, with_drone=True, repeat=1)
    twice, expected2, _ = khayal_phrase(SA_HZ, with_drone=True, repeat=2)
    assert expected2 == expected * 2
    got = _transcribe(twice, detector="melodia").transcription.text
    assert got == " ".join(expected2)
    assert got.count("S") == " ".join(expected2).count("S")


# --- what is documented as not working ------------------------------------


def test_pyin_can_track_the_drone_as_extra_notes() -> None:
    """Record pYIN's behaviour on a drone, so a change in it is noticed.

    Melodia holds the phrase exactly; pYIN emits extra notes, because it is
    more willing to follow the accompaniment. This is a known weakness rather
    than a passing test: if pYIN ever stops doing this, the assertion fails
    and the note in the README needs revisiting.
    """
    y, expected, _ = khayal_phrase(SA_HZ, with_drone=True, repeat=2)
    got = _transcribe(y, detector="pyin").transcription.text.split()
    # 24 notes for 22 expected at the time of writing. The extras are the
    # drone being followed for a note here and there.
    assert len(got) > len(expected), (
        f"pYIN used to over-segment a drone-backed phrase; it now returns "
        f"{len(got)} notes for {len(expected)} expected"
    )


def test_essentia_agrees_on_pitch_class_with_a_sung_phrase_over_a_drone() -> None:
    """With a voice over the drone, Essentia finds the right note one octave down.

    This is the case the pitch-class agreement rule exists for. On a bare
    drone Essentia is unreliable (see the sweep below), but a sung line over a
    tanpura gives it enough to work with, and what it returns is Sa exactly one
    octave low -- which names the right note and the wrong register.
    """
    pytest.importorskip("essentia")
    from swaras.tonic import detect_tonic_essentia

    y, _, _ = khayal_phrase(SA_HZ, with_drone=True, with_vibrato=True)
    got = detect_tonic_essentia(y.astype(np.float32), 22050)
    if got is None:
        pytest.skip("Essentia returned no tonic for this audio")
    raw = abs(float(cents_between(SA_HZ, got)))
    folded = min(raw % 1200, 1200 - (raw % 1200))
    assert folded < 50.0, f"not even the right note: {raw:.0f} cents away"
    # It is the octave that is wrong, not the note -- which is the whole point.
    assert 1100.0 < raw < 1300.0, f"expected an octave error, got {raw:.0f} cents"


def test_essentia_on_a_bare_drone_is_often_a_fifth_away() -> None:
    """Record the weak case honestly, so a change in it is noticed.

    Strip the voice out and Essentia has only a plucked string to work with.
    Over eight tonics it names the right pitch class 4 times in 8 and the right
    octave 1 time in 8; the remaining errors are a fifth away, which is a
    different note, so they are not treated as agreement. A fifth is included
    here on purpose: a change in Essentia that fixed these would be worth
    knowing about.
    """
    pytest.importorskip("essentia")
    from swaras.tonic import detect_tonic_essentia

    from tests.synthetic import tanpura_drone

    right_pitch_class = 0
    for sa in (146.83, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00):
        drone = tanpura_drone(sa, 6.0, 22050).astype(np.float32)
        got = detect_tonic_essentia(drone, 22050)
        if got is None:
            continue
        raw = abs(float(cents_between(sa, got)))
        if min(raw % 1200, 1200 - (raw % 1200)) < 50.0:
            right_pitch_class += 1
    assert right_pitch_class == 4, (
        f"Essentia now names the right pitch class on a bare drone "
        f"{right_pitch_class} times in 8, was 4; revisit the agreement rule"
    )
