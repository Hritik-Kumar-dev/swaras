"""Tests for Stage 1: audio loading/resampling/normalising/trimming."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from swaras.audio_io import (
    frame_rms_db,
    load_bytes,
    load_file,
    normalize,
    resample,
    to_mono,
    trim_silence,
)
from swaras.config import AudioParams
from tests import synthetic


def test_to_mono_averages_channels() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.5], [0.0, 2.0]])
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert mono.shape == (3,)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, [0.0, 0.5, 1.0], atol=1e-6)


def test_to_mono_passes_through_1d() -> None:
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    np.testing.assert_allclose(to_mono(x), x, atol=1e-7)


def test_to_mono_rejects_3d() -> None:
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 2)))


def test_resample_changes_length_proportionally() -> None:
    sr_in = 44_100
    t = np.arange(sr_in) / sr_in
    y = np.sin(2 * np.pi * 440 * t)
    out = resample(y.astype(np.float32), sr_in, 22_050)
    assert abs(out.size - sr_in // 2) <= 2
    # Frequency content should be preserved.
    spectrum = np.abs(np.fft.rfft(out))
    peak_bin = int(np.argmax(spectrum))
    assert abs(peak_bin * (22_050 / out.size) - 440) < 5


def test_resample_is_identity_when_rates_match() -> None:
    x = np.linspace(-1, 1, 100, dtype=np.float32)
    out = resample(x, 22_050, 22_050)
    assert out is x or np.array_equal(out, x)


def test_resample_rejects_bad_rates() -> None:
    with pytest.raises(ValueError):
        resample(np.zeros(10), 0, 22_050)


def test_normalize_scales_to_target_peak() -> None:
    y = np.array([0.0, 0.2, -0.4], dtype=np.float32)
    out = normalize(y, target_peak=0.9)
    assert np.isclose(np.max(np.abs(out)), 0.9, atol=1e-6)


def test_normalize_leaves_silence_alone() -> None:
    z = np.zeros(100, dtype=np.float32)
    out = normalize(z, 0.9)
    assert np.all(out == 0.0)


def test_normalize_rejects_out_of_range_target() -> None:
    with pytest.raises(ValueError):
        normalize(np.ones(10), target_peak=1.5)
    with pytest.raises(ValueError):
        normalize(np.ones(10), target_peak=0.0)


def test_frame_rms_db_tracks_level() -> None:
    sr = 22_050
    loud = np.sin(2 * np.pi * 200 * np.arange(sr) / sr) * 0.5
    quiet = loud * 0.01  # -40 dB
    hop = 512
    loud_db = frame_rms_db(loud, hop * 2, hop)
    quiet_db = frame_rms_db(quiet, hop * 2, hop)
    assert np.max(loud_db) - np.max(quiet_db) > 35


def test_frame_rms_db_on_empty_signal() -> None:
    assert frame_rms_db(np.array([]), 512, 512).size == 0


def test_trim_silence_removes_leading_and_trailing_silence() -> None:
    sr = 22_050
    tone = np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    y = np.concatenate([np.zeros(sr), tone, np.zeros(sr)])
    trimmed, offset = trim_silence(y, sr, trim_db=40.0)
    assert offset > 0.9
    assert trimmed.size < y.size
    # The tone itself should be largely intact.
    assert abs(trimmed.size - sr) < int(0.05 * sr)


def test_trim_silence_on_all_silent_audio() -> None:
    sr = 22_050
    z = np.zeros(sr)
    trimmed, offset = trim_silence(z, sr, min_duration_s=0.05)
    assert offset == 0.0
    assert trimmed.size == int(0.05 * sr)


def test_trim_silence_on_empty() -> None:
    trimmed, offset = trim_silence(np.array([]), 22_050)
    assert trimmed.size == 0
    assert offset == 0.0


def test_load_file_resamples_to_target_rate(tmp_path) -> None:
    sr = 44_100
    y = synthetic.sine_sweep(200, 400, 1.0, sample_rate=sr)
    path = tmp_path / "test.wav"
    sf.write(path, y.astype(np.float32), sr)
    audio = load_file(path, AudioParams(sample_rate=22_050))
    assert audio.sample_rate == 22_050
    assert abs(audio.samples.size - 22_050) < 100
    assert audio.source_duration_s == pytest.approx(1.0, abs=0.05)


def test_load_file_normalizes(tmp_path) -> None:
    y = 0.05 * synthetic.sine_tone(np.full(22_050, 220.0), 22_050)
    path = tmp_path / "quiet.wav"
    sf.write(path, y.astype(np.float32), 22_050)
    audio = load_file(path)
    assert np.max(np.abs(audio.samples)) == pytest.approx(0.95, rel=1e-3)


def test_load_file_downmixes_stereo(tmp_path) -> None:
    sr = 22_050
    left = synthetic.sine_tone(np.full(sr, 300.0), sr)
    right = synthetic.sine_tone(np.full(sr, 300.0), sr)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1).astype(np.float32), sr)
    audio = load_file(path)
    assert audio.samples.ndim == 1
    assert audio.sample_rate == sr


def test_load_file_handles_silence_with_trim(tmp_path) -> None:
    y = synthetic.sine_wave_with_silence(220.0, 1.0, 22_050)
    path = tmp_path / "padded.wav"
    sf.write(path, y.astype(np.float32), 22_050)
    audio = load_file(path)
    assert audio.source_duration_s == pytest.approx(2.0, abs=0.1)
    assert audio.duration_s < audio.source_duration_s
    assert audio.trim_offset_s > 0.4


def test_load_file_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "nope.wav")


def test_load_file_rejects_non_audio(tmp_path) -> None:
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is definitely not a wav file")
    with pytest.raises(ValueError):
        load_file(path)


def test_load_bytes_matches_load_file(tmp_path) -> None:
    y = synthetic.note(220.0, 0.6, 22_050)
    path = tmp_path / "note.wav"
    sf.write(path, y.astype(np.float32), 22_050)
    from_file = load_file(path)
    from_bytes = load_bytes(path.read_bytes(), filename="note.wav")
    np.testing.assert_allclose(from_file.samples, from_bytes.samples, atol=1e-6)
    assert from_file.sample_rate == from_bytes.sample_rate


def test_load_bytes_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        load_bytes(b"not audio at all", filename="x.wav")


def test_ogg_roundtrip_through_load_file(tmp_path) -> None:
    y = synthetic.note(261.63, 0.5, 22_050)
    path = tmp_path / "note.ogg"
    sf.write(path, y.astype(np.float32), 22_050, format="OGG")
    audio = load_file(path)
    assert audio.sample_rate == 22_050
    assert audio.samples.size > 0
