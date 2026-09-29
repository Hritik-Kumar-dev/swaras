"""Stage 1a: loading, resampling, normalisation and silence trimming.

The goal of this stage is to hand the pitch tracker a clean, mono, 22.05 kHz
signal with as little silence as possible.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

from .config import AudioParams

logger = logging.getLogger(__name__)

#: Sample formats that ``soundfile`` can read directly.
_NATIVE_FORMATS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif", ".w64", ".caf", ".mp3"}


@dataclass
class LoadedAudio:
    """A normalised, mono, resampled audio buffer.

    Attributes:
        samples: Mono float32 signal, nominal range ``[-target_peak, target_peak]``.
        sample_rate: Sample rate of :attr:`samples` in Hz.
        source_duration_s: Duration of the file before trimming, in seconds.
        trim_offset_s: Offset of the first retained sample in the source file.
    """

    samples: np.ndarray
    sample_rate: int
    source_duration_s: float
    trim_offset_s: float = 0.0

    @property
    def duration_s(self) -> float:
        """Duration of the retained audio in seconds."""
        return len(self.samples) / float(self.sample_rate)

    def __len__(self) -> int:
        return int(self.samples.shape[0])


def _read_native(data: np.ndarray, dtype: str | None) -> tuple[np.ndarray, int]:
    """Read audio already decoded into memory, as a float32 numpy array."""
    y = np.asarray(data)
    if np.issubdtype(y.dtype, np.integer):
        info_bits = 16 if y.dtype.itemsize <= 2 else 32
        scale = float(1 << (info_bits - 1))
        y = y.astype(np.float32) / scale
    elif not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)
    if dtype is not None:
        y = y.astype(np.dtype(dtype), copy=False)
    return y, int(y.shape[0])


def to_mono(y: np.ndarray) -> np.ndarray:
    """Average a multichannel signal down to a single channel.

    Args:
        y: Array of shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
            Channel-major ``(n_channels, n_samples)`` input is detected for
            arrays with few samples and many rows.

    Returns:
        A 1-D float32 array of length ``n_samples``.

    Raises:
        ValueError: If ``y`` is empty or has more than two dimensions.
    """
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {arr.shape}")
    if arr.shape[0] < arr.shape[1]:  # channel-major layout
        arr = arr.T
    return arr.mean(axis=1).astype(np.float32)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a signal to ``sr_out`` using a band-limited sinc kernel.

    Args:
        y: Input samples.
        sr_in: Input sample rate in Hz.
        sr_out: Target sample rate in Hz.

    Returns:
        Resampled float32 samples. If the rates match, the input is returned
        unchanged (no copy, no filtering).
    """
    if sr_in == sr_out:
        return np.asarray(y, dtype=np.float32)
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sample rates must be positive")
    # Imported lazily: librosa pulls in a fairly heavy import chain.
    import librosa

    return librosa.resample(
        np.asarray(y, dtype=np.float32), orig_sr=int(sr_in), target_sr=int(sr_out), res_type="soxr_hq"
    ).astype(np.float32)


def normalize(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalise a signal.

    Args:
        y: Input samples.
        target_peak: Desired absolute peak amplitude, in ``(0, 1]``.

    Returns:
        A scaled copy of the input. A silent signal is returned unchanged so
        that downstream stages see true silence rather than amplified noise.
    """
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target_peak must be in (0, 1]")
    arr = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0.0:
        return arr.copy()
    return (arr * (target_peak / peak)).astype(np.float32)


def frame_rms_db(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute short-time RMS level in dBFS for every frame.

    Args:
        y: Mono samples.
        frame_length: Window length in samples.
        hop_length: Step between windows in samples.

    Returns:
        1-D array of dB values with one entry per (possibly partial) frame.
        Silent frames return ``-120.0`` rather than ``-inf``.
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    frame_length = max(1, int(frame_length))
    hop_length = max(1, int(hop_length))
    n_frames = 1 + max(0, (arr.size - frame_length) // hop_length)
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        chunk = arr[start : start + frame_length]
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        out[i] = 20.0 * np.log10(rms) if rms > 0 else -120.0
    return out


def trim_silence(
    y: np.ndarray,
    sample_rate: int,
    trim_db: float = 40.0,
    frame_ms: float = 20.0,
    min_duration_s: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Trim leading and trailing silence using a short-time RMS threshold.

    The threshold is ``(global peak level) - trim_db``, so it adapts to the
    recording's own loudness. Internal silences are preserved: only the head
    and tail are cut.

    Args:
        y: Mono samples.
        sample_rate: Sample rate in Hz.
        trim_db: How far below the loudest frame a frame must be to count as
            silence, in dB.
        frame_ms: Analysis window for the level estimate, in milliseconds.
        min_duration_s: Never return less than this many samples.

    Returns:
        Tuple of ``(trimmed_samples, offset_seconds)`` where the offset is the
        position of the first returned sample within the input.
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.size == 0:
        return arr.copy(), 0.0
    hop = max(1, int(sample_rate * frame_ms / 1000.0))
    frame = hop * 2
    levels = frame_rms_db(arr, frame, hop)
    if levels.size == 0:
        return arr.copy(), 0.0
    threshold = float(levels.max()) - float(trim_db)
    # A digital-silence buffer has every frame at the -120 dB floor, so
    # "threshold = peak - trim_db" would still mark it all as loud. Treat a
    # peak at the floor as no signal at all.
    if float(levels.max()) <= -119.0:
        keep = min(arr.size, max(1, int(min_duration_s * sample_rate)))
        return arr[:keep].copy(), 0.0
    loud = np.flatnonzero(levels >= threshold)
    if loud.size == 0:  # everything below threshold: keep a minimal slice
        keep = min(arr.size, max(1, int(min_duration_s * sample_rate)))
        return arr[:keep].copy(), 0.0
    start_sample = int(loud[0] * hop)
    end_sample = min(arr.size, int(loud[-1] * hop) + frame)
    if end_sample - start_sample < int(min_duration_s * sample_rate):
        return arr.copy(), 0.0
    return arr[start_sample:end_sample].copy(), start_sample / float(sample_rate)


def load_bytes(
    data: bytes,
    params: AudioParams | None = None,
    filename: str = "upload.wav",
) -> LoadedAudio:
    """Load audio from an in-memory file (used by the web upload endpoint).

    Args:
        data: Raw encoded file contents.
        params: Audio parameters; defaults to :class:`~swaras.config.AudioParams`.
        filename: Original filename, used only to infer the container format.

    Returns:
        The loaded, resampled, normalised, trimmed audio.

    Raises:
        ValueError: If the bytes cannot be decoded as audio.
    """
    params = params or AudioParams()
    ext = Path(filename).suffix.lower()
    y, sr = _decode(io.BytesIO(data), ext)
    return _finalize(y, sr, params)


def load_file(path: str | Path, params: AudioParams | None = None) -> LoadedAudio:
    """Load an audio file from disk.

    WAV/FLAC/OGG/AIFF are read with ``soundfile``; anything else (for example
    MP4 or M4A containers) is decoded by ``ffmpeg`` and handed to ``soundfile``
    as raw PCM. This keeps ``ffmpeg`` optional rather than mandatory.

    Args:
        path: Path to the audio file.
        params: Audio parameters; defaults to :class:`~swaras.config.AudioParams`.

    Returns:
        The loaded, resampled, normalised, trimmed audio.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be decoded as audio.
    """
    params = params or AudioParams()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    ext = p.suffix.lower()
    if ext in _NATIVE_FORMATS:
        try:
            data, sr = sf.read(str(p), always_2d=False, dtype="float32")
            return _finalize(data, sr, params)
        except Exception as exc:  # fall through to ffmpeg
            logger.debug("soundfile failed on %s (%s); trying ffmpeg", p, exc)
    data, sr = _decode_with_ffmpeg(p)
    return _finalize(data, sr, params)


def _decode(stream: BinaryIO, ext: str) -> tuple[np.ndarray, int]:
    """Decode an audio byte stream with soundfile."""
    try:
        data, sr = sf.read(stream, always_2d=False, dtype="float32")
    except Exception as exc:
        raise ValueError(f"could not decode audio: {exc}") from exc
    return data, int(sr)


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode an arbitrary media file by piping ffmpeg output to soundfile."""
    import subprocess

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError(
            f"cannot decode {path.suffix} without ffmpeg installed; "
            "convert the file to wav/flac first"
        ) from exc
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"could not decode {path.name}: {proc.stderr.decode(errors='replace')[:300]}")
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="int16")
    data = data.astype(np.float32) / 32768.0
    return data, int(sr)


def _finalize(y: np.ndarray, sr: int, params: AudioParams) -> LoadedAudio:
    """Apply mono conversion, resampling, normalisation and trimming."""
    mono = to_mono(y)
    source_duration = len(mono) / float(sr)
    resampled = resample(mono, sr, params.sample_rate)
    normalized = normalize(resampled, params.target_peak)
    trimmed, offset = trim_silence(
        normalized, params.sample_rate, params.trim_db, min_duration_s=params.min_duration_s
    )
    return LoadedAudio(
        samples=trimmed,
        sample_rate=params.sample_rate,
        source_duration_s=source_duration,
        trim_offset_s=offset,
    )
