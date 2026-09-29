"""Synthetic audio generators for tests and demos.

Everything here is pure NumPy so tests stay fast and deterministic. The
generators model the things that make real transcription hard: vibrato, portamento
between notes, octave-error-prone low fundamentals, breath noise, and
amplitude envelopes.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SR = 22_050


def cents_to_hz(cents: float, ref_hz: float = 261.63) -> float:
    """Convert cents above a reference to Hz (convenience re-export)."""
    return float(ref_hz) * 2.0 ** (float(cents) / 1200.0)


def add_vibrato(
    n_samples: int,
    f0: float,
    sample_rate: int = DEFAULT_SR,
    rate_hz: float = 5.0,
    depth_cents: float = 40.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Instantaneous frequency track with sinusoidal vibrato.

    Args:
        n_samples: Length of the track.
        f0: Centre frequency in Hz.
        sample_rate: Sample rate in Hz.
        rate_hz: Vibrato rate in Hz.
        depth_cents: Vibrato depth in cents (peak deviation).
        phase: Vibrato phase offset in radians.

    Returns:
        Instantaneous frequency in Hz, shape ``(n_samples,)``.
    """
    t = np.arange(n_samples) / float(sample_rate)
    lfo = np.sin(2.0 * np.pi * rate_hz * t + phase)
    return f0 * 2.0 ** (depth_cents * lfo / 1200.0)


def add_glide(
    n_samples: int,
    f_start: float,
    f_end: float,
    sample_rate: int = DEFAULT_SR,
    shape: str = "linear",
) -> np.ndarray:
    """Instantaneous frequency track gliding from ``f_start`` to ``f_end``.

    Args:
        n_samples: Length of the track.
        f_start: Starting frequency in Hz.
        f_end: Ending frequency in Hz.
        sample_rate: Sample rate in Hz.
        shape: ``"linear"`` interpolates in frequency, ``"exponential"`` (the
            default behaviour for a musical glissando) interpolates in cents.

    Returns:
        Instantaneous frequency in Hz, shape ``(n_samples,)``.
    """
    x = np.linspace(0.0, 1.0, n_samples)
    if shape == "linear":
        return f_start + (f_end - f_start) * x
    c0 = 1200.0 * np.log2(f_start)
    c1 = 1200.0 * np.log2(f_end)
    return 2.0 ** ((c0 + (c1 - c0) * x) / 1200.0)


def _integrate_phase(f_track: np.ndarray, sample_rate: int) -> np.ndarray:
    """Integrate an instantaneous-frequency track into phase in radians."""
    return 2.0 * np.pi * np.cumsum(f_track) / float(sample_rate)


def harmonic_tone(
    f_track: np.ndarray,
    sample_rate: int = DEFAULT_SR,
    n_harmonics: int = 12,
    harmonic_decay: float = 1.0,
    noise_db: float = -60.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a harmonic tone from an instantaneous-frequency track.

    A sawtooth-like harmonic series is used, which resembles a sung vowel far
    better than a pure sine and keeps Melodia's harmonic model happy.

    Args:
        f_track: Instantaneous frequency in Hz, one value per sample.
        sample_rate: Sample rate in Hz.
        n_harmonics: Number of harmonics to synthesise.
        harmonic_decay: Amplitude falloff exponent; ``1.0`` gives a sawtooth.
        noise_db: Broadband noise level relative to the tone, in dB.
        rng: Random generator for reproducible noise.

    Returns:
        Float64 samples, peak-normalised to 1.0.
    """
    phase = _integrate_phase(np.asarray(f_track, dtype=np.float64), sample_rate)
    out = np.zeros(phase.shape[0], dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        out += np.sin(h * phase) / (h**harmonic_decay)
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak
    if noise_db > -120.0:
        rng = rng or np.random.default_rng(0)
        noise_amp = 10.0 ** (noise_db / 20.0)
        out = out + noise_amp * rng.standard_normal(out.shape[0])
    return out


def sine_tone(
    f_track: np.ndarray, sample_rate: int = DEFAULT_SR, noise_db: float = -60.0
) -> np.ndarray:
    """Render a pure sine from an instantaneous-frequency track."""
    phase = _integrate_phase(np.asarray(f_track, dtype=np.float64), sample_rate)
    out = np.sin(phase)
    if noise_db > -120.0:
        rng = np.random.default_rng(0)
        out = out + (10.0 ** (noise_db / 20.0)) * rng.standard_normal(out.shape[0])
    return out


def adsr(
    n_samples: int,
    sample_rate: int = DEFAULT_SR,
    attack_s: float = 0.02,
    decay_s: float = 0.05,
    sustain_level: float = 0.8,
    release_s: float = 0.05,
) -> np.ndarray:
    """Generate a simple attack/decay/sustain/release amplitude envelope."""
    sr = float(sample_rate)
    a = max(1, int(attack_s * sr))
    d = max(1, int(decay_s * sr))
    r = max(1, int(release_s * sr))
    s = max(0, n_samples - a - d - r)
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, a),
            np.linspace(1.0, sustain_level, d),
            np.full(s, sustain_level),
            np.linspace(sustain_level, 0.0, r),
        ]
    )
    if env.size < n_samples:
        env = np.concatenate([env, np.full(n_samples - env.size, 0.0)])
    return env[:n_samples]


def sine_sweep(
    f_start: float = 80.0,
    f_end: float = 1000.0,
    duration_s: float = 6.0,
    sample_rate: int = DEFAULT_SR,
) -> np.ndarray:
    """Generate a logarithmic sine sweep, useful for verifying F0 tracking.

    Args:
        f_start: Starting frequency in Hz.
        f_end: Ending frequency in Hz.
        duration_s: Sweep duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Float64 samples, peak-normalised to 1.0.
    """
    n = int(duration_s * sample_rate)
    f_track = np.geomspace(f_start, f_end, n)
    return sine_tone(f_track, sample_rate)


def note(
    f0: float,
    duration_s: float = 0.5,
    sample_rate: int = DEFAULT_SR,
    vibrato_rate_hz: float = 5.0,
    vibrato_cents: float = 30.0,
    n_harmonics: int = 12,
    noise_db: float = -55.0,
    envelope: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Generate a single sustained note with vibrato and an envelope.

    Args:
        f0: Fundamental frequency in Hz.
        duration_s: Note duration in seconds.
        sample_rate: Sample rate in Hz.
        vibrato_rate_hz: Vibrato rate; ``0`` disables vibrato.
        vibrato_cents: Vibrato depth in cents.
        n_harmonics: Number of harmonics for the harmonic tone.
        noise_db: Broadband noise floor in dB.
        envelope: Apply an ADSR envelope.
        seed: Seed for the noise generator.

    Returns:
        Float64 samples for the note.
    """
    n = int(duration_s * sample_rate)
    f_track = add_vibrato(n, f0, sample_rate, vibrato_rate_hz or 1.0, vibrato_cents if vibrato_rate_hz else 0.0)
    y = harmonic_tone(f_track, sample_rate, n_harmonics, noise_db=noise_db, rng=np.random.default_rng(seed))
    return y * adsr(n, sample_rate) if envelope else y


def sequence(
    frequencies: list[float],
    note_duration_s: float = 0.5,
    gap_duration_s: float = 0.06,
    sample_rate: int = DEFAULT_SR,
    vibrato_cents: float = 0.0,
    vibrato_rate_hz: float = 5.0,
    n_harmonics: int = 12,
    noise_db: float = -55.0,
    seed: int = 0,
) -> np.ndarray:
    """Concatenate a sequence of notes separated by short silences.

    Args:
        frequencies: Note frequencies in Hz, in order.
        note_duration_s: Duration of each note.
        gap_duration_s: Silence inserted between consecutive notes.
        sample_rate: Sample rate in Hz.
        vibrato_cents: Vibrato depth in cents for each note.
        vibrato_rate_hz: Vibrato rate in Hz for each note.
        n_harmonics: Number of harmonics per note.
        noise_db: Broadband noise floor in dB.
        seed: Base seed; incremented per note for variety.

    Returns:
        Float64 samples of the full sequence.
    """
    parts = [
        note(
            f,
            note_duration_s,
            sample_rate,
            vibrato_rate_hz=vibrato_rate_hz,
            vibrato_cents=vibrato_cents,
            n_harmonics=n_harmonics,
            noise_db=noise_db,
            seed=seed + i,
        )
        for i, f in enumerate(frequencies)
    ]
    gap = np.zeros(int(gap_duration_s * sample_rate))
    out = parts[0] if parts else np.zeros(0)
    for p in parts[1:]:
        out = np.concatenate([out, gap, p])
    return out


def melodic_phrase(
    tonic_hz: float = 261.63,
    scale_cents: list[float] | None = None,
    repeat: int = 2,
    note_duration_s: float = 0.42,
    sample_rate: int = DEFAULT_SR,
    vibrato_cents: float = 25.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Generate a phrase that starts on Sa and outlines Pa, so tonic is clear.

    The default contour is a sa-re-ga-pa-dha-ni ascent, which gives the
    histogram method strong peaks at Sa, the fifth, and the octave.

    Args:
        tonic_hz: Sa in Hz.
        scale_cents: Semitone offsets in cents to use; defaults to a
            sa-re-ga-pa-dha-ni ascent.
        repeat: How many times to repeat the pattern.
        note_duration_s: Duration per note.
        sample_rate: Sample rate in Hz.
        vibrato_cents: Vibrato depth.
        seed: Base random seed.

    Returns:
        Tuple of ``(samples, frequencies)`` where ``frequencies`` lists the
        actual rendered pitches in Hz, in order.
    """
    scale_cents = scale_cents if scale_cents is not None else [0, 200, 400, 700, 900, 1100]
    freqs = [cents_to_hz(c, tonic_hz) for c in scale_cents] * max(1, repeat)
    y = sequence(
        freqs,
        note_duration_s=note_duration_s,
        sample_rate=sample_rate,
        vibrato_cents=vibrato_cents,
        seed=seed,
    )
    return y, freqs


def sine_wave_with_silence(
    f0: float = 220.0, duration_s: float = 1.0, sample_rate: int = DEFAULT_SR
) -> np.ndarray:
    """A steady tone with leading and trailing silence, for trim tests."""
    tone = sine_tone(np.full(int(duration_s * sample_rate), f0), sample_rate)
    pad = int(0.5 * sample_rate)
    return np.concatenate([np.zeros(pad), tone, np.zeros(pad)])


def write_wav(path: str, y: np.ndarray, sample_rate: int = DEFAULT_SR) -> str:
    """Write samples to a WAV file as float32 (no clipping on write)."""
    import soundfile as sf

    sf.write(path, np.asarray(y, dtype=np.float32), int(sample_rate))
    return path
