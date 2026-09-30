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


# --- realistic performance -------------------------------------------------
#
# Everything above is deliberately clean: steady tones, no vibrato, no
# dynamics, no accompaniment. That is enough to prove the arithmetic is right
# but it is much easier than a real performance. The generators below add the
# four things a khayal actually has and that the clean tests never exercise:
# a tanpura drone, vibrato that arrives after the note onset, meends between
# notes, and phrase-level dynamics.


def tanpura_drone(
    tonic_hz: float,
    duration_s: float,
    sample_rate: int = DEFAULT_SR,
    pluck_period_s: float = 0.9,
    fifth_hz: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """A tanpura-like drone: Sa and Pa plucked in turn, with a slow decay.

    A tanpura string is plucked, not bowed, so each note has an attack and a
    long decay, and the pair is re-plucked every second or so. That is the
    signal Essentia's ``TonicIndianArtMusic`` is designed to find, and the
    clean test tones give it nothing to work with, which is why it disagreed
    with the histogram on every synthetic case so far.

    Args:
        tonic_hz: Sa in Hz.
        duration_s: Length of the drone.
        sample_rate: Sample rate in Hz.
        pluck_period_s: Interval between plucks.
        fifth_hz: Pa in Hz; defaults to a perfect fifth above Sa.
        seed: Random seed for the small per-pluck pitch and level variation a
            real instrument has.

    Returns:
        Float64 samples.
    """
    rng = np.random.default_rng(seed)
    fifth_hz = fifth_hz if fifth_hz is not None else tonic_hz * 1.4983070768766815
    out = np.zeros(int(duration_s * sample_rate), dtype=np.float64)
    period = max(1, int(pluck_period_s * sample_rate))
    # A tanpura's strings are tuned slightly apart, so the two beat slowly.
    for start in range(0, out.size, period):
        for f in (tonic_hz, fifth_hz):
            length = min(period * 2, out.size - start)
            if length <= 0:
                continue
            t = np.arange(length) / sample_rate
            detune = 1.0 + rng.normal(0.0, 0.0006)   # about 1 cent
            decay = np.exp(-t / 0.55)
            # Rich, slowly rolling partials, the way a plucked string sounds.
            sig = np.zeros(length)
            for h, amp in ((1, 1.0), (2, 0.62), (3, 0.44), (4, 0.28), (5, 0.19), (6, 0.12)):
                sig += amp * np.sin(2 * np.pi * f * detune * h * t + h * 0.4)
            attack = np.minimum(1.0, t / 0.004)
            out[start : start + length] += sig * decay * attack * 0.05
    peak = np.max(np.abs(out)) if out.size else 0.0
    return out / peak * 0.5 if peak > 0 else out


def sung_note(
    f0: float,
    duration_s: float,
    sample_rate: int = DEFAULT_SR,
    vibrato_rate_hz: float = 5.5,
    vibrato_cents: float = 35.0,
    vibrato_delay_s: float = 0.18,
    level: float = 1.0,
    n_harmonics: int = 10,
    breath_db: float = -50.0,
    seed: int = 0,
) -> np.ndarray:
    """A sung note with delayed vibrato, breath noise and an onset transient.

    The vibrato starts *after* the onset, as a singer's does. A model that
    assumes steady pitch from the first frame will read the delay as a
    transition.

    Args:
        f0: Note pitch in Hz.
        duration_s: Note length in seconds.
        sample_rate: Sample rate in Hz.
        vibrato_rate_hz: Vibrato rate; ``0`` disables it.
        vibrato_cents: Vibrato depth in cents.
        vibrato_delay_s: How long before the vibrato starts.
        level: Peak level, for dynamics.
        n_harmonics: Harmonic count.
        breath_db: Breath noise floor in dB.
        seed: Random seed.

    Returns:
        Float64 samples.
    """
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    depth = np.zeros(n)
    if vibrato_rate_hz and vibrato_cents:
        ramp = np.clip((t - vibrato_delay_s) / max(vibrato_delay_s, 1e-6), 0.0, 1.0)
        depth = vibrato_cents * ramp * np.sin(2 * np.pi * vibrato_rate_hz * t)
    f_track = f0 * 2.0 ** (depth / 1200.0)
    y = harmonic_tone(f_track, sample_rate, n_harmonics, noise_db=breath_db,
                      rng=np.random.default_rng(seed))
    env = adsr(n, sample_rate, attack_s=0.035, decay_s=0.06, sustain_level=0.9, release_s=0.08)
    return y * env * level


def meend(
    f_from: float,
    f_to: float,
    duration_s: float,
    sample_rate: int = DEFAULT_SR,
    curve: str = "ease",
) -> np.ndarray:
    """A portamento between two notes, as sung rather than as a tone slide.

    Args:
        f_from: Starting pitch in Hz.
        f_to: Ending pitch in Hz.
        duration_s: Length of the glide.
        sample_rate: Sample rate in Hz.
        curve: ``"ease"`` accelerates then settles, which is how a meend
            actually arrives; ``"linear"`` is a plain slide.

    Returns:
        Float64 samples.
    """
    n = int(duration_s * sample_rate)
    x = np.linspace(0.0, 1.0, n)
    if curve == "ease":
        x = x * x * (3.0 - 2.0 * x)
    c0 = 1200.0 * np.log2(f_from)
    c1 = 1200.0 * np.log2(f_to)
    f_track = 2.0 ** ((c0 + (c1 - c0) * x) / 1200.0)
    y = harmonic_tone(f_track, sample_rate, n_harmonics=8, noise_db=-55.0)
    # Fade in and out so the glide does not click against its neighbours.
    fade = min(int(0.02 * sample_rate), n // 4)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return y


def khayal_phrase(
    tonic_hz: float = 261.63,
    degrees: list[int] | None = None,
    repeat: int = 1,
    note_s: float = 0.55,
    meend_s: float = 0.11,
    with_drone: bool = True,
    with_vibrato: bool = True,
    with_dynamics: bool = True,
    sample_rate: int = DEFAULT_SR,
) -> tuple[np.ndarray, list[float], float]:
    """A phrase built like a khayal, and the ground truth for it.

    Everything a steady-tone test leaves out: a drone underneath, vibrato that
    arrives late, meends between every pair of notes, and a phrase-level swell.
    The expected notation is derived from ``degrees`` regardless of what the
    generator actually produced, so a caller can compare the two.

    Args:
        tonic_hz: Sa in Hz.
        degrees: Cents offsets above Sa, in order.
        repeat: How many times to repeat the phrase.
        note_s: Length of each note.
        meend_s: Length of each meend; ``0`` disables them.
        with_drone: Include a tanpura drone.
        with_vibrato: Give each note vibrato.
        with_dynamics: Swell the phrase rather than holding it flat.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple ``(audio, expected_labels, duration_s)``, where
        ``expected_labels`` is the short notation for the whole phrase.
    """
    from swaras.swar import cents_to_swar
    from swaras.tuning import load_shruti_table

    degrees = degrees if degrees is not None else [0, 200, 400, 700, 900, 1100, 900, 700, 400, 200, 0]
    table = load_shruti_table()
    freqs = [cents_to_hz(c, tonic_hz) for c in degrees]
    labels = [cents_to_swar(c, table).label for c in degrees]

    parts: list[np.ndarray] = []
    for r in range(max(1, repeat)):
        n_notes = len(freqs)
        for i, f in enumerate(freqs):
            # Dynamics: a slow arc across the phrase, plus a little per note.
            if with_dynamics:
                arc = 0.75 + 0.25 * np.sin(np.pi * (i / max(1, n_notes - 1)))
                level = float(0.7 + 0.5 * arc)
            else:
                level = 1.0
            parts.append(
                sung_note(
                    f, note_s, sample_rate,
                    vibrato_rate_hz=5.5 if with_vibrato else 0.0,
                    vibrato_cents=35.0 if with_vibrato else 0.0,
                    level=level, seed=100 * r + i,
                )
            )
            if i + 1 < n_notes and meend_s > 0:
                parts.append(meend(f, freqs[i + 1], meend_s, sample_rate))
        if r + 1 < repeat:
            parts.append(np.zeros(int(0.35 * sample_rate)))   # breath between phrases

    voice = np.concatenate(parts) if parts else np.zeros(0)
    if with_drone:
        drone = tanpura_drone(tonic_hz, voice.size / sample_rate, sample_rate, seed=7)
        # Sit the voice clearly above the drone, as it is in a recording.
        voice = voice / max(np.max(np.abs(voice)), 1e-9) * 0.8
        voice = voice + drone[: voice.size]
    else:
        voice = voice / max(np.max(np.abs(voice)), 1e-9) * 0.8
    voice = np.concatenate([voice, np.zeros(int(0.3 * sample_rate))])
    return voice, labels * max(1, repeat), voice.size / sample_rate
