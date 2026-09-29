"""Stage 1b: F0 (pitch) tracking.

The public surface is deliberately small:

* :class:`PitchContour` -- the result: frame times, F0 in Hz, voiced flags and
  a per-frame confidence.
* :class:`PitchDetector` -- abstract base class. Swap in CREPE (or any other
  model) by subclassing and implementing :meth:`PitchDetector.estimate`.
* :class:`PyinPitchDetector` -- the default pYIN implementation.

Design rule: the contour stays continuous and float-valued here. No
quantisation to notes or swaras happens in this module.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from .cents import cents_between, cents_to_hz, hz_to_cents  # noqa: F401  (re-exported)
from .config import PitchParams

#: Silence floor used when a frame is declared unvoiced.
UNVOICED_F0 = 0.0


@dataclass
class PitchContour:
    """A frame-synchronous F0 estimate.

    Attributes:
        times: Frame centre times in seconds, shape ``(n_frames,)``, increasing.
        f0: Estimated F0 in Hz, shape ``(n_frames,)``. ``0.0`` marks unvoiced
            frames; use :attr:`voiced` to test validity.
        voiced: Boolean voiced flag per frame, shape ``(n_frames,)``.
        confidence: Per-frame voiced probability in ``[0, 1]``.
        hop_seconds: Time between consecutive frames in seconds.
        method: Name of the algorithm that produced the contour.
    """

    times: np.ndarray
    f0: np.ndarray
    voiced: np.ndarray
    confidence: np.ndarray
    hop_seconds: float
    method: str = "pyin"
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.f0 = np.asarray(self.f0, dtype=np.float64)
        self.voiced = np.asarray(self.voiced, dtype=bool)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        n = self.times.shape[0]
        if not (self.f0.shape == self.voiced.shape == self.confidence.shape == (n,)):
            raise ValueError("times, f0, voiced and confidence must all have the same length")

    @property
    def n_frames(self) -> int:
        """Number of analysis frames."""
        return int(self.times.shape[0])

    @property
    def duration_s(self) -> float:
        """Time span covered by the contour in seconds."""
        return float(self.times[-1] - self.times[0] + self.hop_seconds) if self.n_frames else 0.0

    @property
    def voiced_fraction(self) -> float:
        """Fraction of frames marked voiced."""
        return float(np.mean(self.voiced)) if self.n_frames else 0.0

    def voiced_f0(self) -> np.ndarray:
        """F0 values of voiced frames only."""
        return self.f0[self.voiced]

    def voiced_confidence(self) -> np.ndarray:
        """Confidence values of voiced frames only."""
        return self.confidence[self.voiced]

    def set_f0(self, f0: np.ndarray) -> None:
        """Replace the F0 track, keeping the voiced mask consistent.

        Any non-positive or non-finite value is treated as unvoiced.
        """
        f0 = np.asarray(f0, dtype=np.float64)
        if f0.shape != self.f0.shape:
            raise ValueError("replacement f0 must match the existing shape")
        self.f0 = f0
        self.voiced = np.isfinite(f0) & (f0 > 0)
        if not self.voiced.any():
            return

    def smoothed_f0(self, width_frames: int = 5, octave_tolerance_cents: float = 55.0) -> np.ndarray:
        """Return a median-filtered, octave-corrected copy of the F0 track.

        The median filter runs in the cents domain (equivalent to a geometric
        median in Hz) and ignores unvoiced frames. Octave errors -- frames
        that land on a near-integer multiple of 1200 cents away from their
        neighbours -- are pulled back onto the local contour.

        Args:
            width_frames: Median filter width; rounded up to an odd number.
            octave_tolerance_cents: How close (in cents) a frame must be to a
                1200-cent multiple of its local reference to be snapped.

        Returns:
            A new array of the same length as :attr:`f0`, with unvoiced
            positions still zero.
        """
        return correct_octave_jumps(
            median_filter_f0(self.f0, self.voiced, width_frames),
            self.voiced,
            octave_tolerance_cents=octave_tolerance_cents,
        )


def median_filter_f0(f0: np.ndarray, voiced: np.ndarray, width_frames: int = 5) -> np.ndarray:
    """Median-filter an F0 track in the cents domain, ignoring unvoiced frames.

    Filtering in cents rather than Hz makes the operation pitch-scale
    invariant, so a 5-frame filter smooths vibrato equally at 100 Hz and
    1 kHz.

    Args:
        f0: F0 track in Hz (``0.0`` for unvoiced).
        voiced: Boolean voiced mask.
        width_frames: Filter width; values below 3 disable filtering.

    Returns:
        A new float64 array of the same shape as ``f0``.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    out = np.zeros_like(f0)
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        return out
    width = max(3, int(width_frames))
    if width % 2 == 0:
        width += 1
    if idx.size < width:
        out[idx] = f0[idx]
        return out
    cents = 1200.0 * np.log2(f0[idx])
    padded = np.pad(cents, width // 2, mode="edge")
    filtered = median_filter(padded, size=width, mode="nearest")[width // 2 : width // 2 + cents.size]
    out[idx] = 2.0 ** (filtered / 1200.0)
    return out


def correct_octave_jumps(
    f0: np.ndarray,
    voiced: np.ndarray,
    octave_tolerance_cents: float = 55.0,
    max_run_frames: int = 4,
) -> np.ndarray:
    """Snap frames onto the local contour when they differ by a near-octave.

    A pitch tracker occasionally locks onto a wrong harmonic, reporting a
    pitch an octave (or a twelfth) away from the note actually being sung. Such
    a frame sits a multiple of 1200 cents from the local median but only a few
    cents away once that multiple is removed, which is what this tests.

    The correction is limited to short runs. A three-frame octave spike inside
    a note is an error; a whole note an octave away is a real note, and
    removing its octave would erase it. Comparing each frame only with its
    immediate neighbours cannot make that distinction, so the run-length limit
    does it explicitly.

    Args:
        f0: F0 track in Hz.
        voiced: Boolean voiced mask.
        octave_tolerance_cents: Allowed residual after removing the octave
            multiple.
        max_run_frames: Longest run that may be treated as an error. Longer
            runs are left alone.

    Returns:
        A corrected copy of ``f0``.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    out = f0.copy()
    idx = np.flatnonzero(voiced)
    if idx.size < 3:
        return out
    # Filter the full-length track, not the voiced slice. Selecting voiced
    # frames first and filtering afterwards concatenates non-adjacent frames,
    # so the "local" median spans a whole note instead of a neighbourhood.
    #
    # The reference is the global median of the voiced track, not 1 Hz:
    # 1200*log2(f) is a valid octave-invariant coordinate, but it must be
    # compared against a median at the same baseline, and using 1 Hz while
    # later reasoning in Sa-relative cents mixes the two. One constant
    # reference for the whole function keeps the arithmetic self-consistent.
    voiced_f0 = f0[voiced]
    if voiced_f0.size == 0:
        return out
    reference = float(np.median(voiced_f0))
    work = np.where(voiced, f0, np.nan)
    cents = 1200.0 * np.log2(work / reference)
    finite = np.isfinite(cents)
    if finite.sum() < 3:
        return out
    # Carry the last value across unvoiced frames so gaps do not widen the
    # effective window across a whole phrase.
    filled = _forward_fill(cents, finite)
    # Local reference: running median of the voiced cents track. Its only job
    # is to *detect* candidate glitches, so the window must be wide enough that
    # the error frames cannot dominate their own reference. The detection
    # tolerance is deliberately loose, and the rolling median is deliberately
    # not used to measure the correction: a rolling median lags a sustained
    # offset, which inflated the residual of the trailing frames of a 3-frame
    # spike to 60 cents and pushed them past a 55-cent test, so only the first
    # frame of the spike was corrected.
    width = 2 * int(max_run_frames) + 3
    if width % 2 == 0:
        width += 1
    width = min(width, cents.size if cents.size % 2 == 1 else cents.size - 1)
    if width < 3:
        return out
    padded = np.pad(filled, width // 2, mode="edge")
    local_median = median_filter(padded, size=width, mode="nearest")[width // 2 : width // 2 + filled.size]
    delta = filled - local_median
    multiples = np.round(delta / 1200.0)
    detect_tolerance = max(float(octave_tolerance_cents), 200.0)
    candidate = (
        finite & (multiples != 0) & (np.abs(delta - multiples * 1200.0) <= detect_tolerance)
    )
    if not np.any(candidate):
        return out

    runs = _merge_runs(_mask_runs(candidate), max_gap=max_run_frames)
    fixed = filled.copy()
    for start, end in runs:
        if end - start > max_run_frames:
            # A long run is a real note an octave away, not a glitch.
            continue
        # Measure the correction against the run's *clean neighbours*, which
        # belong to the note the glitch interrupts, rather than against a
        # rolling median that is itself contaminated.
        pad = max(2, int(max_run_frames))
        lo, hi = max(0, start - pad), min(filled.size, end + pad)
        neighbour = np.concatenate([filled[lo:start], filled[end:hi]])
        neighbour = neighbour[np.isfinite(neighbour)]
        if neighbour.size == 0:
            continue
        target = float(np.median(neighbour))
        run_median = float(np.median(cents[start:end]))
        offset = run_median - target
        shift = int(round(offset / 1200.0)) * 1200.0
        if shift == 0:
            continue
        # Confirm with the strict tolerance. This is what stops a genuine
        # octave leap from being folded away: after removing the octave, the
        # note has to land back on its neighbours.
        if abs(offset - shift) > octave_tolerance_cents:
            continue
        fixed[start:end] = cents[start:end] - shift
    result = np.where(finite, reference * 2.0 ** (fixed / 1200.0), np.nan)
    return np.where(finite, result, f0)


def _forward_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Carry the last valid value forward over gaps.

    Args:
        values: Track, possibly containing ``nan``.
        valid: Boolean mask of valid samples.

    Returns:
        A filled track. Leading gaps take the first valid value.
    """
    v = np.asarray(values, dtype=np.float64)
    if not valid.any():
        return np.zeros_like(v)
    known = np.flatnonzero(valid)
    out = np.array(v, dtype=np.float64, copy=True)
    out[~valid] = np.interp(np.flatnonzero(~valid), known, v[known])
    return out


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Group a boolean mask into maximal runs of ``True``.

    Args:
        mask: Boolean array.

    Returns:
        List of ``(start, end)`` index pairs, end exclusive.
    """
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2].tolist(), changes[1::2].tolist()))


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Join runs separated by no more than ``max_gap`` frames.

    Args:
        runs: ``(start, end)`` pairs, ordered and non-overlapping.
        max_gap: Largest gap to bridge.

    Returns:
        The merged pairs.
    """
    if not runs or max_gap <= 0 or len(runs) == 1:
        return list(runs)
    out = [tuple(runs[0])]  # type: ignore[list-item]
    for start, end in runs[1:]:
        if start - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


class PitchDetector(abc.ABC):
    """Abstract F0 estimator.

    Subclass and implement :meth:`estimate` to plug in a different tracker
    (CREPE, pWORLD, an autocorrelation tracker, ...) without touching any
    downstream stage.
    """

    #: Short identifier used in :attr:`PitchContour.method` and CLI output.
    name: str = "abstract"

    @abc.abstractmethod
    def estimate(self, samples: np.ndarray, sample_rate: int) -> PitchContour:
        """Estimate F0 for a mono signal.

        Args:
            samples: Mono float samples, nominally in ``[-1, 1]``.
            sample_rate: Sample rate of ``samples`` in Hz.

        Returns:
            The estimated pitch contour.
        """

    def __call__(self, samples: np.ndarray, sample_rate: int) -> PitchContour:
        return self.estimate(np.asarray(samples, dtype=np.float32), sample_rate)


class PyinPitchDetector(PitchDetector):
    """F0 tracker built on ``librosa.pyin`` (probabilistic YIN)."""

    name = "pyin"

    def __init__(self, params: PitchParams | None = None) -> None:
        self.params = params or PitchParams()

    def estimate(self, samples: np.ndarray, sample_rate: int) -> PitchContour:
        """Run pYIN over ``samples``.

        Args:
            samples: Mono float samples.
            sample_rate: Sample rate in Hz.

        Returns:
            A :class:`PitchContour` whose ``f0`` is the raw pYIN estimate
            (``0.0`` where unvoiced). Smoothing is available on demand via
            :meth:`PitchContour.smoothed_f0`; the raw track is kept so that
            later stages can choose their own filtering.
        """
        import librosa

        p = self.params
        if len(samples) == 0:
            return PitchContour(
                times=np.zeros(0), f0=np.zeros(0), voiced=np.zeros(0, dtype=bool),
                confidence=np.zeros(0), hop_seconds=p.hop_length / sample_rate, method=self.name,
            )
        f0, voiced_flag, voiced_prob = librosa.pyin(
            np.asarray(samples, dtype=np.float32),
            fmin=p.fmin,
            fmax=p.fmax,
            sr=int(sample_rate),
            frame_length=int(p.frame_length),
            hop_length=int(p.hop_length),
            fill_na=UNVOICED_F0,
        )
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=UNVOICED_F0)
        voiced = np.asarray(voiced_flag, dtype=bool) & (f0 > 0)
        f0 = np.where(voiced, f0, UNVOICED_F0)
        n = f0.shape[0]
        hop_seconds = p.hop_length / float(sample_rate)
        # Report frame centre times so that a note onset is not reported half
        # a window early.
        times = (np.arange(n) * p.hop_length + p.frame_length / 2.0) / float(sample_rate)
        return PitchContour(
            times=times,
            f0=f0,
            voiced=voiced,
            confidence=np.clip(np.asarray(voiced_prob, dtype=np.float64), 0.0, 1.0),
            hop_seconds=hop_seconds,
            method=self.name,
        )


class MelodiaPitchDetector(PitchDetector):
    """F0 tracker built on Essentia's ``PredominantPitchMelodia``.

    Melodia models a pitch as a peak plus a harmonic comb, so it tolerates
    accompaniment and reports a pitch even when the fundamental is missing.
    It is the better choice for real songs and for a voice recorded over
    instruments; :class:`PyinPitchDetector` is usually sharper on a dry solo
    voice. Requires Essentia (Linux/macOS).
    """

    name = "melodia"

    def __init__(self, params: PitchParams | None = None) -> None:
        self.params = params or PitchParams()

    def _build(self) -> object:
        """Instantiate and configure the underlying Essentia algorithm.

        Returns:
            A configured ``essentia.standard.PredominantPitchMelodia``.

        Raises:
            RuntimeError: If Essentia is not installed.
        """
        from ._essentia import configure_algo, require_essentia

        es = require_essentia("Melodia F0 tracking")
        p = self.params
        algo = es.PredominantPitchMelodia()
        configure_algo(
            algo,
            sampleRate=int(p.sample_rate),
            frameSize=int(p.frame_length),
            hopSize=int(p.hop_length),
            minFrequency=float(p.fmin),
            maxFrequency=float(p.fmax),
            numberHarmonics=int(p.melodia_number_harmonics),
            # These three are INTEGER parameters in Essentia; passing a float
            # makes configure() raise, so send them as ints.
            magnitudeThreshold=int(p.melodia_magnitude_threshold),
            voicingTolerance=float(p.melodia_voicing_tolerance),
            minDuration=int(p.melodia_min_duration_ms),
            pitchContinuity=float(p.melodia_pitch_continuity_cents),
            timeContinuity=int(p.melodia_time_continuity_ms),
        )
        return algo

    def estimate(self, samples: np.ndarray, sample_rate: int) -> PitchContour:
        """Run Melodia over ``samples``.

        Args:
            samples: Mono float samples.
            sample_rate: Sample rate in Hz.

        Returns:
            A :class:`PitchContour`. Melodia emits ``0.0`` for unvoiced frames
            and a salience-based confidence in ``[0, 1]``; the voiced mask is
            ``f0 > 0``.
        """
        from ._essentia import as_essentia_signal

        p = self.params
        if len(samples) == 0:
            return PitchContour(
                times=np.zeros(0), f0=np.zeros(0), voiced=np.zeros(0, dtype=bool),
                confidence=np.zeros(0), hop_seconds=p.hop_length / sample_rate, method=self.name,
            )
        algo = self._build()
        f0, confidence = algo(as_essentia_signal(samples))
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=UNVOICED_F0)
        confidence = np.clip(np.nan_to_num(np.asarray(confidence, dtype=np.float64)), 0.0, 1.0)
        # Melodia's frame grid is start-anchored; shift to frame centres so
        # that reported times match pYIN's convention.
        f0, confidence = _pad_to_match(f0, confidence, len(samples), p.hop_length)
        voiced = f0 > 0
        f0 = np.where(voiced, f0, UNVOICED_F0)
        n = f0.shape[0]
        # Same frame-centre convention as PyinPitchDetector, deliberately not
        # clipped to the signal length: the trailing frames hang off the end,
        # but keeping both trackers on an identical grid is worth more than a
        # tidy last timestamp.
        times = (np.arange(n) * p.hop_length + p.frame_length / 2.0) / float(sample_rate)
        return PitchContour(
            times=times,
            f0=f0,
            voiced=voiced,
            confidence=confidence,
            hop_seconds=p.hop_length / float(sample_rate),
            method=self.name,
        )


def _pad_to_match(
    f0: np.ndarray, confidence: np.ndarray, n_samples: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trim or pad an Essentia frame track to the pYIN frame count.

    Essentia centres its analysis window and emits ``floor(n / hop)`` frames,
    while ``librosa.pyin`` emits ``1 + floor((n - frame) / hop)``. Keeping the
    grids identical lets the two detectors be compared and combined frame by
    frame.

    Args:
        f0: Essentia F0 track.
        confidence: Essentia confidence track.
        n_samples: Length of the input signal in samples.
        hop_length: Hop size in samples.

    Returns:
        ``(f0, confidence)`` trimmed or zero-padded to the librosa frame count.
    """
    target = 1 + max(0, (n_samples - 1) // hop_length)
    n = f0.shape[0]
    if n == target:
        return f0, confidence
    if n > target:
        return f0[:target], confidence[:target]
    pad = target - n
    return (
        np.concatenate([f0, np.zeros(pad)]),
        np.concatenate([confidence, np.zeros(pad)]),
    )


#: Registry of built-in pitch detectors, by name.
DETECTORS: dict[str, type[PitchDetector]] = {
    PyinPitchDetector.name: PyinPitchDetector,
    MelodiaPitchDetector.name: MelodiaPitchDetector,
}


def get_detector(name: str = "melodia", params: PitchParams | None = None) -> PitchDetector:
    """Look up a pitch detector by name.

    Args:
        name: ``"melodia"`` (default, Essentia) or ``"pyin"`` (librosa).
        params: Parameters forwarded to the detector.

    Returns:
        A :class:`PitchDetector` instance.

    Raises:
        ValueError: If ``name`` is not a known detector.
    """
    key = name.strip().lower()
    if key not in DETECTORS:
        raise ValueError(f"unknown pitch detector {name!r}; available: {sorted(DETECTORS)}")
    return DETECTORS[key](params)
