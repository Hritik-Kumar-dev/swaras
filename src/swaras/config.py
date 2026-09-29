"""Global configuration and default parameters for the swaras pipeline.

All tunable constants live here so that the behaviour of every stage can be
adjusted without editing algorithm code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Directory that ships package data (shruti tables, etc.).
PACKAGE_DIR: Path = Path(__file__).resolve().parent
CONFIG_DIR: Path = PACKAGE_DIR / "config"
DEFAULT_SHRUTI_TABLE: Path = CONFIG_DIR / "shruti_table.json"

#: Internal analysis sample rate for the whole pipeline.
SAMPLE_RATE: int = 22_050

#: Analysis frame size in samples at :data:`SAMPLE_RATE`. 1024 samples is
#: 46.4 ms, which balances two competing errors: a window that is too long
#: blurs a glissando (pYIN reported 26 cents of lag on an octave sweep at 2048
#: samples, versus 11 cents at 1024), and a window that is too short cannot
#: contain two periods of the lowest supported pitch. At fmin = 65 Hz two
#: periods need 679 samples, so 1024 keeps a safe margin.
N_FFT: int = 1024
HOP_LENGTH: int = 512


@dataclass(frozen=True)
class AudioParams:
    """Parameters for the audio loading stage."""

    sample_rate: int = SAMPLE_RATE
    #: Peak-normalise the signal to this amplitude (0 < target <= 1).
    target_peak: float = 0.95
    #: Trim leading/trailing silence if quieter than this (dBFS).
    trim_db: float = 40.0
    #: Minimum voiced-region length after trimming, in seconds.
    min_duration_s: float = 0.05


@dataclass(frozen=True)
class PitchParams:
    """Parameters for F0 estimation.

    Defaults are tuned for a solo voice: roughly C2 (65 Hz, low Sa) to
    C6 (1046 Hz, high Sa' plus headroom).
    """

    sample_rate: int = SAMPLE_RATE
    fmin: float = 65.0
    fmax: float = 1100.0
    frame_length: int = N_FFT
    hop_length: int = HOP_LENGTH
    #: Median-filter width in frames used to remove octave-jump spikes.
    median_width_frames: int = 5
    #: A frame is voiced if pYIN voiced prob exceeds this.
    voiced_prob_threshold: float = 0.5
    #: Cents deviation tolerance (per frame) to fold octave errors.
    octave_tolerance_cents: float = 55.0
    #: pYIN's measured systematic bias, subtracted from the F0 track. Measured
    #: on steady sine tones with frame_length 2048, the bias is a smooth
    #: function of pitch, not noise: about +9.2 cents below 250 Hz, +4.2 cents
    #: at 261 Hz, and about -0.8 cents above 293 Hz. Essentia's Melodia is
    #: accurate to well under 1 cent over the same range, which is the main
    #: reason it is the default tracker. Set to 0.0 to leave the track raw.
    pyin_bias_correction_cents: float = 0.0
    # --- Essentia Melodia specific -------------------------------------
    #: Number of harmonics used by Melodia's salience model. Lower values are
    #: more robust for a single voice; Essentia's default is 20.
    melodia_number_harmonics: int = 6
    #: Spectral magnitude threshold in dB (Melodia's own default is 40).
    melodia_magnitude_threshold: float = 40.0
    #: A weaker frame must be within this factor of the best salience in its
    #: neighbourhood to stay voiced.
    melodia_voicing_tolerance: float = 0.2
    #: Frames shorter than this (ms) are not reported as voiced notes.
    melodia_min_duration_ms: float = 100.0
    #: Melodia's pitch-continuity threshold, in cents. Essentia's default is
    #: 27.5625, which is tighter than a singer's vibrato and than the size of
    #: an ordinary melodic leap: on a Sa-Pa-Sa'-Ni' phrase at 220 Hz it left
    #: only 19% of frames voiced, because each new note failed the continuity
    #: test against the previous one and the whole phrase was discarded as
    #: unvoiced. At 100 cents the same phrase reaches 86% voiced, and a
    #: continuous sine sweep is unaffected (identical 16.2 cent median error
    #: at 27.6, 100 and 200 cents) since a sweep never jumps discontinuously.
    melodia_pitch_continuity_cents: float = 100.0
    #: Melodia's time-continuity threshold, in ms (Essentia's default).
    melodia_time_continuity_ms: float = 100.0


@dataclass(frozen=True)
class TonicParams:
    """Parameters for Sa (tonic) detection."""

    #: Width of the histogram kernel, in cents.
    bin_width_cents: float = 1.0
    #: Gaussian smoothing sigma in cents applied to the histogram.
    smooth_sigma_cents: float = 18.0
    #: Ignore F0 frames outside this range when building the histogram.
    fmin_hz: float = 60.0
    fmax_hz: float = 1200.0
    #: Interval above Sa scored as a strong corroborating peak (Pa).
    pa_cents: float = 702.0
    #: Weights for [Sa, Pa, octave] peaks in the candidate score.
    w_sa: float = 1.0
    w_pa: float = 0.55
    w_octave: float = 0.45
    #: Weight of the lowest-note prior. Peak scoring alone cannot separate Sa
    #: from Re in a scale run that touches every degree, because every
    #: candidate has an equally strong Sa peak and a fifth above it. The tonic
    #: of a scale is its lowest note, so this term breaks such ties in favour
    #: of the lowest well-supported candidate. Set to 0.0 to disable.
    w_lowest: float = 0.12
    #: A candidate is treated as "the lowest note" when it lies within this
    #: many cents of the lowest strongly-supported pitch in the recording.
    lowest_tolerance_cents: float = 60.0
    #: How quickly the lowest-note credit decays above that tolerance, in
    #: cents. A slow decay (a full octave) gives too little separation to
    #: break the scale-run tie, so this is deliberately much shorter than an
    #: octave: a note a third above the lowest is already a weak tonic guess.
    lowest_falloff_cents: float = 240.0
    #: Tolerance for treating two candidates as octave-equivalent, in cents.
    #: Candidate positions come off a 1-cent grid, so an exact octave
    #: comparison fails by a fraction of a cent.
    octave_equivalence_cents: float = 3.0
    #: A candidate is rejected if its Sa-peak is below this fraction of best.
    min_peak_ratio: float = 0.18
    #: Only consider Sa candidates in this Hz range.
    sa_min_hz: float = 55.0
    sa_max_hz: float = 800.0
    #: Number of candidates to report.
    top_k: int = 3
    #: Reference frequency for the cents histogram axis, chosen low enough
    #: that the whole search range is a positive integer number of octaves.
    histogram_ref_hz: float = 27.5
    #: Minimum separation between reported candidates, in cents, so that a
    #: broad peak does not fill the top-3 list with near-duplicates.
    candidate_separation_cents: float = 45.0
    #: Among candidates that are octave-equivalent and score within this
    #: fraction of the best, the lowest is preferred. A melody that both
    #: starts and ends on Sa gives Sa and Sa' near-identical evidence; the
    #: lower one is the safer reading, and the upper stays visible among the
    #: alternatives for the user to override.
    octave_collapse_ratio: float = 0.9
    #: Minimum separation from a histogram peak for a candidate to be scored.
    #: Below this the Sa term is treated as absent rather than negative.
    min_peak_prominence_cents: float = 12.0
    # --- Essentia TonicIndianArtMusic specific -------------------------
    #: Enable the Essentia tonic algorithm (it needs a drone, see README).
    use_essentia: bool = True
    #: Essentia's tonic search range; its default ceiling of 375 Hz would
    #: miss a Sa at A4 or above, so we widen it.
    essentia_min_tonic_hz: float = 55.0
    essentia_max_tonic_hz: float = 800.0
    #: Salience peaks considered by Essentia's tonic model.
    essentia_number_salience_peaks: int = 5
    #: If Essentia's tonic lies within this many cents of a histogram
    #: candidate, the two are treated as agreeing and the candidate is
    #: boosted. Larger disagreements are reported separately rather than
    #: merged, because Essentia returns its search-range floor on audio
    #: without a drone.
    essentia_agree_cents: float = 50.0
    #: Score multiplier applied to a candidate that Essentia also supports.
    essentia_agreement_boost: float = 1.15
    #: Essentia's answer is discarded if it sits at the very edge of its
    #: search range, which is its failure mode on drone-less audio.
    essentia_reject_edge_hz: float = 5.0


@dataclass(frozen=True)
class SegmentParams:
    """Parameters for turning a continuous contour into discrete notes."""

    #: Minimum duration of a stable note, in seconds.
    min_note_s: float = 0.10
    #: How far a frame may wander from the region it belongs to before the
    #: region ends, in cents. This is sized for vibrato and andolan, not for
    #: note-to-note motion: a trained singer rocks a steady note by 50 to 100
    #: cents several times a second, and at the 40 cents that separate two
    #: swara positions a deep vibrato would shatter every note into fragments.
    #: A glide crosses hundreds of cents, so it still ends the region.
    vibrato_tolerance_cents: float = 80.0
    #: Cents tolerance for a note to count as a settled pitch rather than a
    #: drift, and for deciding whether a region is a tracking artefact. Tighter
    #: than ``vibrato_tolerance_cents`` because it compares distinct events.
    stability_cents: float = 40.0
    #: Fraction of a region's frames that must lie within ``stability_cents``
    #: of its median. This is what separates a note from a slow meend: a glide
    #: is chopped into short regions by the growth tolerance, and each slice is
    #: locally flat, but a slice of a rising ramp is not flat to within 40
    #: cents over its own length, so the fraction test rejects it.
    stability_fraction: float = 0.8
    #: Median filter width used for a region's pitch, in frames. Filtering in
    #: the cents domain makes this pitch-scale invariant.
    #:
    #: The width is set by the vibrato rate, not by the analysis window. At a
    #: 22.05 kHz sample rate and hop 512 the frame rate is 43 Hz, so a 5 Hz
    #: vibrato has a half-period of 4.3 frames and a five-frame median cannot
    #: touch it: a 25-cent vibrato then fragmented a phrase into 16 notes
    #: instead of 12. A width of 9 covers a full period and gives the correct
    #: 12, and widths from 7 to 15 all agree, so the result is not sensitive
    #: to the exact value.
    smooth_frames: int = 9
    #: How far a region may sit from the pitch around it before it is
    #: discarded as a tracking artefact, as a multiple of ``min_note_s``. A
    #: region shorter than this that is perfectly steady in isolation but sits
    #: between two notes which agree with each other is a glitch, not a note.
    #: Length is the only thing that separates the two cases: a three-frame
    #: octave spike is surrounded by one pitch, and so is a deliberately held
    #: Sa' between two Sa's, but nobody holds an octave for 0.2 s in the
    #: middle of a note and means it.
    outlier_max_note_ratio: float = 2.0
    #: Merge notes closer together than this in pitch, in cents.
    merge_cents: float = 45.0
    #: Any note shorter than this is absorbed into its neighbour.
    blip_s: float = 0.05
    #: Two notes at the same pitch separated by a shorter gap than this are
    #: joined into one. A few frames of lost voicing in the middle of a held
    #: note is a tracker dropout, not a re-articulation; a real re-articulation
    #: needs a consonant and a breath, which is longer than this.
    same_pitch_gap_s: float = 0.12


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate of every stage's parameters."""

    audio: AudioParams = field(default_factory=AudioParams)
    pitch: PitchParams = field(default_factory=PitchParams)
    tonic: TonicParams = field(default_factory=TonicParams)
    segment: SegmentParams = field(default_factory=SegmentParams)
    #: Path of the shruti table to use.
    shruti_table: Path = DEFAULT_SHRUTI_TABLE
    #: If set, overrides detected Sa.
    sa_hz: float | None = None


DEFAULT_CONFIG = PipelineConfig()
