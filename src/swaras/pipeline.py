"""End-to-end transcription pipeline.

Assembles the stages into one call so the CLI and the API share a single
implementation:

    audio_io -> pitch -> normalize -> segment -> swar -> format
                    \\-> tonic (needs F0 before cents can be computed)

Tonic detection sits between pitch and normalize because everything downstream
is tonic-relative: the cents track is meaningless until Sa is known.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import LoadedAudio, load_bytes, load_file
from .config import PipelineConfig
from .format import Transcription
from .normalize import contour_to_cents
from .pitch import PitchContour, get_detector
from .segment import Segmentation, segment_contour
from .shruti import notes_to_shrutis
from .swar import SwaraNote, segment_to_swaras
from .tonic import TonicResult, detect_tonic, manual_tonic
from .tuning import ShrutiTable, load_shruti_table

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for inspection and for the API.

    Attributes:
        transcription: The formatted result.
        audio: The loaded audio.
        contour: The raw F0 contour.
        cents_contour: F0 expressed in cents relative to Sa.
        segmentation: Notes and transitions.
        tonic: The full tonic result, including alternatives.
    """

    transcription: Transcription
    audio: LoadedAudio
    contour: PitchContour
    cents_contour: object
    segmentation: Segmentation
    tonic: TonicResult


def transcribe(
    samples: np.ndarray,
    sample_rate: int,
    config: PipelineConfig | None = None,
    sa_hz: float | None = None,
    detector: str | None = None,
    table: ShrutiTable | None = None,
) -> PipelineResult:
    """Transcribe a mono signal into swara notation.

    Args:
        samples: Mono float samples.
        sample_rate: Sample rate in Hz.
        config: Pipeline configuration; defaults to :data:`DEFAULT_CONFIG`.
        sa_hz: Sa in Hz. When given, detection is skipped entirely, which is
            what the UI does after the user corrects a detection.
        detector: Pitch detector name; defaults to the config's.
        table: Tuning table; defaults to the packaged one.

    Returns:
        The full pipeline result.

    Raises:
        ValueError: If no Sa could be established, or the audio has no
            detectable voiced pitch.
    """
    cfg = config or PipelineConfig()
    tracker = get_detector(detector or "melodia", cfg.pitch)
    contour = tracker.estimate(samples, sample_rate)

    if not contour.voiced.any():
        raise ValueError("no voiced pitch detected: is the audio silent or out of the 65-1100 Hz range?")

    if sa_hz is not None:
        tonic = manual_tonic(sa_hz)
    else:
        tonic = detect_tonic(contour, samples, sample_rate, cfg.tonic)
        if not tonic.is_valid:
            raise ValueError(f"could not detect Sa: {tonic.warning}")
        logger.info("Sa = %.2f Hz (confidence %.2f)", tonic.hz, tonic.confidence)

    cents = contour_to_cents(contour, tonic.hz)
    segmentation = segment_contour(cents, cfg.segment)
    tuning = table or load_shruti_table()
    notes: list[SwaraNote] = segment_to_swaras(segmentation.notes, tuning)
    # The shruti layer is a second reading of the same measurements, not a
    # competing one: a finer grid plus the distance from it. Both are reported.
    shrutis = notes_to_shrutis(notes, tuning)

    transcription = Transcription(
        notes=notes,
        shruti_notes=shrutis,
        sa_hz=tonic.hz,
        sa_source=tonic.method,
        sa_confidence=tonic.confidence,
        tonic_candidates=[c.to_dict() for c in tonic.candidates[1:4]],
        tonic_warning=tonic.warning,
        detector=contour.method,
        diagnostics=dict(segmentation.diagnostics),
    )
    return PipelineResult(
        transcription=transcription,
        audio=LoadedAudio(samples, sample_rate, len(samples) / sample_rate),
        contour=contour,
        cents_contour=cents,
        segmentation=segmentation,
        tonic=tonic,
    )


def transcribe_contour(
    contour: PitchContour,
    sa_hz: float,
    config: PipelineConfig | None = None,
    table: ShrutiTable | None = None,
    tonic: TonicResult | None = None,
    duration_s: float = 0.0,
) -> PipelineResult:
    """Re-transcribe an already-tracked contour, with Sa supplied.

    This is the path taken when the user corrects Sa in the UI. Everything
    after pitch tracking is a few milliseconds of array work, so re-running
    from a cached contour is what makes the correction feel instant; re-running
    the tracker would cost seconds and nothing about it would change.

    Args:
        contour: A pitch contour, typically from the contour cache.
        sa_hz: Sa in Hz, already decided by the caller.
        config: Pipeline configuration; defaults to :data:`DEFAULT_CONFIG`.
        table: Tuning table; defaults to the packaged one.
        tonic: A pre-built tonic result to report, if the caller has one.
        duration_s: Duration of the source audio, for reporting only. The
            re-tune path has no audio in hand, and the caller knows this from
            the first pass.

    Returns:
        The full pipeline result.
    """
    cfg = config or PipelineConfig()
    tonic_result = tonic if tonic is not None else manual_tonic(sa_hz)
    cents = contour_to_cents(contour, tonic_result.hz)
    segmentation = segment_contour(cents, cfg.segment)
    tuning = table or load_shruti_table()
    notes: list[SwaraNote] = segment_to_swaras(segmentation.notes, tuning)
    shrutis = notes_to_shrutis(notes, tuning)
    transcription = Transcription(
        notes=notes,
        shruti_notes=shrutis,
        sa_hz=tonic_result.hz,
        sa_source=tonic_result.method,
        sa_confidence=tonic_result.confidence,
        tonic_candidates=[c.to_dict() for c in tonic_result.candidates[1:4]],
        tonic_warning=tonic_result.warning,
        detector=contour.method,
        diagnostics=dict(segmentation.diagnostics),
    )
    return PipelineResult(
        transcription=transcription,
        # No audio is held on this path; the duration is carried through for
        # reporting rather than inventing an empty buffer.
        audio=LoadedAudio(np.zeros(0), 22050.0, duration_s),
        contour=contour,
        cents_contour=cents,
        segmentation=segmentation,
        tonic=tonic_result,
    )


def transcribe_file(
    path: str | Path,
    config: PipelineConfig | None = None,
    sa_hz: float | None = None,
    detector: str | None = None,
) -> PipelineResult:
    """Transcribe an audio file.

    Args:
        path: Path to the audio file.
        config: Pipeline configuration.
        sa_hz: Manual Sa in Hz, bypassing detection.
        detector: Pitch detector name.

    Returns:
        The full pipeline result.
    """
    cfg = config or PipelineConfig()
    audio = load_file(path, cfg.audio)
    return transcribe(audio.samples, audio.sample_rate, cfg, sa_hz, detector)


def transcribe_upload(
    data: bytes,
    filename: str = "upload.wav",
    config: PipelineConfig | None = None,
    sa_hz: float | None = None,
    detector: str | None = None,
) -> PipelineResult:
    """Transcribe uploaded audio bytes.

    Args:
        data: Raw encoded file contents.
        filename: Original filename, used to infer the container.
        config: Pipeline configuration.
        sa_hz: Manual Sa in Hz, bypassing detection.
        detector: Pitch detector name.

    Returns:
        The full pipeline result.
    """
    cfg = config or PipelineConfig()
    audio = load_bytes(data, cfg.audio, filename)
    return transcribe(audio.samples, audio.sample_rate, cfg, sa_hz, detector)
