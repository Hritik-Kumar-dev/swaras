"""Stage 3a: continuous contour to stable notes and transitions.

The rule this module implements: a note is a *stable region* of the contour,
and everything between two stable regions is a *transition*. A meend is
therefore not a note in its own right, and by default only its target is
emitted.

A region counts as stable when it is at least ``min_note_s`` long and its
pitch varies by less than ``stability_cents`` around its own median. The
median, not the mean, is what collapses vibrato and andolan: it is unmoved by
the periodic wobble a singer adds, and it resists the single glitch that a
mean would chase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.ndimage import median_filter

from .config import SegmentParams
from .normalize import CentsContour


class SegmentKind(str, Enum):
    """What a segment represents.

    Attributes:
        NOTE: A stable pitch, long enough to count as a sung note.
        TRANSITION: A glide (meend), or a short unstable passage between notes.
        BLIP: A region too short to be a note; kept for diagnostics but not
            emitted as notation.
    """

    NOTE = "note"
    TRANSITION = "transition"
    BLIP = "blip"


@dataclass(frozen=True)
class Segment:
    """A contiguous run of frames with one behaviour.

    Attributes:
        start_s: Start time in seconds.
        end_s: End time in seconds.
        kind: Which of :class:`SegmentKind` this is.
        cents: Representative pitch in cents above Sa (folded for a note,
            target pitch for a transition, mean for a blip).
        start_cents: Pitch at the first frame, cents above Sa.
        end_cents: Pitch at the last frame, cents above Sa.
        spread_cents: Max deviation from the segment's own median, in cents.
            Measures how stable the region is.
        confidence: Mean frame confidence over the segment.
        frames: Indices of the frames this segment covers.
        emitted: Whether this segment produced a swar in the output.
    """

    start_s: float
    end_s: float
    kind: SegmentKind
    cents: float
    start_cents: float
    end_cents: float
    spread_cents: float
    confidence: float
    frames: np.ndarray
    emitted: bool = False

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return max(0.0, self.end_s - self.start_s)

    @property
    def n_frames(self) -> int:
        """Number of frames in the segment."""
        return int(self.frames.size)

    @property
    def glide_cents(self) -> float:
        """Signed size of the pitch change across the segment, in cents."""
        return self.end_cents - self.start_cents


@dataclass
class Segmentation:
    """The full result of segmenting a contour.

    Attributes:
        segments: Every segment, in time order, including transitions.
        notes: Just the stable notes, in time order.
        params: The parameters used.
        diagnostics: Free-form counters for the caller.
    """

    segments: list[Segment] = field(default_factory=list)
    notes: list[Segment] = field(default_factory=list)
    params: SegmentParams = field(default_factory=SegmentParams)
    diagnostics: dict = field(default_factory=dict)

    @property
    def emitted_notes(self) -> list[Segment]:
        """Notes that produced a swar in the output."""
        return [s for s in self.notes if s.emitted]

    def to_dict(self) -> list[dict]:
        """Return a JSON-serialisable view of every segment."""
        return [
            {
                "start_s": round(s.start_s, 3),
                "end_s": round(s.end_s, 3),
                "duration_s": round(s.duration_s, 3),
                "kind": s.kind.value,
                "cents": round(s.cents, 1),
                "start_cents": round(s.start_cents, 1),
                "end_cents": round(s.end_cents, 1),
                "spread_cents": round(s.spread_cents, 1),
                "confidence": round(s.confidence, 3),
                "emitted": s.emitted,
            }
            for s in self.segments
        ]


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find runs of ``True`` in a boolean array.

    Args:
        mask: Boolean array.

    Returns:
        List of ``(start, end)`` index pairs, end exclusive. Runs separated by
        even one ``False`` are separate, which is how a micro-gap in voicing
        splits two notes.
    """
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2].tolist(), changes[1::2].tolist()))


def _forward_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Carry the last valid value forward over gaps.

    Median filters need no ``nan`` in the window. Substituting the previous
    value keeps a gap flat instead of letting the filter interpolate a pitch
    that was never sung, and the caller's valid mask discards those frames
    afterwards.

    Args:
        values: Value track, possibly containing ``nan``.
        valid: Boolean mask of the valid samples.

    Returns:
        A filled track of the same length, with leading gaps filled by the
        first valid value.
    """
    v = np.asarray(values, dtype=np.float64)
    out = np.array(v, dtype=np.float64, copy=True)
    if not valid.any():
        return np.zeros_like(out)
    idx = np.arange(v.size)
    known = np.flatnonzero(valid)
    out[~valid] = np.interp(idx[~valid], known, v[known])
    return out


def smooth_cents(cents: np.ndarray, voiced: np.ndarray, width_frames: int) -> np.ndarray:
    """Median-filter a cents track, ignoring unvoiced frames.

    Filtering in the cents domain makes the operation pitch-scale invariant,
    so the same width smooths a low Sa and a high Sa' equally.

    Args:
        cents: Cents track, ``nan`` where unvoiced.
        voiced: Boolean voiced mask.
        width_frames: Median filter width; rounded up to odd, values below 3
            disable filtering.

    Returns:
        The smoothed track, with ``nan`` preserved at unvoiced frames.
    """
    c = np.asarray(cents, dtype=np.float64)
    out = np.array(c, dtype=np.float64, copy=True)
    idx = np.flatnonzero(np.asarray(voiced, bool))
    if idx.size == 0:
        return out
    width = max(3, int(width_frames))
    if width % 2 == 0:
        width += 1
    if idx.size < width:
        return out
    values = c[idx]
    padded = np.pad(values, width // 2, mode="edge")
    filtered = median_filter(padded, size=width, mode="nearest")[width // 2 : width // 2 + values.size]
    out[idx] = filtered
    return out


def _frame_duration(contour: CentsContour) -> float:
    """Duration of one frame in seconds, from the contour's own time grid."""
    if contour.n_frames < 2:
        return 0.023
    return float(np.median(np.diff(contour.times)))


def find_stable_regions(
    cents: np.ndarray,
    voiced: np.ndarray,
    frame_s: float,
    params: SegmentParams,
    smoothed: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Find stable note regions within the voiced contour.

    Regions are grown greedily: frames accumulate while they stay within
    ``stability_cents`` of the running median, and a frame that breaks that
    starts a new region. This handles the three things a contour does at once,
    which a threshold on frame-to-frame change cannot:

    * a **sharp step** between two notes, where the first frame of the new
      note is already outside the old note's median;
    * a **slow meend**, where no single frame moves more than a few cents and
      a step threshold never fires. A Sa-to-Pa glide spread over 30 frames
      steps by 24 cents at a time, so under a 60-cent threshold the whole
      phrase stayed one region, was unstable, and was discarded: the
      transcription came back with *no* notes at all;
    * **vibrato**, which stays close to the running median and so never
      splits a note.

    A candidate region is then rejected if it is an isolated outlier, i.e. one
    that is perfectly steady on its own but sits between two notes that agree
    with each other. That is the signature of a tracking error, such as a
    three-frame octave spike, and it is perfectly steady in isolation because
    the spike is its own median.

    Args:
        cents: Cents track used for the greedy accumulation. Pass the raw
            track: a median filter spreads a step over about half its window,
            which makes a real step look gradual.
        voiced: Boolean voiced mask.
        frame_s: Frame duration in seconds.
        params: Segmentation parameters.
        smoothed: Median-filtered cents used for medians and stability. Falls
            back to ``cents``.

    Returns:
        List of ``(start, end)`` index pairs, end exclusive, time ordered.
    """
    values = np.asarray(cents, dtype=np.float64)
    smooth = values if smoothed is None else np.asarray(smoothed, dtype=np.float64)
    valid = np.isfinite(values) & np.asarray(voiced, bool)
    if not valid.any():
        return []

    min_frames = max(2, int(round(params.min_note_s / max(frame_s, 1e-6))))
    # Regions are grown on the *smoothed* track, not the raw one. The pre-filter
    # suppresses vibrato (a 40-cent 5 Hz wobble came out at 10 cents), so the
    # tight stability tolerance can then be used without shattering a note, and
    # a meend still gets chopped into slices too short to be notes.
    smooth_valid = np.isfinite(smooth) & np.asarray(voiced, bool)
    raw_regions = _grow_regions(smooth, smooth_valid, params.stability_cents, min_frames)
    if not raw_regions:
        return []

    notes: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(raw_regions):
        region = smooth[start:end]
        region = region[np.isfinite(region)]
        if region.size < min_frames:
            continue  # too short to be a sung note: it is a transition
        med = float(np.median(region))
        settled = float(np.count_nonzero(np.abs(region - med) <= params.stability_cents))
        if settled < params.stability_fraction * region.size:
            continue  # not a settled pitch
        if _is_drift_slice(values[start:end], params):
            continue  # a slice of a meend, not a settled pitch
        outlier_limit = int(round(params.outlier_max_note_ratio * params.min_note_s / max(frame_s, 1e-6)))
        if (end - start) <= outlier_limit and _is_isolated_outlier(
            smooth, raw_regions, i, med, params, min_frames
        ):
            continue
        notes.append((start, end))
    return notes


def _grow_regions(
    values: np.ndarray,
    valid: np.ndarray,
    tolerance: float,
    min_frames: int,
) -> list[tuple[int, int]]:
    """Accumulate frames into maximal runs of near-constant pitch.

    A frame joins the current region while it stays within ``tolerance`` of
    that region's own running median. Using the region's median rather than a
    global window is what separates the three things a contour does:

    * a **sharp step** immediately breaks the tolerance, because the region's
      median still describes the old note;
    * a **slow meend** stays inside the tolerance, since the median follows the
      ramp, so the whole glide becomes one region, which is then long and
      unsettled and so is reported as a transition rather than as notes;
    * **vibrato** stays inside the tolerance, so the note is not shattered,
      and the region's median is the centre pitch, which is what collapses
      andolan to the note.

    The median is recomputed only when a frame is about to be rejected, which
    keeps this linear in practice.

    Args:
        values: Cents track.
        valid: Boolean mask of usable frames.
        tolerance: Cents a frame may sit from the running median and still
            join the region.
        min_frames: Shortest acceptable region, carried for signature symmetry.

    Returns:
        List of ``(start, end)`` index pairs, end exclusive, time ordered.
    """
    _ = min_frames
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    regions: list[tuple[int, int]] = []
    start: int | None = None
    current: list[float] = []
    previous_index: int | None = None
    for i in indices:
        i = int(i)
        v = float(values[i])
        if start is None:
            start, current, previous_index = i, [v], i
            continue
        # A break in voicing ends the region. Two notes at the same pitch
        # separated by a rest are two notes, and without this they were joined
        # across the silence and read as one long note.
        if i != previous_index + 1:
            regions.append((start, previous_index + 1))
            start, current = i, [v]
            previous_index = i
            continue
        if abs(v - float(np.median(current))) <= tolerance:
            current.append(v)
            previous_index = i
            continue
        regions.append((start, i))
        start, current, previous_index = i, [v], i
    if start is not None:
        regions.append((start, int(indices[-1]) + 1))
    return regions


def _is_drift_slice(region: np.ndarray, params: SegmentParams) -> bool:
    """Whether a region is a slice of a glide rather than a settled pitch.

    A meend gets chopped into short regions by the growth tolerance, and each
    slice is locally flat: four frames of a ramp rising 23 cents per frame span
    69 cents, and every one of those frames is within 40 cents of the slice's
    own median, so no test on magnitude can tell it from a short note. Shape
    does distinguish them, and the measure that works is path length against
    pitch range.

    A slice of a ramp travels in one direction, so the total distance the
    values cover is about the range they span: the ratio is near 1. A sung
    note with vibrato covers the same range several times over in each
    direction, so its path is several times its range. The ratio is scale
    free, so one threshold works for a slice of any ramp speed.

    Comparing the first and last frame instead of the whole range does not
    work: a note that happens to span an exact number of vibrato cycles starts
    and ends at the same pitch, which looks perfectly monotone. A real Ga note
    was rejected by that version on a 0.0002-cent margin.

    This must be judged on the *raw* contour. The median pre-filter flattens a
    slow ramp almost completely, so a nine-frame slice of one looks settled to
    the filtered track and was accepted as a note.

    Args:
        region: The region's cents values.
        params: Segmentation parameters.

    Returns:
        ``True`` when the region looks like part of a monotone drift.
    """
    values = region[np.isfinite(region)]
    if values.size < 4:
        return False
    # Ignore the attack and the decay. A region usually begins a frame or two
    # before the note settles and ends a frame or two after, and that ramp is a
    # real transition rather than evidence about the note itself: a Pa whose
    # region started one frame early looked like a 170-cent drift and was
    # discarded.
    trim = min(max(1, values.size // 4), max(0, (values.size - 4) // 2))
    if values.size - 2 * trim >= 4:
        values = values[trim : values.size - trim]
    pitch_range = float(values.max() - values.min())
    if pitch_range <= params.stability_cents:
        return False  # settled: no excursion to explain
    path = float(np.abs(np.diff(values)).sum())
    return path <= 1.5 * pitch_range


def _is_isolated_outlier(
    values: np.ndarray,
    regions: list[tuple[int, int]],
    index: int,
    median: float,
    params: SegmentParams,
    min_frames: int,
) -> bool:
    """Whether a region is surrounded by one other pitch and differs from it.

    A tracking error is a short pitch excursion *surrounded by the note it
    interrupts*, so its two neighbours agree with each other and disagree with
    it. A real note sits between two different pitches, so its neighbours
    disagree too. That is the only thing which separates them, and it has to
    be checked against the neighbours rather than the region itself, because a
    short excursion is perfectly steady in isolation: a three-frame octave
    spike is its own median.

    The neighbours are sampled from *inside* the adjacent regions, not from the
    frames immediately next to this one. Those adjacent frames are transition
    junk at a note boundary, and measuring there compares artefacts against
    notes and rejects perfectly good ones.

    The test only applies when both neighbours are long enough to be notes
    themselves. A one- or two-frame region between two notes is boundary junk;
    its median looked like agreement and cost a real Ma from every phrase.

    Args:
        values: The cents track.
        regions: All candidate regions, time ordered.
        index: Which of ``regions`` to test.
        median: The region's median pitch.
        params: Segmentation parameters.
        min_frames: Minimum region length that counts as a note.

    Returns:
        ``True`` when the neighbouring notes agree with each other but not
        with this region. A region at the start or end has no pair of
        neighbours and is never rejected.
    """
    if index == 0 or index == len(regions) - 1:
        return False
    left_len = regions[index - 1][1] - regions[index - 1][0]
    right_len = regions[index + 1][1] - regions[index + 1][0]
    if left_len < min_frames or right_len < min_frames:
        return False
    left = _interior_median(values, regions[index - 1])
    right = _interior_median(values, regions[index + 1])
    if not (np.isfinite(left) and np.isfinite(right)):
        return False
    if abs(left - right) > params.stability_cents:
        return False  # genuinely between two different pitches: a real note
    return abs(median - left) > params.stability_cents


def _interior_median(values: np.ndarray, region: tuple[int, int]) -> float:
    """Median of a region's middle frames, ignoring its edges.

    Args:
        values: The cents track.
        region: ``(start, end)`` frame range, end exclusive.

    Returns:
        The median of the central half of the region, or ``nan``.
    """
    start, end = region
    length = end - start
    if length <= 0:
        return float("nan")
    inset = max(0, length // 4)
    middle = values[start + inset : end - inset]
    middle = middle[np.isfinite(middle)]
    if middle.size == 0:
        middle = values[start:end]
        middle = middle[np.isfinite(middle)]
    return float(np.median(middle)) if middle.size else float("nan")


def split_unstable_regions(
    cents: np.ndarray,
    voiced: np.ndarray,
    stable: list[tuple[int, int]],
    frame_s: float,
    params: SegmentParams,
) -> list[tuple[int, int]]:
    """Find the non-stable voiced regions between stable ones.

    Args:
        cents: Smoothed cents track.
        voiced: Boolean voiced mask.
        stable: Stable regions from :func:`find_stable_regions`.
        frame_s: Frame duration in seconds.
        params: Segmentation parameters.

    Returns:
        List of ``(start, end)`` index pairs, time ordered, covering voiced
        frames outside the stable regions.
    """
    covered = np.zeros(voiced.shape[0], dtype=bool)
    for start, end in stable:
        covered[start:end] = True
    _ = cents, frame_s  # signature kept for symmetry; region shape needs only the mask
    return _contiguous_runs(voiced & ~covered)


def classify_transition(
    start_cents: float,
    end_cents: float,
    duration_s: float,
    params: SegmentParams,
) -> SegmentKind:
    """Decide whether a non-stable region is a glide or a blip.

    A glide is a real melodic move: it spans a recognisable interval. A blip is
    noise or a tracking artifact. The distinction is drawn on pitch distance
    rather than duration, because a slow meend and a fast one are both glides
    and a short scoop is still a glide.

    Args:
        start_cents: Pitch at the start of the region.
        end_cents: Pitch at the end of the region.
        duration_s: Length of the region.
        params: Segmentation parameters.

    Returns:
        :attr:`SegmentKind.TRANSITION` for a glide, :attr:`SegmentKind.BLIP`
        otherwise.
    """
    _ = duration_s
    if abs(end_cents - start_cents) >= params.merge_cents:
        return SegmentKind.TRANSITION
    return SegmentKind.BLIP


def make_note(
    contour: CentsContour,
    start: int,
    end: int,
    params: SegmentParams,
) -> Segment:
    """Build a stable-note segment from a frame range.

    The representative pitch is the median of the segment's folded cents,
    which places vibrato and andolan at the centre pitch rather than anywhere
    in the range.

    Args:
        contour: Source cents contour.
        start: First frame index.
        end: Last frame index, exclusive.
        params: Segmentation parameters.

    Returns:
        The segment, ready to be mapped to a swar.
    """
    idx = np.arange(start, end)
    values = contour.cents[idx]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Segment(
            start_s=float(contour.times[start]), end_s=float(contour.times[min(end, contour.n_frames - 1)]),
            kind=SegmentKind.BLIP, cents=float("nan"), start_cents=float("nan"), end_cents=float("nan"),
            spread_cents=0.0, confidence=0.0, frames=idx,
        )
    med = float(np.median(values))
    spread = float(np.max(np.abs(values - med)))
    frame_s = _frame_duration(contour)
    return Segment(
        start_s=float(contour.times[start]),
        end_s=float(contour.times[start] + values.size * frame_s),
        kind=SegmentKind.NOTE,
        cents=med,
        start_cents=float(values[0]),
        end_cents=float(values[-1]),
        spread_cents=spread,
        confidence=float(np.mean(contour.confidence[idx])) if idx.size else 0.0,
        frames=idx,
    )


def make_transition(
    contour: CentsContour,
    start: int,
    end: int,
    params: SegmentParams,
) -> Segment:
    """Build a transition or blip segment from a frame range.

    The representative pitch is the pitch at the *end* of the region, because a
    glide is identified by where it arrives: a meend into Sa is reported as Sa.

    Args:
        contour: Source cents contour.
        start: First frame index.
        end: Last frame index, exclusive.
        params: Segmentation parameters.

    Returns:
        The segment.
    """
    idx = np.arange(start, end)
    values = contour.cents[idx]
    values = values[np.isfinite(values)]
    frame_s = _frame_duration(contour)
    if values.size == 0:
        return Segment(
            start_s=float(contour.times[start]), end_s=float(contour.times[start]),
            kind=SegmentKind.BLIP, cents=float("nan"), start_cents=float("nan"), end_cents=float("nan"),
            spread_cents=0.0, confidence=0.0, frames=idx,
        )
    start_c = float(values[0])
    end_c = float(values[-1])
    kind = classify_transition(start_c, end_c, values.size * frame_s, params)
    return Segment(
        start_s=float(contour.times[start]),
        end_s=float(contour.times[start] + values.size * frame_s),
        kind=kind,
        cents=end_c,
        start_cents=start_c,
        end_cents=end_c,
        spread_cents=float(np.max(np.abs(values - np.median(values)))),
        confidence=float(np.mean(contour.confidence[idx])),
        frames=idx,
    )


def merge_blips(notes: list[Segment], blips: list[Segment], params: SegmentParams) -> list[Segment]:
    """Absorb blips into the notes they interrupt, and drop stray notes.

    A blip is a one- or two-frame pitch excursion, almost always a tracking
    artefact rather than a sung note. A blip that sits between two notes at the
    same pitch has split one sustained note in two, so those neighbours are
    joined back together and the blip's frames are swallowed. A blip that
    touches only one note is simply ignored, since blips are never emitted as
    notation in the first place.

    Notes shorter than ``params.blip_s`` are dropped: a note that short is an
    artefact too, and keeping it adds a phantom swar to the output.

    Args:
        notes: Stable notes, time ordered.
        blips: Blip segments, time ordered.
        params: Segmentation parameters.

    Returns:
        The surviving notes, with split notes rejoined.
    """
    survivors = [n for n in notes if n.duration_s >= params.blip_s]
    if not blips or len(survivors) < 2:
        return survivors

    out = list(survivors)
    for blip in blips:
        for i in range(len(out) - 1):
            left, right = out[i], out[i + 1]
            # The blip must lie in the gap between the two notes, not inside
            # either of them.
            if blip.start_s < left.end_s or blip.end_s > right.start_s:
                continue
            if abs(left.cents - right.cents) > params.merge_cents:
                continue
            if abs(blip.start_cents - left.cents) > params.merge_cents:
                continue
            if abs(blip.end_cents - right.cents) > params.merge_cents:
                continue
            out[i] = _join_notes(left, right, blip)
            out.pop(i + 1)
            break
    return out


def _join_notes(left: Segment, right: Segment, blip: Segment | None = None) -> Segment:
    """Combine two adjacent notes at the same pitch into one.

    Args:
        left: The earlier note.
        right: The later note.
        blip: The blip between them, if any; its frames are included.

    Returns:
        A single note spanning both.
    """
    frames = np.concatenate([left.frames, right.frames])
    if blip is not None and blip.frames.size:
        frames = np.concatenate([left.frames, blip.frames, right.frames])
    values = np.concatenate(
        [np.full(left.frames.size, left.cents), np.full(right.frames.size, right.cents)]
    )
    return Segment(
        start_s=left.start_s,
        end_s=right.end_s,
        kind=SegmentKind.NOTE,
        cents=float(np.median(values)),
        start_cents=left.start_cents,
        end_cents=right.end_cents,
        spread_cents=max(left.spread_cents, right.spread_cents),
        confidence=min(left.confidence, right.confidence),
        frames=frames,
        emitted=left.emitted or right.emitted,
    )


def dedupe_repeated_notes(
    notes: list[Segment], max_gap_s: float, merge_cents: float = 45.0
) -> list[Segment]:
    """Merge consecutive notes at the same pitch separated by a short gap.

    Sung phrases repeat a pitch across breaths and consonants. Without this, a
    single sustained Sa sung as two attacks becomes "Sa Sa", which is wrong.

    Args:
        notes: Notes, time ordered.
        max_gap_s: Largest gap that still counts as the same note.
        merge_cents: Largest pitch difference that still counts as the same
            note. Compared as a plain difference of cents values: the
            ``cents_between`` helper takes frequencies, and feeding it two cents
            values gives log2(0/0), so every same-pitch merge silently failed.

    Returns:
        The merged notes.
    """
    if not notes:
        return []
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start_s - prev.end_s
        if gap <= max_gap_s and abs(prev.cents - note.cents) <= merge_cents:
            frames = np.concatenate([prev.frames, note.frames])
            merged[-1] = Segment(
                start_s=prev.start_s,
                end_s=note.end_s,
                kind=SegmentKind.NOTE,
                cents=prev.cents,
                start_cents=prev.start_cents,
                end_cents=note.end_cents,
                spread_cents=max(prev.spread_cents, note.spread_cents),
                confidence=min(prev.confidence, note.confidence),
                frames=frames,
                emitted=prev.emitted or note.emitted,
            )
        else:
            merged.append(note)
    return merged


def segment_contour(
    contour: CentsContour,
    params: SegmentParams | None = None,
) -> Segmentation:
    """Split a cents contour into stable notes and transitions.

    The pipeline is: median-filter, find stable regions, treat everything else
    as a transition candidate, classify glides against blips, absorb blips,
    then re-merge same-pitch neighbours. Keeping the contour continuous until
    this point is what lets a vibrato-heavy note survive as one note instead of
    fragmenting into several.

    Args:
        contour: Cents contour relative to Sa.
        params: Segmentation parameters.

    Returns:
        The segmentation, with notes and transitions in time order.
    """
    p = params or SegmentParams()
    if contour.n_frames == 0 or not contour.voiced.any():
        return Segmentation(segments=[], notes=[], params=p, diagnostics={"reason": "no voiced frames"})

    frame_s = _frame_duration(contour)
    smoothed = smooth_cents(contour.cents, contour.voiced, p.smooth_frames)
    work = CentsContour(
        times=contour.times,
        cents=smoothed,
        voiced=contour.voiced,
        octave=contour.octave,
        sa_hz=contour.sa_hz,
        source=contour.source,
    )

    # Boundaries are found on the raw contour and pitch on the smoothed one;
    # see find_stable_regions for why the two must not be the same track.
    stable = find_stable_regions(contour.cents, contour.voiced, frame_s, p, smoothed=smoothed)
    notes = [make_note(work, s, e, p) for s, e in stable]
    notes = [n for n in notes if n.kind == SegmentKind.NOTE and np.isfinite(n.cents)]
    if not notes:
        return Segmentation(segments=[], notes=[], params=p, diagnostics={"reason": "no stable regions"})

    # Regions from _grow_regions are disjoint and time ordered by
    # construction, so every frame is claimed once and no trimming is needed.
    stable = [(int(n.frames[0]), int(n.frames[-1]) + 1) for n in notes]
    unstable = split_unstable_regions(smoothed, contour.voiced, stable, frame_s, p)
    transitions = [make_transition(work, s, e, p) for s, e in unstable]
    blips = [t for t in transitions if t.kind == SegmentKind.BLIP]
    glides = [t for t in transitions if t.kind == SegmentKind.TRANSITION]

    notes = merge_blips(notes, blips, p)
    notes = dedupe_repeated_notes(notes, p.same_pitch_gap_s, p.merge_cents)

    # Interleave by start time so the caller sees the real chronology.
    everything: list[Segment] = sorted(notes + glides, key=lambda s: s.start_s)
    return Segmentation(
        segments=everything,
        notes=notes,
        params=p,
        diagnostics={
            "stable_regions": len(stable),
            "transitions": len(glides),
            "blips_absorbed": len(blips),
            "voiced_frames": int(contour.voiced.sum()),
            "frame_s": round(frame_s, 5),
        },
    )
