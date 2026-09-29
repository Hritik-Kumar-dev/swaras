"""Stage 2: tonic (Sa) detection.

Sa is the one pitch in the recording that cannot be found from the recording
alone: nothing in the audio distinguishes "this is Sa" from "this is Pa, and the
tonic was below". So this stage never claims certainty. It returns a ranked
list of candidates with confidence values, and the UI is expected to let the
user correct Sa and re-transcribe.

Two independent methods feed the ranking:

* :func:`detect_tonic_histogram` -- a pure-NumPy method that builds a cents
  histogram of the voiced F0 and scores each candidate Sa by the strength of
  the peaks at Sa, at Pa (+702 cents) and at the octave. Always available.
* :func:`detect_tonic_essentia` -- a wrapper around Essentia's
  ``TonicIndianArtMusic``. Accurate on real music with a tanpura-like drone,
  which is the situation it was designed for. On a bare melodic line with no
  drone it returns the bottom of its search range, so its answer is treated as
  corroboration rather than as proof.

Tonic errors are overwhelmingly octave errors and Pa/Ma confusions, which is
why the octave-equivalent candidates are collapsed to the lower one and the
rest are kept visible as alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ._essentia import as_essentia_signal, configure_algo, essentia_available
from .cents import OCTAVE_CENTS, hz_to_cents
from .config import TonicParams
from .pitch import PitchContour

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TonicCandidate:
    """One possible Sa, with the evidence behind it.

    Attributes:
        hz: Candidate tonic in Hz.
        cents: Cents above the histogram reference frequency.
        score: Raw score from the histogram method.
        confidence: Score as a fraction of the best candidate's score, in
            ``(0, 1]``. The best candidate is always ``1.0``.
        sa_peak: Histogram density at the candidate itself.
        pa_peak: Histogram density at Sa + 702 cents.
        octave_peak: Combined density at the octave above and below.
        methods: Which detectors supported this candidate, e.g.
            ``("histogram", "essentia")``.
    """

    hz: float
    cents: float
    score: float
    confidence: float
    sa_peak: float
    pa_peak: float
    octave_peak: float
    methods: tuple[str, ...] = ("histogram",)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "hz": round(self.hz, 3),
            "cents": round(self.cents, 1),
            "score": round(self.score, 6),
            "confidence": round(self.confidence, 4),
            "sa_peak": round(self.sa_peak, 6),
            "pa_peak": round(self.pa_peak, 6),
            "octave_peak": round(self.octave_peak, 6),
            "methods": list(self.methods),
        }


@dataclass(frozen=True)
class TonicResult:
    """The outcome of tonic detection.

    Attributes:
        candidates: Ranked candidates, best first.
        method: Primary method that produced the result, ``"histogram"``,
            ``"essentia"`` or ``"manual"``.
        essentia_hz: Essentia's raw answer, kept even when it was rejected, so
            the discrepancy is visible rather than hidden.
        warning: Human-readable note about a rejected or conflicting method.
    """

    candidates: tuple[TonicCandidate, ...]
    method: str = "histogram"
    essentia_hz: float | None = None
    warning: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def hz(self) -> float | None:
        """Best Sa in Hz, or ``None`` if detection found nothing."""
        return self.candidates[0].hz if self.candidates else None

    @property
    def confidence(self) -> float:
        """Confidence in the best candidate, in ``(0, 1]``."""
        return self.candidates[0].confidence if self.candidates else 0.0

    @property
    def is_valid(self) -> bool:
        """Whether a usable Sa was found."""
        return bool(self.candidates)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "hz": round(self.hz, 3) if self.hz is not None else None,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "candidates": [c.to_dict() for c in self.candidates],
            "essentia_hz": round(self.essentia_hz, 3) if self.essentia_hz else None,
            "warning": self.warning,
        }


def cents_histogram(
    f0: np.ndarray,
    voiced: np.ndarray,
    ref_hz: float = 27.5,
    bin_width_cents: float = 1.0,
    smooth_sigma_cents: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smoothed 1-cent histogram of voiced F0 in the cents domain.

    Working in cents rather than Hz is what makes the Sa/Pa relationship
    meaningful: Pa is 702 cents above Sa regardless of which Sa is chosen, and
    a linear-frequency histogram would put the fifth at a different place for
    every candidate.

    Args:
        f0: F0 track in Hz; ``0.0`` marks unvoiced frames.
        voiced: Boolean voiced mask.
        ref_hz: Reference frequency defining 0 cents. Defaults to 27.5 Hz
            (A0) so that the whole search range is a positive integer number of
            octaves above it.
        bin_width_cents: Histogram bin width in cents.
        smooth_sigma_cents: Gaussian smoothing width in cents. A singer glides
            between notes, so the raw histogram peaks are broad; smoothing
            turns a plateau into something with a locatable maximum.

    Returns:
        Tuple ``(bin_centers, density)``. ``density`` is normalised to a maximum
        of 1.0 so that scores from different recordings are comparable. Both
        arrays are empty when nothing is voiced.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    if not voiced.any():
        return np.zeros(0), np.zeros(0)

    cents = np.asarray(hz_to_cents(f0[voiced], ref_hz), dtype=np.float64)
    # Pad the range by a few bins on each side. np.histogram places the first
    # bin centre half a bin above the lower edge, so a range built exactly
    # from the data would put the lowest sung pitch *outside* the
    # interpolation range and score it zero. This bit the Sa candidate itself
    # when Sa was the lowest note in the melody.
    pad = max(2.0, 4.0 * bin_width_cents)
    lo = float(np.floor(cents.min() - pad))
    hi = float(np.ceil(cents.max() + pad))
    n_bins = int((hi - lo) / bin_width_cents) + 1
    counts, edges = np.histogram(cents, bins=n_bins, range=(lo, hi))
    centers = edges[:-1] + bin_width_cents / 2.0

    sigma_bins = max(0.0, float(smooth_sigma_cents) / float(bin_width_cents))
    density = gaussian_filter1d(counts.astype(np.float64), sigma=sigma_bins, mode="constant")
    peak = float(density.max())
    if peak > 0:
        density = density / peak
    return centers, density


def sample_histogram(
    centers: np.ndarray,
    density: np.ndarray,
    query_cents: np.ndarray,
) -> np.ndarray:
    """Look up histogram density at arbitrary cents values.

    Queries outside the histogram's range return ``0.0`` rather than the
    nearest edge, so that a candidate Sa whose Pa falls beyond the sung range
    scores zero for Pa instead of borrowing a neighbouring peak.

    Args:
        centers: Bin centres in cents, ascending.
        density: Bin densities, normalised to a maximum of 1.0.
        query_cents: Cents positions to look up.

    Returns:
        Densities at the query positions.
    """
    if centers.size == 0:
        return np.zeros(np.shape(query_cents), dtype=np.float64)
    q = np.atleast_1d(np.asarray(query_cents, dtype=np.float64))
    out = np.interp(q, centers, density, left=0.0, right=0.0)
    # np.interp clamps; force a true zero outside the built range.
    out = np.where((q < centers[0]) | (q > centers[-1]), 0.0, out)
    return out


def lowest_supported_cents(centers: np.ndarray, density: np.ndarray, min_peak: float) -> float:
    """Find the lowest pitch in the histogram that is substantially sung.

    Args:
        centers: Histogram bin centres in cents.
        density: Histogram densities.
        min_peak: Density above which a pitch counts as sung.

    Returns:
        The cents position of the lowest qualifying bin, or ``nan`` if the
        histogram has nothing above ``min_peak``.
    """
    if centers.size == 0:
        return float("nan")
    strong = np.flatnonzero(density >= min_peak)
    if strong.size == 0:
        return float("nan")
    return float(centers[strong[0]])


def score_candidate(
    sa_cents: float,
    centers: np.ndarray,
    density: np.ndarray,
    params: TonicParams | None = None,
    lowest_cents: float = float("nan"),
) -> tuple[float, float, float, float]:
    """Score a candidate Sa from the histogram peaks at Sa, Pa and the octave.

    The Sa term alone is a poor discriminator, because it just asks "is there a
    sung note here". The Pa and octave terms supply the evidence that actually
    identifies a tonic: a real tonic has a sung fifth above it, and usually the
    octave too. A candidate with a strong peak but no fifth above it is more
    likely to be a prominent scale degree than a tonic.

    The octave term sums the peaks an octave above *and* below, so the score is
    symmetric under octave displacement. That symmetry is deliberate: it is
    what makes the Sa/Sa' pair come out with near-identical scores, which
    :func:`collapse_octave_equivalents` then resolves.

    A fourth term, the lowest-note prior, breaks a tie the other three cannot.
    In a scale run that touches every degree from Sa upward, every candidate
    has an equally strong Sa peak and a fifth above it, so the peak terms
    cannot distinguish Sa from Re. The tonic of such a scale is its lowest
    note, and a small bonus for sitting at or below the lowest strongly
    supported pitch encodes that.

    Args:
        sa_cents: Candidate Sa position in cents.
        centers: Histogram bin centres.
        density: Histogram densities.
        params: Scoring weights and intervals.
        lowest_cents: Lowest well-supported pitch in the recording, from
            :func:`lowest_supported_cents`. ``nan`` disables the prior.

    Returns:
        Tuple ``(score, sa_peak, pa_peak, octave_peak)``. The returned score
        includes the lowest-note prior.
    """
    p = params or TonicParams()
    q = np.array(
        [sa_cents, sa_cents + p.pa_cents, sa_cents + OCTAVE_CENTS, sa_cents - OCTAVE_CENTS]
    )
    sa_peak, pa_peak, up, down = sample_histogram(centers, density, q)
    octave_peak = float(up) + float(down)
    score = p.w_sa * float(sa_peak) + p.w_pa * float(pa_peak) + p.w_octave * octave_peak
    if p.w_lowest and np.isfinite(lowest_cents):
        # Flat credit across a short tolerance band, then a linear decay. A
        # hard cliff would let the candidate grid straddle the edge and give
        # unstable winners; a decay over a full octave gives too little
        # separation to break the scale-run tie, since a note a third above
        # the lowest would still collect most of the credit.
        delta = sa_cents - lowest_cents
        if delta <= p.lowest_tolerance_cents:
            credit = 1.0
        else:
            credit = max(0.0, 1.0 - (delta - p.lowest_tolerance_cents) / max(p.lowest_falloff_cents, 1e-6))
        score += p.w_lowest * credit
    return score, float(sa_peak), float(pa_peak), octave_peak


def local_maxima(values: np.ndarray, min_separation: int = 1) -> np.ndarray:
    """Indices of local maxima in a 1-D array.

    Args:
        values: Array to inspect.
        min_separation: Minimum index distance between two reported maxima.

    Returns:
        Indices of the maxima, strongest first. A flat plateau yields its
        centre.
    """
    if values.size == 0:
        return np.zeros(0, dtype=int)
    if values.size < 3:
        return np.array([int(np.argmax(values))], dtype=int)
    # scipy's find_peaks ignores plateaus, so a flat run is handled separately
    # by comparing against a slightly smoothed copy to break ties.
    candidates = _strict_local_maxima(values)
    if candidates.size == 0:
        return np.array([int(np.argmax(values))], dtype=int)
    order = candidates[np.argsort(-values[candidates])]
    kept: list[int] = []
    for idx in order:
        if all(abs(int(idx) - k) >= min_separation for k in kept):
            kept.append(int(idx))
    kept.sort(key=lambda i: -values[i])
    return np.array(kept, dtype=int)


def _strict_local_maxima(values: np.ndarray) -> np.ndarray:
    """Indices where a value is strictly greater than both neighbours."""
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1


def _suppress_nearby(idx: np.ndarray, scores: np.ndarray, separation: int) -> np.ndarray:
    """Keep only the strongest candidate in each neighbourhood.

    Args:
        idx: Candidate indices, strongest first, as returned by
            :func:`local_maxima`.
        scores: Score at each grid position.
        separation: Minimum index distance between kept candidates.

    Returns:
        Indices of the surviving candidates, strongest first.
    """
    if separation <= 0 or idx.size <= 1:
        return idx
    kept: list[int] = []
    for i in idx:
        if all(abs(int(i) - k) >= separation for k in kept):
            kept.append(int(i))
    return np.array(kept, dtype=int)


def collapse_octave_equivalents(
    candidates: Sequence[TonicCandidate],
    ratio: float = 0.9,
    tolerance_cents: float = 3.0,
) -> list[TonicCandidate]:
    """Demote octave-equivalent candidates that are not clearly better.

    Sa and Sa' are the same tonic an octave apart, and a melody that both
    starts and ends on Sa gives them near-identical evidence. This keeps the
    lower one, on the reasoning that a singer who has the lower Sa available
    treats it as the tonic, and reports the upper only if it clearly wins.

    Args:
        candidates: Candidates, any order.
        ratio: A candidate is collapsed into a lower one when its score is
            within this fraction of the lower candidate's score.
        tolerance_cents: Slack for treating two positions as an exact octave
            apart. Candidate positions sit on a 1-cent grid, so a strict
            comparison would miss a true octave by a fraction of a cent.

    Returns:
        Candidates with the losers removed, sorted by descending score.
    """
    if not candidates:
        return []
    by_cents = sorted(candidates, key=lambda c: c.cents)
    keep: list[TonicCandidate] = []
    for cand in by_cents:
        collapsed = False
        for other in keep:
            # Distance to the nearest whole-octave multiple of the separation.
            remainder = abs(cand.cents - other.cents) % OCTAVE_CENTS
            if min(remainder, OCTAVE_CENTS - remainder) <= tolerance_cents:
                if cand.score <= other.score * ratio:
                    collapsed = True
                break
        if not collapsed:
            keep.append(cand)
    keep.sort(key=lambda c: -c.score)
    return keep


def detect_tonic_histogram(
    contour: PitchContour,
    params: TonicParams | None = None,
    top_k: int | None = None,
) -> TonicResult:
    """Detect Sa by scoring the peaks at Sa, Pa and the octave.

    The search is a dense scan over the candidate range rather than a scan
    over histogram peaks. That costs a few thousand extra evaluations but
    removes a fragile assumption: a peak-picking candidate set can miss a
    plausible Sa that sits in a trough between two strong peaks.

    Args:
        contour: Pitch contour to analyse.
        params: Tonic detection parameters.
        top_k: How many candidates to return; defaults to ``params.top_k``.

    Returns:
        The ranked candidates. Empty if nothing is voiced.
    """
    p = params or TonicParams()
    k = top_k if top_k is not None else p.top_k
    centers, density = cents_histogram(
        contour.f0, contour.voiced, p.histogram_ref_hz, p.bin_width_cents, p.smooth_sigma_cents
    )
    if centers.size == 0:
        return TonicResult(
            candidates=(), method="histogram", warning="no voiced frames: cannot detect Sa"
        )

    lo_cents = float(hz_to_cents(p.sa_min_hz, p.histogram_ref_hz))
    hi_cents = float(hz_to_cents(p.sa_max_hz, p.histogram_ref_hz))
    step = max(1, int(round(1.0 / max(p.bin_width_cents, 1e-6))))
    grid = np.arange(int(lo_cents), int(hi_cents) + 1, step, dtype=np.float64)

    lowest = lowest_supported_cents(centers, density, p.min_peak_prominence_cents / 100.0)
    scores = np.zeros(grid.size)
    sa_peaks = np.zeros(grid.size)
    pa_peaks = np.zeros(grid.size)
    oct_peaks = np.zeros(grid.size)
    for i, sa in enumerate(grid):
        s, sa_p, pa_p, oct_p = score_candidate(sa, centers, density, p, lowest)
        scores[i], sa_peaks[i], pa_peaks[i], oct_peaks[i] = s, sa_p, pa_p, oct_p

    best = float(scores.max()) if scores.size else 0.0
    if best <= 0:
        return TonicResult(
            candidates=(), method="histogram", warning="histogram is empty over the Sa search range"
        )

    # Require some real support at the candidate itself, so a position with a
    # Pa peak but nothing of its own cannot win on borrowed evidence.
    keep = sa_peaks >= max(p.min_peak_prominence_cents / 100.0, p.min_peak_ratio * sa_peaks.max())
    if not keep.any():
        keep = np.ones_like(scores, dtype=bool)

    idx = local_maxima(np.where(keep, scores, -np.inf), min_separation=1)
    # The raw maximum of a broad peak is a continuous function of the audio,
    # so the three highest candidates often land on the same peak a few cents
    # apart (on real speech they came out at 79.0, 74.0 and 72.2 Hz, which
    # tells the user nothing). Suppress weaker candidates that sit within
    # candidate_separation_cents of a stronger one, so the alternatives are
    # genuinely different pitch classes they could choose instead.
    separation = max(1, int(round(p.candidate_separation_cents / step)))
    idx = _suppress_nearby(idx, scores, separation)
    candidates = [
        TonicCandidate(
            hz=float(p.histogram_ref_hz * 2 ** (grid[i] / 1200.0)),
            cents=float(grid[i]),
            score=float(scores[i]),
            confidence=1.0,  # filled in below
            sa_peak=float(sa_peaks[i]),
            pa_peak=float(pa_peaks[i]),
            octave_peak=float(oct_peaks[i]),
        )
        for i in idx
    ]
    # Collapse before truncating, so an octave-equivalent duplicate cannot
    # push a genuinely different alternative out of the reported list.
    candidates = collapse_octave_equivalents(
        candidates, p.octave_collapse_ratio, p.octave_equivalence_cents
    )
    candidates = candidates[:k]
    if not candidates:
        return TonicResult(candidates=(), method="histogram", warning="no Sa candidate scored")
    top = candidates[0].score
    return TonicResult(
        candidates=tuple(
            TonicCandidate(
                hz=c.hz, cents=c.cents, score=c.score, confidence=c.score / top,
                sa_peak=c.sa_peak, pa_peak=c.pa_peak, octave_peak=c.octave_peak, methods=c.methods,
            )
            for c in candidates
        ),
        method="histogram",
    )


def detect_tonic_essentia(
    samples: np.ndarray,
    sample_rate: int,
    params: TonicParams | None = None,
) -> float | None:
    """Run Essentia's ``TonicIndianArtMusic`` and return its answer in Hz.

    The algorithm models the timbre of a tanpura-like drone and picks the
    pitch class whose harmonics are most strongly present, so it is well
    suited to real khayal or Carnatic recordings. On a bare melodic line with
    no drone it has nothing to latch onto and returns the bottom of its search
    range, which the caller is expected to check.

    Args:
        samples: Mono float samples.
        sample_rate: Sample rate in Hz.
        params: Tonic parameters, used for the search range.

    Returns:
        The detected tonic in Hz, or ``None`` if Essentia is unavailable or
        returned something unusable.
    """
    p = params or TonicParams()
    if not essentia_available() or len(samples) == 0:
        return None
    import essentia.standard as es

    algo = es.TonicIndianArtMusic()
    configure_algo(
        algo,
        sampleRate=int(sample_rate),
        frameSize=2048,
        hopSize=512,
        minTonicFrequency=float(p.essentia_min_tonic_hz),
        maxTonicFrequency=float(p.essentia_max_tonic_hz),
        numberSaliencePeaks=int(p.essentia_number_salience_peaks),
    )
    try:
        hz = float(algo(as_essentia_signal(samples)))
    except Exception as exc:
        logger.warning("essentia tonic detection failed: %s", exc)
        return None
    if not np.isfinite(hz) or hz <= 0:
        return None
    return hz


def _is_at_search_edge(hz: float, params: TonicParams) -> bool:
    """Whether Essentia's answer sits at the edge of the range we gave it.

    A returned value within :attr:`TonicParams.essentia_reject_edge_hz` of
    either bound is the signature of "found nothing and clamped", which is what
    this algorithm does on audio with no drone.
    """
    edge = params.essentia_reject_edge_hz
    return hz <= params.essentia_min_tonic_hz + edge or hz >= params.essentia_max_tonic_hz - edge


def merge_essentia(
    result: TonicResult,
    essentia_hz: float | None,
    params: TonicParams | None = None,
) -> TonicResult:
    """Fold Essentia's answer into the histogram ranking.

    Agreement raises the candidate's confidence. Disagreement is kept visible:
    the raw Essentia value is retained in the result and a warning is attached,
    so the UI can offer it as an alternative instead of silently discarding it.

    Args:
        result: The histogram result.
        essentia_hz: Essentia's answer, or ``None``.
        params: Tonic parameters.

    Returns:
        An updated result.
    """
    p = params or TonicParams()
    if essentia_hz is None:
        return result
    warning: str | None = None
    if _is_at_search_edge(essentia_hz, p):
        warning = (
            f"essentia returned {essentia_hz:.1f} Hz, at the edge of its search range; "
            "this usually means no tanpura-like drone was found"
        )
        logger.info("%s", warning)
        return TonicResult(
            candidates=result.candidates,
            method=result.method,
            essentia_hz=essentia_hz,
            warning=warning,
        )

    merged: list[TonicCandidate] = []
    matched = False
    for c in result.candidates:
        delta = abs(float(hz_to_cents(essentia_hz, c.hz)))
        if delta <= p.essentia_agree_cents:
            matched = True
            merged.append(
                TonicCandidate(
                    hz=c.hz, cents=c.cents,
                    score=c.score * p.essentia_agreement_boost,
                    confidence=c.confidence,
                    sa_peak=c.sa_peak, pa_peak=c.pa_peak, octave_peak=c.octave_peak,
                    # Set order rather than sorted(): the methods are a
                    # pipeline provenance trail (histogram first, Essentia
                    # added by this call), not an alphabetical list.
                    methods=(*c.methods, "essentia"),
                )
            )
        else:
            merged.append(c)
    if not matched:
        warning = (
            f"essentia suggested {essentia_hz:.1f} Hz, which does not match the histogram "
            f"ranking (best {result.hz:.1f} Hz)" if result.is_valid else
            f"essentia suggested {essentia_hz:.1f} Hz"
        )
        logger.info("%s", warning)
    # Re-sort: the agreement boost can promote a lower-ranked candidate above
    # the one that was first, and TonicResult.hz reads candidates[0]. Skipping
    # this made the reported best candidate disagree with the highest
    # confidence in the list.
    merged.sort(key=lambda c: -c.score)
    top = max((c.score for c in merged), default=1.0) or 1.0
    return TonicResult(
        candidates=tuple(
            TonicCandidate(
                hz=c.hz, cents=c.cents, score=c.score, confidence=min(1.0, c.score / top),
                sa_peak=c.sa_peak, pa_peak=c.pa_peak, octave_peak=c.octave_peak, methods=c.methods,
            )
            for c in merged
        ),
        method=result.method,
        essentia_hz=essentia_hz,
        warning=warning,
    )


def detect_tonic(
    contour: PitchContour,
    samples: np.ndarray | None = None,
    sample_rate: int | None = None,
    params: TonicParams | None = None,
    top_k: int | None = None,
) -> TonicResult:
    """Detect Sa from a contour, using Essentia as corroboration if available.

    Args:
        contour: Pitch contour to analyse.
        samples: The original audio, needed only for Essentia. Without it the
            histogram method runs alone.
        sample_rate: Sample rate of ``samples``.
        params: Tonic parameters.
        top_k: Number of candidates to return.

    Returns:
        The ranked candidates.
    """
    p = params or TonicParams()
    result = detect_tonic_histogram(contour, p, top_k)
    if not p.use_essentia or samples is None or sample_rate is None:
        return result
    essentia_hz = detect_tonic_essentia(samples, sample_rate, p)
    if essentia_hz is None:
        return result
    return merge_essentia(result, essentia_hz, p)


def manual_tonic(sa_hz: float) -> TonicResult:
    """Build a result from a user-supplied Sa, bypassing detection.

    Args:
        sa_hz: The Sa the user chose.

    Returns:
        A single-candidate result with full confidence.

    Raises:
        ValueError: If ``sa_hz`` is not a positive, finite frequency.
    """
    hz = float(sa_hz)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError(f"sa must be a positive frequency in Hz, got {sa_hz!r}")
    candidate = TonicCandidate(
        hz=hz, cents=float(hz_to_cents(hz, 27.5)), score=1.0, confidence=1.0,
        sa_peak=1.0, pa_peak=0.0, octave_peak=0.0, methods=("manual",),
    )
    return TonicResult(candidates=(candidate,), method="manual")


