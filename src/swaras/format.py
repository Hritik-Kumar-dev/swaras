"""Stage 3c: rendering transcribed notes as text or JSON.

The plain-text output is the point of the whole pipeline, so it is a single
line of swaras with no timing:

    Ni Sa Ma Ma Pa Dha Ni

Octave is carried by a notation marker rather than a word: a dot below
(``Ni.``) for mandra and a tick above (``Sa'``) for taar. That keeps a phrase
readable at a glance, which is how the notation is read by a musician.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .shruti import shruti_statistics
from .swar import SwaraNote


@dataclass(frozen=True)
class Transcription:
    """A complete transcription result.

    Attributes:
        notes: The swara notes in time order.
        sa_hz: The Sa used, in Hz.
        sa_source: ``"detected"``, ``"manual"`` or a method name.
        sa_confidence: Confidence in the chosen Sa.
        tonic_candidates: The other Sa candidates, for the UI.
        tonic_warning: Any warning raised during Sa detection.
        detector: Which pitch tracker produced the contour.
        shruti_notes: The same notes on the 22-shruti grid, with deviations.
        diagnostics: Free-form counters from the pipeline.
    """

    notes: list[SwaraNote]
    sa_hz: float
    sa_source: str = "detected"
    sa_confidence: float = 1.0
    tonic_candidates: list[dict] = None  # type: ignore[assignment]
    tonic_warning: str | None = None
    detector: str = "melodia"
    shruti_notes: list = None  # type: ignore[assignment]
    diagnostics: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tonic_candidates is None:
            object.__setattr__(self, "tonic_candidates", [])
        if self.shruti_notes is None:
            object.__setattr__(self, "shruti_notes", [])
        if self.diagnostics is None:
            object.__setattr__(self, "diagnostics", {})

    @property
    def text(self) -> str:
        """The plain-text notation, one line of space-separated swaras."""
        return " ".join(n.label for n in self.notes)

    @property
    def text_long(self) -> str:
        """Readable form with full names, e.g. ``"Ni (taar) Sa"``."""
        return " ".join(n.label_long for n in self.notes)

    @property
    def shruti_text(self) -> str:
        """The shruti line: each note with its signed deviation, e.g. ``"S0 +0c"``."""
        if not self.shruti_notes:
            return ""
        return " ".join(n.label_with_deviation for n in self.shruti_notes)

    @property
    def counts(self) -> dict[str, int]:
        """How many times each swara appears, ignoring octave."""
        out: dict[str, int] = {}
        for n in self.notes:
            out[n.swara] = out.get(n.swara, 0) + 1
        return out

    def to_dict(self) -> dict:
        """Return the full JSON-serialisable result."""
        payload = {
            "notation": self.text,
            "notation_long": self.text_long,
            "sa_hz": round(self.sa_hz, 3),
            "sa_source": self.sa_source,
            "sa_confidence": round(self.sa_confidence, 4),
            "tonic_candidates": self.tonic_candidates,
            "tonic_warning": self.tonic_warning,
            "detector": self.detector,
            "note_count": len(self.notes),
            "swar_counts": self.counts,
            "notes": [n.to_dict() for n in self.notes],
            "diagnostics": self.diagnostics,
        }
        if self.shruti_notes:
            payload["shruti"] = {
                "line": self.shruti_text,
                "notes": [n.to_dict() for n in self.shruti_notes],
                "statistics": shruti_statistics(self.shruti_notes),
            }
        return payload

    def to_json(self, indent: int | None = 2) -> str:
        """Serialise to a JSON string.

        Args:
            indent: Passed to :func:`json.dumps`; ``None`` for compact output.

        Returns:
            The JSON text.
        """
        return json.dumps(self.to_dict(), indent=indent)


def format_text(notes: Sequence[SwaraNote], separator: str = " ") -> str:
    """Render notes as plain text notation.

    Args:
        notes: The notes to render.
        separator: String between swaras.

    Returns:
        The notation line, empty if there are no notes.
    """
    return separator.join(n.label for n in notes)


def format_detailed(notes: Sequence[SwaraNote]) -> str:
    """Render notes with timing and cents deviation, one note per line.

    Args:
        notes: The notes to render.

    Returns:
        A multi-line report.
    """
    if not notes:
        return "(no notes)"
    lines = [
        f"{'#':>3}  {'swar':<10} {'label':<5} {'start':>7} {'end':>7} "
        f"{'dur':>6} {'cents':>8} {'dev':>7} {'conf':>5}"
    ]
    for i, n in enumerate(notes, start=1):
        lines.append(
            f"{i:>3}  {n.swara:<10} {n.label:<5} {n.start_s:>7.3f} {n.end_s:>7.3f} "
            f"{n.duration_s:>6.3f} {n.absolute_cents:>8.1f} "
            f"{n.deviation_cents:>+7.1f} {n.confidence:>5.2f}"
        )
    return "\n".join(lines)


def format_shruti_table(notes: Sequence) -> str:
    """Render shruti matches as a table, one note per line.

    Args:
        notes: :class:`~swaras.shruti.ShrutiNote` matches, in time order.

    Returns:
        A multi-line report, showing the deviation, the clarity margin and the
        confidence of each match.
    """
    if not notes:
        return "(no notes)"
    lines = [
        f"{'#':>3}  {'shruti':<8} {'swar':<10} {'var':<8} {'dev':>7} "
        f"{'clarity':>8} {'conf':>5} {'start':>7}"
    ]
    for i, n in enumerate(notes, start=1):
        lines.append(
            f"{i:>3}  {n.label:<8} {n.swara:<10} {n.variant:<8} "
            f"{n.deviation_cents:>+7.1f} {n.clarity_cents:>8.1f} "
            f"{n.confidence:>5.2f} {n.start_s:>7.3f}"
        )
    stats = shruti_statistics(list(notes))
    lines.append("-" * 62)
    lines.append(
        f"mean |deviation| {stats['mean_abs_deviation_cents']}c   "
        f"worst {stats['max_abs_deviation_cents']}c   "
        f"unclear {stats['unclear_count']}/{stats['count']}"
    )
    return "\n".join(lines)


def format_summary(result: Transcription) -> str:
    """Render a human-readable summary of a whole transcription.

    Args:
        result: The transcription.

    Returns:
        A multi-line report.
    """
    sa_note = {
        "detected": "detected",
        "manual": "manual",
    }.get(result.sa_source, result.sa_source)
    lines = [
        f"Sa      : {result.sa_hz:.2f} Hz ({sa_note}, confidence {result.sa_confidence:.2f})",
        f"Notes   : {len(result.notes)}",
        f"Notation: {result.text or '(none)'}",
    ]
    if result.tonic_candidates:
        alts = ", ".join(
            f"{c.get('hz', 0):.1f} Hz (conf {c.get('confidence', 0):.2f})"
            for c in result.tonic_candidates[:3]
        )
        lines.append(f"Alts    : {alts}")
    if result.tonic_warning:
        lines.append(f"Warning : {result.tonic_warning}")
    if result.counts:
        top = ", ".join(f"{k} x{v}" for k, v in sorted(result.counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"Counts  : {top}")
    if result.shruti_notes:
        stats = shruti_statistics(result.shruti_notes)
        lines.append(
            f"Shruti  : mean |dev| {stats['mean_abs_deviation_cents']}c, "
            f"worst {stats['max_abs_deviation_cents']}c, "
            f"{stats['unclear_count']} unclear"
        )
    return "\n".join(lines)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars and arrays to plain Python types.

    Args:
        obj: Any JSON-unfriendly value, possibly nested.

    Returns:
        A structure :func:`json.dumps` accepts.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj
