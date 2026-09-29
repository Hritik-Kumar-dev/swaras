"""Pitch-contour plotting for Stage 1 inspection.

Matplotlib is an optional (dev) dependency: importing this module is only safe
when you actually want a plot. ``python -m swaras.plot`` uses it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio_io import load_file
from .cents import hz_to_cents
from .config import DEFAULT_CONFIG, PipelineConfig
from .pitch import PitchContour, get_detector
from .tonic import detect_tonic

logger = logging.getLogger(__name__)

#: Reference pitch used when no Sa is known yet, so the y-axis has a meaning.
DEFAULT_SA_HZ = 261.63


def _require_matplotlib():
    """Import matplotlib with a non-interactive backend.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "plotting needs matplotlib; install it with `pip install matplotlib`"
        ) from exc
    return plt


def plot_contour(
    contour: PitchContour,
    sa_hz: float | None = None,
    ax=None,
    title: str | None = None,
    color: str | None = None,
    show_confidence: bool = True,
):
    """Plot one F0 contour, with unvoiced frames shown as gaps.

    Args:
        contour: The contour to draw.
        sa_hz: Sa in Hz. When given, the y-axis is relabelled in cents above
            Sa and octave lines are drawn; otherwise it is plain Hz.
        ax: Existing axes to draw on; a new figure is created if ``None``.
        title: Plot title; defaults to the detector name.
        color: Line colour.
        show_confidence: Shade the plot by frame confidence.

    Returns:
        The matplotlib axes containing the plot.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))

    f0 = np.where(contour.voiced, contour.f0, np.nan)
    line, = ax.plot(
        contour.times, f0, lw=1.2, color=color, label=contour.method, zorder=2
    )
    ax.fill_between(
        contour.times, contour.confidence.min(), contour.confidence,
        color=line.get_color(), alpha=0.12, lw=0, zorder=1,
    )
    if contour.n_frames:
        ax.set_xlim(contour.times.min(), contour.times.max())
    ax.set_xlabel("time (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_title(title or f"F0 contour ({contour.method})")

    if sa_hz is not None and contour.n_frames:
        _overlay_sa(ax, contour, sa_hz)
    return ax


#: Vertical offsets (in data units) for fanning out overlaid lines, so two
#: trackers on the same grid stay individually readable.
_OFFSET_UNIT = 0.012


def _overlay_sa(ax, contour: PitchContour, sa_hz: float) -> None:
    """Relabel the y-axis in cents above Sa and draw octave / Sa grid lines.

    Args:
        ax: The axes to relabel.
        contour: The contour whose voiced range sets the axis limits.
        sa_hz: Sa in Hz.
    """
    voiced = contour.voiced
    if not voiced.any():
        return
    lo = float(hz_to_cents(np.nanmin(contour.f0[voiced]), sa_hz))
    hi = float(hz_to_cents(np.nanmax(contour.f0[voiced]), sa_hz))
    # Pad by a fraction of an octave, then choose a tick step that keeps the
    # axis to a readable number of labels instead of snapping to whole octaves
    # (which produced 25 labels for a one-octave melody).
    pad = 90.0
    lo, hi = lo - pad, hi + pad
    step = _choose_tick_step(hi - lo)
    lo_tick = int(np.floor(lo / step) * step)
    hi_tick = int(np.ceil(hi / step) * step)

    ticks: list[float] = []
    labels: list[str] = []
    for cents in range(lo_tick, hi_tick + 1, step):
        hz = sa_hz * 2 ** (cents / 1200.0)
        if cents % 1200 == 0:
            ax.axhline(hz, color="0.25", ls="--", lw=1.0, zorder=0)
        ticks.append(hz)
        labels.append(f"{cents:+d}c")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(
        sa_hz * 2 ** (lo_tick / 1200.0),
        sa_hz * 2 ** (hi_tick / 1200.0),
    )
    ax.set_ylabel("cents rel. Sa")


def _fan_out(ax, index: int, total: int) -> None:
    """Nudge one contour's line off its neighbours so both stay visible.

    Overlaid trackers agree closely, so their lines sit on top of each other
    and the upper one hides the lower. Shifting each line by a few percent of
    the y-range separates them without distorting the comparison (the vertical
    axis is in cents, so an offset of ~1% of the range is a few cents).

    Args:
        ax: Axes holding the lines.
        index: Zero-based index of the contour just drawn.
        total: Number of contours being drawn.
    """
    lines = [ln for ln in ax.get_lines() if ln.get_label() in ax.get_legend_handles_labels()[1]]
    if len(lines) < 2 or total < 2:
        return
    lo, hi = ax.get_ylim()
    span = float(hi - lo)
    if span <= 0:
        return
    lines[index].set_ydata(lines[index].get_ydata() + (index - (total - 1) / 2.0) * span * _OFFSET_UNIT)


def _choose_tick_step(span_cents: float) -> int:
    """Pick a cents-per-tick step giving roughly 6-16 labels.

    Args:
        span_cents: Width of the axis in cents.

    Returns:
        A step in ``{10, 20, 25, 50, 100, 200}``.
    """
    for step in (10, 20, 25, 50, 100, 200):
        if span_cents / step <= 16:
            return step
    return 200


def plot_comparison(
    contours: Sequence[PitchContour],
    sa_hz: float | None = None,
    out: str | Path | None = None,
):
    """Plot several contours on shared axes for direct comparison.

    Args:
        contours: Contours to overlay, one line each.
        sa_hz: Optional Sa in Hz for the cents relabelling.
        out: If given, the figure is saved to this path.

    Returns:
        The matplotlib axes containing the plot.
    """
    plt = _require_matplotlib()
    _, ax = plt.subplots(figsize=(12, 4.5))
    for i, c in enumerate(contours):
        # Small vertical fan so that two trackers on the same grid do not
        # hide each other; the offset is a fraction of the plotted span.
        plot_contour(c, sa_hz=sa_hz, ax=ax, color=f"C{i}")
        _fan_out(ax, i, len(contours))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=9)
    ax.set_title("F0 contour comparison")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        logger.info("wrote %s", out)
    return ax


def plot_tonic_candidates(
    contour: PitchContour,
    candidates: Sequence,
    out: str | Path | None = None,
):
    """Plot the cents histogram with the Sa candidates marked.

    Args:
        contour: Contour whose voiced frames form the histogram.
        candidates: The ranked :class:`~swaras.tonic.TonicCandidate` list.
        out: If given, the figure is saved here.

    Returns:
        The matplotlib axes.
    """
    plt = _require_matplotlib()
    from .tonic import cents_histogram

    centers, density = cents_histogram(contour.f0, contour.voiced)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(centers, density, lw=0.8, color="0.35", label="voiced F0")
    for i, c in enumerate(candidates):
        best = i == 0
        ax.axvline(
            c.cents, color="C3" if best else "0.6", ls="-" if best else "--",
            lw=1.8 if best else 1.0, zorder=3 if best else 2,
        )
        ax.annotate(
            f"Sa = {c.hz:.1f} Hz\nconf {c.confidence:.2f}",
            xy=(c.cents, 1.0 if best else 0.92 if i == 1 else 0.84),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color="C3" if best else "0.35", va="top",
        )
    # The Pa peaks that corroborate each candidate.
    for c in candidates[:1]:
        ax.axvspan(c.cents, c.cents + 702.0, color="C3", alpha=0.08, zorder=1)
        ax.annotate(
            f"Pa (+702c)", xy=(c.cents + 351, 0.55), fontsize=8, color="C3", ha="center"
        )
    ax.set_xlabel("cents above A0 (27.5 Hz)")
    ax.set_ylabel("density")
    ax.set_xlim(centers[0], centers[-1]) if centers.size else None
    ax.set_ylim(0, 1.15)
    ax.set_title("Sa candidates, scored on the peaks at Sa, Pa and the octave")
    ax.legend(loc="upper right", fontsize=9)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
    return ax


def plot_shruti_grid(
    contour: PitchContour,
    notes,
    sa_hz: float,
    out: str | Path | None = None,
    table=None,
):
    """Plot each note's measured pitch against the 22-shruti grid.

    The point of this view is that a note is not *on* a shruti, it is *near*
    one. Each note is drawn at the pitch that was measured, with a connector to
    the shruti it matched, so the length of the connector is the deviation and
    an ambiguous match is visible as a dot sitting between two grid lines
    rather than having to be inferred from a number.

    Args:
        contour: The pitch contour, for the faint background trace.
        notes: :class:`~swaras.shruti.ShrutiNote` matches, in time order.
        sa_hz: Sa in Hz, used to label the grid.
        out: If given, the figure is saved here.
        table: Tuning table; defaults to the packaged one.

    Returns:
        The matplotlib axes.
    """
    plt = _require_matplotlib()
    from .tuning import load_shruti_table

    tbl = table or load_shruti_table()
    grid = tbl.shruti_cents
    if not notes:
        return None

    times = np.array([n.start_s for n in notes], dtype=np.float64)
    measured = np.array([n.cents for n in notes], dtype=np.float64)
    target = np.array([tbl.shrutis[n.index].cents for n in notes], dtype=np.float64)
    confidence = np.array([n.confidence for n in notes], dtype=np.float64)

    # Tall enough that all 22 grid labels fit: at 6 inches the shruti names
    # overprint each other into an unreadable column.
    fig, ax = plt.subplots(figsize=(13, 8.0))
    # Background trace, so the note markers have context.
    trace = contour.f0 if contour is not None else None
    if trace is not None and contour.voiced.any():
        # Unvoiced frames are 0, and log2(0) warns; mask them first.
        safe = np.where(contour.voiced, trace, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = 1200.0 * np.log2(safe / sa_hz)
        ax.plot(contour.times, y, lw=0.8, color="0.82", zorder=1)

    for c in grid:
        ax.axhline(c, color="0.88", lw=0.5, zorder=0)
    # Sa and the octave boundaries are the load-bearing lines.
    for c in (0.0, 1200.0):
        if c in set(grid.tolist()):
            ax.axhline(c, color="0.55", lw=1.1, zorder=0)

    # Connector: measured pitch down to the shruti it matched.
    for t, m, g in zip(times, measured, target):
        ax.plot([t, t], [m, g], color="0.45", lw=1.2, zorder=2)
    # Marker colour encodes how sure the match is.
    scatter = ax.scatter(
        times, measured, c=confidence, cmap="RdYlGn", vmin=0.0, vmax=1.0,
        s=70, edgecolor="0.25", linewidth=0.6, zorder=4,
    )
    for t, m, n in zip(times, measured, notes):
        ax.annotate(
            n.label, xy=(t, m), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=7, color="0.25",
        )

    lo = float(min(measured.min(), target.min())) - 60.0
    hi = float(max(measured.max(), target.max())) + 60.0
    visible = grid[(grid >= lo) & (grid <= hi)]
    ax.set_yticks(visible)
    ax.set_yticklabels(
        [f"{c:.0f}  {tbl.shrutis[i].swar}" for c, i in zip(visible, range(grid.size)) if lo <= c <= hi],
        fontsize=6.5,
    )
    ax.set_ylim(lo, hi)
    ax.set_xlim(times.min() - 0.1, times.max() + 0.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cents above Sa")
    ax.set_title("Measured pitch against the 22-shruti grid")
    bar = fig.colorbar(scatter, ax=ax, pad=0.01)
    bar.set_label("match confidence", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=130)
    return ax


def _detect(samples: np.ndarray, sample_rate: int, name: str, config: PipelineConfig) -> PitchContour | None:
    """Run one detector, logging and skipping on failure."""
    try:
        return get_detector(name, config.pitch).estimate(samples, sample_rate)
    except Exception as exc:
        logger.warning("detector %s unavailable (%s); skipping", name, exc)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: plot the contour of an audio file.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m swaras.plot",
        description="Plot the F0 contour of an audio file, with the detected Sa.",
    )
    parser.add_argument("audio", help="path to an audio file")
    parser.add_argument("--out", default="contour.png", help="output PNG path (default: contour.png)")
    parser.add_argument(
        "--detectors",
        default="melodia,pyin",
        help="comma-separated detector names (default: melodia,pyin)",
    )
    parser.add_argument(
        "--sa",
        type=float,
        default=None,
        help="Sa in Hz; overrides detection, to relabel the y-axis in cents",
    )
    parser.add_argument(
        "--detector",
        default="melodia",
        help="pitch detector used for Sa detection (default: melodia)",
    )
    parser.add_argument(
        "--histogram", action="store_true", help="also plot the Sa candidate histogram"
    )
    parser.add_argument(
        "--shruti-plot", action="store_true", help="also plot the notes against the shruti grid"
    )
    args = parser.parse_args(argv)

    config = DEFAULT_CONFIG
    audio = load_file(args.audio, config.audio)
    print(
        f"loaded {args.audio}: {audio.duration_s:.2f}s @ {audio.sample_rate} Hz "
        f"(source {audio.source_duration_s:.2f}s)"
    )

    sa_hz = args.sa
    tonic_line = ""
    tonic_result = None
    tonic_contour = None
    if sa_hz is None:
        try:
            tonic_contour = _detect(audio.samples, audio.sample_rate, args.detector, config)
            if tonic_contour is not None:
                tonic_result = detect_tonic(
                    tonic_contour, audio.samples, audio.sample_rate, config.tonic
                )
                if tonic_result.is_valid:
                    sa_hz = tonic_result.hz
                    tonic_line = (
                        f"Sa = {sa_hz:.2f} Hz (confidence {tonic_result.confidence:.2f}, "
                        f"method {tonic_result.method})"
                    )
                    alts = ", ".join(f"{c.hz:.1f} Hz" for c in tonic_result.candidates[1:3])
                    if alts:
                        tonic_line += f"\n  alternatives: {alts}"
                    if tonic_result.warning:
                        tonic_line += f"\n  {tonic_result.warning}"
                else:
                    tonic_line = f"Sa not detected: {tonic_result.warning}"
            else:
                tonic_line = f"pitch detector {args.detector!r} unavailable; no Sa detected"
        except Exception as exc:
            tonic_line = f"Sa detection failed: {exc}"
    else:
        tonic_line = f"Sa = {sa_hz:.2f} Hz (manual)"
    if tonic_line:
        print(tonic_line)

    contours = []
    for name in [d.strip() for d in args.detectors.split(",") if d.strip()]:
        contour = _detect(audio.samples, audio.sample_rate, name, config)
        if contour is None:
            continue
        contours.append(contour)
        med = float(np.median(contour.voiced_f0())) if contour.voiced.any() else float("nan")
        print(
            f"  {contour.method:8s} {contour.n_frames:5d} frames  "
            f"voiced {contour.voiced_fraction:5.1%}  median f0 {med:7.1f} Hz"
        )
    if not contours:
        print("no detector produced a contour")
        return 1
    plot_comparison(contours, sa_hz=sa_hz, out=args.out)
    print(f"wrote {args.out}")
    if args.histogram and tonic_contour is not None and tonic_result is not None and tonic_result.is_valid:
        hist_out = Path(args.out).with_name(Path(args.out).stem + "_sa_candidates.png")
        plot_tonic_candidates(tonic_contour, tonic_result.candidates, out=hist_out)
        print(f"wrote {hist_out}")
    if args.shruti_plot and sa_hz:
        from .pipeline import transcribe
        from .plot import plot_shruti_grid

        result = transcribe(
            audio.samples, audio.sample_rate, config, sa_hz=sa_hz, detector=args.detector
        )
        grid_out = Path(args.out).with_name(Path(args.out).stem + "_shruti.png")
        plot_shruti_grid(result.contour, result.transcription.shruti_notes, sa_hz, out=grid_out)
        print(f"wrote {grid_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
