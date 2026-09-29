"""Command-line interface: ``python -m swaras transcribe file.wav``.

The entry point is deliberately thin. All it does is parse arguments, call the
pipeline, and print, so the CLI can never disagree with the API about how a
recording is transcribed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from ._essentia import essentia_available
from .format import format_detailed, format_shruti_table, format_summary
from .pipeline import transcribe_file

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m swaras",
        description="Transcribe a melodic recording as Indian swar notation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("transcribe", help="transcribe an audio file")
    tr.add_argument("file", help="path to an audio file (wav, flac, ogg, mp3, m4a...)")
    tr.add_argument(
        "--sa", type=float, default=None, help="Sa in Hz; skips tonic detection entirely"
    )
    tr.add_argument(
        "--detector",
        default="melodia",
        help="pitch tracker: melodia (default) or pyin",
    )
    tr.add_argument(
        "--detailed",
        action="store_true",
        help="print per-note timing, cents and deviation",
    )
    tr.add_argument(
        "--shruti",
        action="store_true",
        help="print the 22-shruti layer: nearest shruti and cents deviation",
    )
    tr.add_argument("--json", action="store_true", help="print the full result as JSON")
    tr.add_argument(
        "--plain", action="store_true", help="print only the notation, nothing else"
    )
    tr.add_argument("--out", type=Path, default=None, help="write the output to this file")

    tonic = sub.add_parser("tonic", help="detect Sa only")
    tonic.add_argument("file", help="path to an audio file")
    tonic.add_argument("--detector", default="melodia", help="pitch tracker (default: melodia)")

    serve = sub.add_parser("serve", help="run the web UI and API")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    serve.add_argument("--reload", action="store_true", help="reload on code changes")

    return parser


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Run the transcribe subcommand.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    try:
        result = transcribe_file(args.file, sa_hz=args.sa, detector=args.detector)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # A missing optional backend. The message is for the user, not a
        # traceback: the most likely cause is a plain `pip install swaras`
        # on a machine without Essentia, and the fix is one command.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    t = result.transcription
    if args.json:
        output = t.to_json()
    elif args.plain:
        output = t.shruti_text if args.shruti and t.shruti_text else t.text
    else:
        parts = [format_summary(t)]
        if args.detailed:
            parts.append("")
            parts.append(format_detailed(t.notes))
        if args.shruti:
            parts.append("")
            parts.append(format_shruti_table(t.shruti_notes))
        output = "\n".join(parts)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(output)
    return 0


def _cmd_tonic(args: argparse.Namespace) -> int:
    """Run the tonic subcommand.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    from .audio_io import load_file
    from .config import DEFAULT_CONFIG
    from .pitch import get_detector
    from .tonic import detect_tonic

    try:
        audio = load_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        contour = get_detector(args.detector).estimate(audio.samples, audio.sample_rate)
        result = detect_tonic(contour, audio.samples, audio.sample_rate)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not result.is_valid:
        print(f"error: {result.warning}", file=sys.stderr)
        return 1

    print(f"Sa = {result.hz:.2f} Hz  (confidence {result.confidence:.2f}, method {result.method})")
    best = result.candidates[0]
    print(
        f"  evidence: sa {best.sa_peak:.2f}  pa {best.pa_peak:.2f}  "
        f"octave {best.octave_peak:.2f}  methods {','.join(best.methods)}"
    )
    for c in result.candidates[1:]:
        print(f"  alt: {c.hz:8.2f} Hz  confidence {c.confidence:.2f}  ({','.join(c.methods)})")
    if result.essentia_hz:
        print(f"  essentia raw: {result.essentia_hz:.2f} Hz")
    if result.warning:
        print(f"  warning: {result.warning}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI application with uvicorn.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed; run `pip install 'uvicorn[standard]'`", file=sys.stderr)
        return 1
    print(f"Serving the web UI on http://{args.host}:{args.port}/")
    uvicorn.run(
        "swaras.api:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_level="info",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "transcribe":
        return _cmd_transcribe(args)
    if args.command == "tonic":
        return _cmd_tonic(args)
    if args.command == "serve":
        return _cmd_serve(args)
    return 1  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
