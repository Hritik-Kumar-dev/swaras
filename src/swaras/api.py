"""Stage 6: the FastAPI service.

Two endpoints matter. ``POST /transcribe`` does the expensive work once:
decode, track pitch, detect Sa, segment. ``POST /retune`` re-runs everything
after pitch tracking with a corrected Sa, which is a few milliseconds of array
work. That split is the whole point -- tonic errors are usually octave errors,
so the user will correct Sa often, and making them re-upload and re-analyse a
recording each time would make the correction unusable.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import transcribe, transcribe_contour
from .pitch import PitchContour
from .tuning import load_shruti_table

logger = logging.getLogger(__name__)

#: Uploads larger than this are rejected outright. A long recording is minutes
#: of audio, not gigabytes, and an unbounded body is a denial-of-service
#: waiting to happen.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

#: How long a decoded contour is kept for re-tuning, and how many at once.
CACHE_SIZE = 8
CACHE_TTL_S = 30 * 60


class RetuneRequest(BaseModel):
    """Body of a re-transcription request with a corrected Sa."""

    contour_id: str = Field(..., description="Identifier returned by /transcribe")
    sa_hz: float = Field(..., gt=0, description="The corrected Sa, in Hz")
    detector: str | None = Field(None, description="Unused; kept for symmetry")


class _ContourEntry:
    """A cached pitch contour plus its bookkeeping."""

    __slots__ = ("contour", "created", "duration_s")

    def __init__(self, contour: PitchContour, duration_s: float) -> None:
        self.contour = contour
        self.created = time.monotonic()
        self.duration_s = duration_s


class ContourCache:
    """A small LRU cache of pitch contours, keyed by upload content.

    Keying on the content hash means the same file uploaded twice hits the
    cache, and a *different* file can never collide onto someone else's
    contour.
    """

    def __init__(self, size: int = CACHE_SIZE, ttl_s: float = CACHE_TTL_S) -> None:
        self._entries: OrderedDict[str, _ContourEntry] = OrderedDict()
        self._size = size
        self._ttl = ttl_s

    def key(self, data: bytes, detector: str) -> str:
        """Stable identifier for an upload and the tracker used on it."""
        digest = hashlib.sha256(data).hexdigest()[:32]
        return f"{digest}-{detector}"

    def put(self, key: str, contour: PitchContour, duration_s: float) -> None:
        """Store a contour, evicting the least recently used entry if full."""
        self._entries[key] = _ContourEntry(contour, duration_s)
        self._entries.move_to_end(key)
        while len(self._entries) > self._size:
            self._entries.popitem(last=False)

    def get(self, key: str) -> _ContourEntry | None:
        """Fetch a contour, or ``None`` if absent or expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.created > self._ttl:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: Pipeline configuration; defaults to :data:`DEFAULT_CONFIG`.

    Returns:
        The configured application.
    """
    cfg = config or DEFAULT_CONFIG
    tuning = load_shruti_table()
    cache = ContourCache()
    static_dir = __import__("pathlib").Path(__file__).resolve().parent / "web"

    app = FastAPI(
        title="Audio-to-Swar",
        version="0.1.0",
        description="Transcribe a melodic recording as Indian swar notation.",
    )
    app.state.cache = cache
    app.state.config = cfg

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe, also reporting what is available."""
        from ._essentia import essentia_available

        return {
            "status": "ok",
            "detectors": ["melodia", "pyin"] if essentia_available() else ["pyin"],
            "essentia": essentia_available(),
            "cached_contours": len(cache),
        }

    @app.post("/transcribe")
    async def transcribe_endpoint(
        file: UploadFile = File(..., description="an audio file"),
        sa_hz: float | None = Form(None, description="Sa in Hz; skips detection"),
        detector: str = Form("melodia", description="melodia or pyin"),
    ) -> JSONResponse:
        """Transcribe an uploaded file, detecting Sa unless one is given."""
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
            )
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")

        started = time.monotonic()
        try:
            result = transcribe_upload_bytes(data, file.filename or "upload.wav", cfg, sa_hz, detector)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            # Raised when an optional backend is missing.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        elapsed = time.monotonic() - started

        contour = result.contour
        if sa_hz is None:
            # Only cache when Sa was detected: a cached contour is what makes
            # a later correction cheap, and that is only worth keeping for a
            # first pass.
            cid = cache.key(data, contour.method)
            cache.put(cid, contour, result.audio.duration_s)
        else:
            cid = ""

        return JSONResponse(build_payload(result, contour_id=cid, elapsed_s=elapsed))

    @app.post("/retune")
    def retune_endpoint(request: RetuneRequest) -> JSONResponse:
        """Re-transcribe a cached contour with a corrected Sa.

        Skips pitch tracking and tonic detection entirely, so it is
        milliseconds rather than seconds. Returns 404 when the cached contour
        has expired or the id is unknown, which the UI reports as "re-upload".
        """
        entry = cache.get(request.contour_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="that contour is no longer cached; please upload the file again",
            )
        started = time.monotonic()
        result = transcribe_contour(
            entry.contour, request.sa_hz, cfg, tuning, duration_s=entry.duration_s
        )
        elapsed = time.monotonic() - started
        return JSONResponse(build_payload(result, contour_id=request.contour_id, elapsed_s=elapsed))

    @app.delete("/cache")
    def clear_cache() -> dict[str, str]:
        """Drop every cached contour."""
        cache.clear()
        return {"status": "cleared"}

    @app.get("/")
    def index() -> FileResponse:
        """The single-page UI."""
        return FileResponse(static_dir / "index.html")

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(static_dir / "app.js")

    @app.get("/style.css")
    def app_css() -> FileResponse:
        return FileResponse(static_dir / "style.css")

    return app


def transcribe_upload_bytes(
    data: bytes,
    filename: str,
    config: PipelineConfig,
    sa_hz: float | None,
    detector: str,
):
    """Transcribe raw upload bytes.

    Split out from the endpoint so the route handler stays about HTTP.

    Args:
        data: Raw encoded file contents.
        filename: Original filename, used only to infer the container.
        config: Pipeline configuration.
        sa_hz: Sa in Hz, or ``None`` to detect it.
        detector: Pitch detector name.

    Returns:
        The full pipeline result.
    """
    from .pipeline import transcribe_upload

    return transcribe_upload(data, filename, config, sa_hz, detector)


def build_payload(result, contour_id: str, elapsed_s: float) -> dict[str, Any]:
    """Build the JSON response body from a pipeline result.

    Args:
        result: The pipeline result.
        contour_id: Cache identifier to hand back, or ``""`` when the audio was
            not cached.
        elapsed_s: Wall-clock time the request took.

    Returns:
        A JSON-serialisable dict.
    """
    from .format import to_jsonable

    t = result.transcription
    payload = to_jsonable(t.to_dict())
    payload["contour_id"] = contour_id
    payload["elapsed_s"] = round(elapsed_s, 3)
    payload["audio"] = {
        "duration_s": round(result.audio.duration_s, 3),
        "source_duration_s": round(result.audio.source_duration_s, 3),
        "sample_rate": result.audio.sample_rate,
    }
    # `sa` groups the tonic information the UI needs to correct it, rather than
    # making the page know the flat field names.
    payload["sa"] = {
        "hz": round(t.sa_hz, 3),
        "confidence": round(t.sa_confidence, 4),
        "source": t.sa_source,
        "candidates": t.tonic_candidates,
        "warning": t.tonic_warning,
    }
    return payload


app = create_app()
