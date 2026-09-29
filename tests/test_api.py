"""Tests for Stage 6: the FastAPI service and the web UI assets.

The behaviour that matters most is that a corrected Sa is re-transcribed from
a cached contour, so the user is not made to wait for pitch tracking again.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from swaras._essentia import essentia_available
from swaras.api import MAX_UPLOAD_BYTES, ContourCache, RetuneRequest, create_app
from tests import synthetic

SR = 22_050
DETECTOR = "pyin"  # available without Essentia


def wav_bytes(phrase=(0, 200, 400, 700), tonic: float = 261.63, note_s: float = 0.45) -> bytes:
    """Render a short phrase and return it as WAV bytes.

    Args:
        phrase: Cents offsets relative to ``tonic``.
        tonic: Sa in Hz.
        note_s: Duration of each note.

    Returns:
        The encoded WAV file.
    """
    freqs = [synthetic.cents_to_hz(c, tonic) for c in phrase]
    y = synthetic.sequence(freqs, note_duration_s=note_s, vibrato_cents=0.0)
    y = np.concatenate([y, np.zeros(int(0.3 * SR))])
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), SR, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    """A test client for a fresh application."""
    return TestClient(create_app())


def post_audio(client: TestClient, data: bytes, **fields) -> dict:
    """POST an audio file to /transcribe and return the JSON body."""
    form = {"detector": DETECTOR, **fields}
    res = client.post(
        "/transcribe",
        files={"file": ("test.wav", data, "audio/wav")},
        data=form,
    )
    assert res.status_code == 200, res.text
    return res.json()


# --- health and assets -----------------------------------------------------


def test_health_reports_availability(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert DETECTOR in body["detectors"]
    assert body["essentia"] is essentia_available()


def test_index_and_assets_are_served(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "Audio to Swar" in index.text
    # The page must reference the assets it needs, or it renders unstyled and
    # inert in a browser.
    assert "/app.js" in index.text
    assert "/style.css" in index.text
    for asset, kind in (("/app.js", "javascript"), ("/style.css", "css")):
        res = client.get(asset)
        assert res.status_code == 200
        assert kind in res.headers["content-type"]


def test_ui_has_the_controls_the_task_requires(client: TestClient) -> None:
    html = client.get("/").text
    for needed in ('id="file"', 'id="record"', 'id="copy"', 'id="sa-fix"',
                   'id="show-shruti"', 'id="show-detail"', 'id="sa-alts"'):
        assert needed in html, needed
    js = client.get("/app.js").text
    # The candidate buttons must actually re-transcribe rather than just render.
    assert "retune" in js
    assert "/retune" in js


# --- transcribe ------------------------------------------------------------


def test_transcribe_returns_notation(client: TestClient) -> None:
    body = post_audio(client, wav_bytes())
    assert body["notation"] == "S R G P"
    assert body["note_count"] == 4
    assert body["sa"]["hz"] == pytest.approx(261.63, abs=2.0)


def test_transcribe_payload_shape(client: TestClient) -> None:
    body = post_audio(client, wav_bytes())
    for key in ("notation", "notation_long", "sa", "notes", "shruti",
                "detector", "elapsed_s", "audio", "contour_id", "diagnostics"):
        assert key in body, key
    n = body["notes"][0]
    for key in ("swar", "short", "label", "octave", "cents",
                "absolute_cents", "deviation_cents", "start_s", "end_s"):
        assert key in n, key
    s = body["sa"]
    for key in ("hz", "confidence", "source", "candidates", "warning"):
        assert key in s, key


def test_transcribe_with_supplied_sa_skips_detection(client: TestClient) -> None:
    body = post_audio(client, wav_bytes(), sa_hz="220.0")
    assert body["sa"]["hz"] == pytest.approx(220.0)
    assert body["sa"]["source"] == "manual"
    # With Sa given there is nothing to correct later, so nothing is cached.
    assert body["contour_id"] == ""


def test_transcribe_includes_the_shruti_layer(client: TestClient) -> None:
    body = post_audio(client, wav_bytes())
    assert body["shruti"]["line"]
    assert body["shruti"]["statistics"]["count"] == 4
    assert body["shruti"]["notes"][0]["shruti_index"] == 0


def test_transcribe_rejects_empty_upload(client: TestClient) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"detector": DETECTOR},
    )
    assert res.status_code == 400


def test_transcribe_rejects_non_audio(client: TestClient) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("junk.wav", b"definitely not audio" * 100, "audio/wav")},
        data={"detector": DETECTOR},
    )
    assert res.status_code == 422
    assert "detail" in res.json()


def test_transcribe_rejects_oversized_upload(client: TestClient) -> None:
    big = b"\0" * (MAX_UPLOAD_BYTES + 1)
    res = client.post(
        "/transcribe",
        files={"file": ("big.wav", big, "audio/wav")},
        data={"detector": DETECTOR},
    )
    assert res.status_code == 413


# --- retune: the reason this endpoint exists --------------------------------


def test_retune_changes_the_transcription(client: TestClient) -> None:
    """Correcting Sa must actually change the notes, not just the label."""
    body = post_audio(client, wav_bytes())
    assert body["notation"] == "S R G P"
    cid = body["contour_id"]
    res = client.post("/retune", json={"contour_id": cid, "sa_hz": 196.0})
    assert res.status_code == 200
    assert res.json()["notation"] != "S R G P"
    assert res.json()["sa"]["hz"] == pytest.approx(196.0)


def test_retune_is_far_faster_than_a_fresh_transcription(client: TestClient) -> None:
    """The whole point of the cache: a correction must not re-track pitch."""
    data = wav_bytes()
    first = post_audio(client, data)
    cid = first["contour_id"]

    t0 = time.monotonic()
    for _ in range(5):
        client.post("/retune", json={"contour_id": cid, "sa_hz": 220.0})
    per_call = (time.monotonic() - t0) / 5

    # pYIN over four notes takes a substantial fraction of a second. The
    # assertion is deliberately loose; it is here to catch the cache silently
    # not being used, not to pin a wall-clock number.
    assert per_call < first["elapsed_s"], (per_call, first["elapsed_s"])


def test_retune_reuses_the_cached_contour(client: TestClient) -> None:
    data = wav_bytes()
    first = post_audio(client, data)
    cid = first["contour_id"]
    client.post("/retune", json={"contour_id": cid, "sa_hz": 220.0})
    # Keying on content means the same file uploaded again reuses the contour.
    second = post_audio(client, data)
    assert second["contour_id"] == cid


def test_retune_unknown_contour_is_404(client: TestClient) -> None:
    res = client.post("/retune", json={"contour_id": "does-not-exist", "sa_hz": 261.63})
    assert res.status_code == 404
    assert "upload" in res.json()["detail"].lower()


def test_retune_rejects_bad_sa(client: TestClient) -> None:
    body = post_audio(client, wav_bytes())
    res = client.post("/retune", json={"contour_id": body["contour_id"], "sa_hz": -5})
    assert res.status_code == 422


def test_retune_is_deterministic(client: TestClient) -> None:
    body = post_audio(client, wav_bytes())
    cid = body["contour_id"]
    a = client.post("/retune", json={"contour_id": cid, "sa_hz": 220.0}).json()
    b = client.post("/retune", json={"contour_id": cid, "sa_hz": 220.0}).json()
    assert a["notation"] == b["notation"]
    assert a["shruti"]["line"] == b["shruti"]["line"]


# --- front-end wiring ------------------------------------------------------


def test_every_element_the_script_uses_exists_in_the_page(client: TestClient) -> None:
    """Catch the likeliest front-end bug: an id typoed in one file only.

    Without a browser in the test loop this is the only way to know that
    ``$("sa-fix")`` and friends actually resolve. An id that appears in the
    script but not the page silently does nothing at runtime.
    """
    import re

    html = client.get("/").text
    js = client.get("/app.js").text
    page_ids = set(re.findall(r'id="([^"]+)"', html))
    # Ids the script reaches for. The $("tab-x") family is built from the tab
    # data attributes, so those are checked separately below.
    used = set(re.findall(r'\$\("([^"]+)"\)', js))
    missing = sorted(used - page_ids)
    assert not missing, f"app.js refers to ids absent from index.html: {missing}"


def test_frontend_logic_is_checked_under_node() -> None:
    """Run tests/frontend_check.js if node is available.

    The Python suite cannot see a stuck spinner: a `fetch` that never settles
    leaves the page looking permanently busy, and nothing in the API responds.
    That bug is real -- there were no request deadlines at all -- so the
    front-end logic is driven under a minimal DOM instead. Skipped, not failed,
    where node is absent, since it is a development-only dependency.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; front-end logic check skipped")
    script = Path(__file__).with_name("frontend_check.js")
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, cwd=root, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_page_declares_an_inline_favicon(client: TestClient) -> None:
    """A missing favicon makes every page load log a 404 for /favicon.ico.

    Browsers request it automatically whether or not the page asks for one, so
    the fix is a <link rel="icon"> with a data URI: that costs no extra request
    and needs no route.
    """
    import re

    html = client.get("/").text
    match = re.search(r'<link rel="icon" href="([^"]+)"', html)
    assert match, "no <link rel=icon>, so browsers will request /favicon.ico"
    href = match.group(1)
    assert href.startswith("data:image/svg+xml,"), "favicon should be inline, not a request"
    # It has to be well-formed XML, or the browser silently ignores it.
    import urllib.parse
    import xml.dom.minidom

    xml.dom.minidom.parseString(urllib.parse.unquote(href.split(",", 1)[1]))


def test_tab_bodies_are_named_after_their_tab_buttons(client: TestClient) -> None:
    """The tab switcher builds ids as ``tab-<name>``; they must all exist."""
    import re

    html = client.get("/").text
    tabs = set(re.findall(r'data-tab="([^"]+)"', html))
    assert tabs, "no tabs found"
    for name in tabs:
        assert f'id="tab-{name}"' in html, f"no body for tab {name}"


def test_script_only_calls_endpoints_that_exist(client: TestClient) -> None:
    """Every URL the page posts to must be a real route."""
    import re

    js = client.get("/app.js").text
    paths = set(re.findall(r'(?:post|fetch)\("(/[^"]*)"', js))
    routes = {r.path for r in client.app.routes}
    for p in paths:
        assert p in routes, f"app.js calls {p}, which is not a route ({sorted(routes)})"


# --- the cache itself ------------------------------------------------------


def test_cache_roundtrip() -> None:
    from swaras.pitch import PitchContour

    contour = PitchContour(
        times=np.zeros(3), f0=np.array([220.0, 0.0, 440.0]),
        voiced=np.array([True, False, True]), confidence=np.zeros(3), hop_seconds=0.02,
    )
    cache = ContourCache(size=2)
    key = cache.key(b"abc", "pyin")
    assert cache.get(key) is None
    cache.put(key, contour, 1.0)
    assert cache.get(key) is not None
    assert len(cache) == 1


def test_cache_evicts_least_recently_used() -> None:
    from swaras.pitch import PitchContour

    contour = PitchContour(
        times=np.zeros(1), f0=np.array([220.0]), voiced=np.array([True]),
        confidence=np.array([1.0]), hop_seconds=0.02,
    )
    cache = ContourCache(size=2)
    a, b, c = "a", "b", "c"
    cache.put(a, contour, 1.0)
    cache.put(b, contour, 1.0)
    cache.get(a)          # a becomes most recently used
    cache.put(c, contour, 1.0)
    assert cache.get(b) is None, "b was least recently used and should be gone"
    assert cache.get(a) is not None


def test_cache_expires_entries() -> None:
    from swaras.pitch import PitchContour

    contour = PitchContour(
        times=np.zeros(1), f0=np.array([220.0]), voiced=np.array([True]),
        confidence=np.array([1.0]), hop_seconds=0.02,
    )
    cache = ContourCache(size=2, ttl_s=-1.0)  # already expired
    cache.put("k", contour, 1.0)
    assert cache.get("k") is None


def test_cache_key_depends_on_content_and_detector() -> None:
    cache = ContourCache()
    assert cache.key(b"a", "pyin") == cache.key(b"a", "pyin")
    assert cache.key(b"a", "pyin") != cache.key(b"b", "pyin")
    assert cache.key(b"a", "pyin") != cache.key(b"a", "melodia")


def test_clear_cache_endpoint(client: TestClient) -> None:
    body = post_audio(client, wav_bytes())
    assert client.get("/health").json()["cached_contours"] == 1
    assert client.delete("/cache").status_code == 200
    assert client.get("/health").json()["cached_contours"] == 0
    res = client.post("/retune", json={"contour_id": body["contour_id"], "sa_hz": 220.0})
    assert res.status_code == 404


def test_retune_request_model_rejects_non_positive_sa() -> None:
    with pytest.raises(Exception):
        RetuneRequest(contour_id="x", sa_hz=0.0)
