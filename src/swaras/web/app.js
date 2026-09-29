/* Audio-to-Swar front end.
 *
 * The one structural idea: `state.contourId` is what makes correcting Sa
 * instant. The first request tracks pitch and detects Sa and hands back a
 * contour id; every correction re-posts only that id and the new Sa, and the
 * server re-runs everything after pitch tracking, which is milliseconds.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  blob: null,        // the audio to send
  filename: "recording.wav",
  result: null,
  contourId: null,
  recorder: null,
  chunks: [],
  recTimer: null,
  recSeconds: 0,
};

/* --- tabs ---------------------------------------------------------------- */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((b) => b.classList.remove("active"));
    tab.classList.add("active");
    $(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

/* --- file input and drag and drop ----------------------------------------- */

function setAudio(blob, name) {
  state.blob = blob;
  state.filename = name || "audio";
  state.contourId = null;      // a new file invalidates the old contour
  $("go").disabled = false;
  $("go").textContent = "Transcribe";
}

$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) setAudio(f, f.name);
});

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) setAudio(f, f.name);
});

/* --- microphone ---------------------------------------------------------- */

$("record").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : MediaRecorder.isTypeSupported("audio/ogg")
        ? "audio/ogg"
        : "";
    state.recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    state.chunks = [];
    state.recorder.ondataavailable = (e) => {
      if (e.data.size) state.chunks.push(e.data);
    };
    state.recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const type = state.recorder.mimeType || "audio/webm";
      const ext = type.includes("ogg") ? "ogg" : "webm";
      const blob = new Blob(state.chunks, { type });
      const url = URL.createObjectURL(blob);
      const player = $("playback");
      player.src = url;
      player.hidden = false;
      setAudio(blob, `recording.${ext}`);
    };
    state.recorder.start();
    state.recSeconds = 0;
    $("rec-time").textContent = "0:00";
    state.recTimer = setInterval(() => {
      state.recSeconds += 1;
      const m = Math.floor(state.recSeconds / 60);
      const s = String(state.recSeconds % 60).padStart(2, "0");
      $("rec-time").textContent = `${m}:${s}`;
    }, 1000);
    $("record").hidden = true;
    $("stop").hidden = false;
  } catch (err) {
    showError(`Microphone unavailable: ${err.message}. Use the Upload tab instead.`);
  }
});

$("stop").addEventListener("click", () => {
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  clearInterval(state.recTimer);
  $("record").hidden = false;
  $("stop").hidden = true;
});

/* --- requests ------------------------------------------------------------ */

/* The busy overlay must never be able to stick. fetch() has no timeout of its
 * own, so a request that never settles would leave the spinner up for ever and
 * the page would look permanently frozen. Every call therefore carries an
 * AbortController deadline, and a visible elapsed counter runs alongside it so
 * a genuinely slow job reads as slow rather than as hung. */
let busyTimer = null;
let busyStarted = 0;

function busy(on, text) {
  $("busy").hidden = !on;
  $("busy-text").textContent = text || "Working…";
  clearInterval(busyTimer);
  busyTimer = null;
  if (on) {
    busyStarted = Date.now();
    busyTimer = setInterval(() => {
      const s = Math.round((Date.now() - busyStarted) / 1000);
      $("busy-text").textContent = `${text || "Working…"} (${s}s)`;
    }, 1000);
  }
}

/** Raise a fetch failure to something a user can act on. */
function friendlyError(err) {
  if (err && err.name === "AbortError") {
    return "The server took too long to answer and the request was cancelled. " +
      "Try a shorter excerpt, or --detector pyin, which is lighter than Melodia.";
  }
  return err && err.message ? err.message : "unexpected error";
}

function showError(message) {
  $("error-panel").hidden = false;
  $("error-text").textContent = message;
  $("result-panel").hidden = true;
}

async function post(url, options, timeoutMs = 300000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : `request failed (${res.status})`;
    throw new Error(detail);
  }
  return body;
}

$("go").addEventListener("click", async () => {
  if (!state.blob) return;
  $("error-panel").hidden = true;
  busy(true, "Tracking pitch and detecting Sa…");
  try {
    const form = new FormData();
    form.append("file", state.blob, state.filename);
    form.append("detector", $("detector").value);
    const sa = parseFloat($("sa").value);
    if (!Number.isNaN(sa) && sa > 0) form.append("sa_hz", String(sa));
    const data = await post("/transcribe", { method: "POST", body: form });
    state.result = data;
    state.contourId = data.contour_id || null;
    render(data);
  } catch (err) {
    showError(friendlyError(err));
  } finally {
    busy(false);
  }
});

$("apply-sa").addEventListener("click", retune);
$("sa-fix").addEventListener("keydown", (e) => {
  if (e.key === "Enter") retune();
});

async function retune(saHz) {
  if (saHz === undefined) {
    saHz = parseFloat($("sa-fix").value);
    if (Number.isNaN(saHz) || saHz <= 0) {
      $("sa-warning").textContent = "Enter a Sa in Hz first.";
      return;
    }
  }
  if (!state.contourId) {
    $("sa-warning").textContent = "Re-upload the file: its pitch contour is no longer cached.";
    return;
  }
  $("sa-warning").textContent = "";
  busy(true, "Re-transcribing…");
  try {
    const data = await post("/retune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contour_id: state.contourId, sa_hz: saHz }),
    }, 30000);
    // A corrected Sa replaces the detection, so the old alternatives are no
    // longer meaningful and are not carried over.
    data.sa.candidates = [];
    data.sa.source = "manual";
    state.result = data;
    render(data);
  } catch (err) {
    $("sa-warning").textContent = friendlyError(err);
  } finally {
    busy(false);
  }
}

/* --- rendering ----------------------------------------------------------- */

function render(d) {
  $("result-panel").hidden = false;
  $("notation").textContent = d.notation || "(no notes)";

  $("sa-hz").textContent = `${d.sa.hz.toFixed(2)} Hz`;
  const srcLabel = d.sa.source === "manual" ? "corrected" : d.sa.source;
  $("sa-meta").textContent = `(${srcLabel}, confidence ${d.sa.confidence.toFixed(2)})`;
  $("sa-warning").textContent = d.sa.warning || "";
  $("sa-fix").value = d.sa.hz.toFixed(2);

  const alts = $("sa-alts");
  alts.innerHTML = "";
  (d.sa.candidates || []).forEach((c) => {
    const b = document.createElement("button");
    b.className = "alt";
    b.innerHTML = `${c.hz.toFixed(1)} Hz<span class="conf">conf ${c.confidence.toFixed(2)}</span>`;
    b.title = `Re-transcribe with Sa = ${c.hz.toFixed(2)} Hz`;
    b.addEventListener("click", () => retune(c.hz));
    alts.appendChild(b);
  });

  renderShruti(d);
  renderNotes(d);

  const a = d.audio || {};
  $("meta").textContent =
    `${d.note_count} notes · ${(a.duration_s || 0).toFixed(2)}s audio · ` +
    `${d.detector} · Sa by ${d.sa.source} · ${d.elapsed_s}s`;
}

function renderShruti(d) {
  const on = $("show-shruti").checked;
  $("shruti-block").hidden = !on || !d.shruti;
  if (!on || !d.shruti) return;
  $("shruti-line").textContent = d.shruti.line;
  const s = d.shruti.statistics;
  $("shruti-stats").textContent =
    `mean deviation ${s.mean_abs_deviation_cents}c · worst ${s.max_abs_deviation_cents}c · ` +
    `${s.unclear_count} of ${s.count} notes are ambiguous (confidence below 0.5)`;
}

function renderNotes(d) {
  const on = $("show-detail").checked;
  $("detail-block").hidden = !on;
  if (!on) return;
  const body = document.querySelector("#notes tbody");
  body.innerHTML = "";
  const shrutis = (d.shruti && d.shruti.notes) || [];
  d.notes.forEach((n, i) => {
    const s = shrutis[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="num">${i + 1}</td>
      <td>${n.swar}</td>
      <td class="swar">${n.label}</td>
      <td class="num">${n.start_s.toFixed(2)}</td>
      <td class="num">${(n.end_s - n.start_s).toFixed(2)}</td>
      <td class="num">${n.absolute_cents.toFixed(1)}</td>
      <td class="num">${n.deviation_cents >= 0 ? "+" : ""}${n.deviation_cents.toFixed(1)}</td>
      <td>${s ? s.label : ""}</td>
      <td class="num">${s ? (s.deviation_cents >= 0 ? "+" : "") + s.deviation_cents.toFixed(1) : ""}</td>
      <td class="num">${s ? s.confidence.toFixed(2) : ""}</td>`;
    body.appendChild(tr);
  });
}

$("show-shruti").addEventListener("change", () => state.result && renderShruti(state.result));
$("show-detail").addEventListener("change", () => state.result && renderNotes(state.result));

/* --- copy ---------------------------------------------------------------- */

$("copy").addEventListener("click", async () => {
  const d = state.result;
  if (!d) return;
  const text = $("show-shruti").checked && d.shruti
    ? `${d.notation}\n${d.shruti.line}\nSa = ${d.sa.hz.toFixed(2)} Hz`
    : `${d.notation}\nSa = ${d.sa.hz.toFixed(2)} Hz`;
  const button = $("copy");
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch {
    // Clipboard access can be refused; the textarea fallback still works.
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    button.textContent = "Copied";
  }
  setTimeout(() => { button.textContent = "Copy"; }, 1600);
});

/* --- startup check ------------------------------------------------------- */

const status = $("server-status");
fetch("/health", { signal: AbortSignal.timeout(10000) })
  .then((r) => r.json())
  .then((h) => {
    const which = h.essentia ? "Melodia + pYIN" : "pYIN only (no Essentia)";
    status.textContent = `server connected · ${which}`;
    status.className = "status ok";
    if (!h.essentia) {
      const opt = $("detector").querySelector('option[value="melodia"]');
      opt.disabled = true;
      $("detector").value = "pyin";
    }
  })
  .catch((err) => {
    status.textContent = err.name === "TimeoutError"
      ? "server did not answer within 10s"
      : "cannot reach the server";
    status.className = "status bad";
  });
