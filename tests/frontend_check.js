// Minimal DOM + fetch shim to exercise app.js logic under node.
//
// Purpose: the busy overlay used to be able to stick for ever, because fetch()
// has no timeout of its own. That class of bug is invisible to a static check
// and to the Python suite, so drive the real script here against a fake DOM
// and assert the overlay always clears, including when a request fails.
//
// Run: node tests/frontend_check.js

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");
const HTML = fs.readFileSync(path.join(ROOT, "src/swaras/web/index.html"), "utf8");

/* --- a DOM just rich enough for app.js ------------------------------------ */

function makeEl(tag) {
  return {
    tagName: String(tag).toUpperCase(),
    children: [],
    _attrs: {},
    _listeners: {},
    style: {},
    dataset: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); },
      remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    set className(v) { this._className = v; },
    get className() { return this._className || ""; },
    set textContent(v) { this._text = String(v); },
    get textContent() { return this._text || ""; },
    set innerHTML(v) { this._html = v; },
    get innerHTML() { return this._html || ""; },
    set hidden(v) { this._hidden = !!v; },
    get hidden() { return !!this._hidden; },
    set value(v) { this._value = v; },
    get value() { return this._value === undefined ? "" : this._value; },
    set src(v) { this._src = v; },
    get src() { return this._src; },
    set disabled(v) { this._disabled = !!v; },
    get disabled() { return !!this._disabled; },
    get files() { return this._files || []; },
    set files(v) { this._files = v; },
    addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); },
    dispatch(ev, arg) { (this._listeners[ev] || []).forEach((f) => f(arg || {})); },
    appendChild(c) { this.children.push(c); return c; },
    remove() {},
    select() {},
    querySelector() { return makeEl("option"); },
    querySelectorAll() { return []; },
  };
}

const IDS = [
  "busy", "busy-text", "go", "sa", "detector", "file", "dropzone", "record", "stop",
  "rec-time", "playback", "result-panel", "error-panel", "error-text", "notation",
  "copy", "sa-hz", "sa-meta", "sa-alts", "sa-fix", "apply-sa", "sa-warning",
  "show-shruti", "show-detail", "shruti-block", "shruti-line", "shruti-stats",
  "detail-block", "meta", "server-status", "notes", "sa-hint",
];

function freshDom() {
  const els = new Map();
  for (const id of IDS) els.set(id, makeEl("div"));
  els.get("sa").value = "";
  els.get("sa-fix").value = "";
  els.get("detector").value = "pyin";
  const tabs = [makeEl("button"), makeEl("button")];
  tabs[0].dataset.tab = "upload";
  tabs[1].dataset.tab = "record";
  const body = makeEl("tbody");
  const document_ = {
    getElementById: (id) => els.get(id) || null,
    querySelector: (sel) => (sel === "#notes tbody" ? body : makeEl("div")),
    querySelectorAll: (sel) => (sel === ".tab" ? tabs : []),
    createElement: (t) => makeEl(t),
    body: makeEl("body"),
    execCommand: () => true,
  };
  return { els, document_ };
}

function payload() {
  return {
    notation: "S R G P",
    notation_long: "Sa Re Ga Pa",
    note_count: 4,
    detector: "pyin",
    elapsed_s: 0.06,
    sa: { hz: 262.08, confidence: 1, source: "histogram", candidates: [], warning: null },
    audio: { duration_s: 2.0 },
    notes: [
      { swar: "Sa", label: "S", start_s: 0, end_s: 0.4, absolute_cents: 2, deviation_cents: 2 },
    ],
    shruti: {
      line: "S0 +2c",
      notes: [{ label: "S0", deviation_cents: 2, confidence: 1 }],
      statistics: { mean_abs_deviation_cents: 2, max_abs_deviation_cents: 2, unclear_count: 0, count: 1 },
    },
  };
}

function health(essentia) {
  return { ok: true, status: 200, json: async () => ({ status: "ok", essentia, detectors: [] }) };
}

/* --- harness -------------------------------------------------------------- */

let failures = 0;
function assert(cond, what) {
  console.log(`  ${cond ? "ok  " : "FAIL"} ${what}`);
  if (!cond) failures += 1;
}

const tick = () => new Promise((r) => setTimeout(r, 10));

/** Load app.js against a fresh DOM, with the given fetch behaviour.
 *
 * `transform` may rewrite the source, which is how the hung-request test
 * shortens the deadline without a test hook in production code. */
function boot(fetchImpl, transform) {
  const { els, document_ } = freshDom();
  let src = fs.readFileSync(path.join(ROOT, "src/swaras/web/app.js"), "utf8");
  if (transform) src = transform(src);
  const ctx = {
    document: document_,
    // Always hand back a real Promise, even for a plain object, because app.js
    // chains .then() on the result of the startup probe.
    fetch: (url, opts) => new Promise((resolve, reject) => {
      if (opts && opts.signal) {
        if (opts.signal.aborted) {
          reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
          return;
        }
        opts.signal.addEventListener("abort", () => {
          reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
        });
      }
      Promise.resolve()
        .then(() => fetchImpl(url, opts))
        .then(resolve, reject);
    }),
    FormData: class { append() {} },
    Blob: class {},
    URL: { createObjectURL: () => "blob:x", revokeObjectURL: () => {} },
    MediaRecorder: undefined,
    navigator: { clipboard: { writeText: async () => {} }, mediaDevices: undefined },
    AbortController,
    AbortSignal,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    console,
  };
  vm.createContext(ctx);
  vm.runInContext(src, ctx, { filename: "app.js" });
  return { els, ctx };
}

/** Choose a file, the way the drop zone does. */
function chooseFile(els) {
  els.get("file").files = [{ name: "x.wav", size: 10, type: "audio/wav" }];
  els.get("file").dispatch("change", { target: { files: els.get("file").files } });
}

async function main() {
  console.log("frontend_check: app.js under a minimal DOM\n");

  // 1. Every element the script reaches for exists in the page.
  const missing = IDS.filter((id) => !HTML.includes(`id="${id}"`));
  assert(missing.length === 0,
    `index.html declares all ${IDS.length} ids app.js uses` +
    (missing.length ? ` (missing: ${missing.join(", ")})` : ""));

  // 2. Startup probe reports a reachable server.
  {
    const { els } = boot(() => health(true));
    await tick();
    assert(els.get("server-status").textContent.indexOf("server connected") === 0,
      `startup probe says connected: "${els.get("server-status").textContent}"`);
  }

  // 3. Unreachable server is reported, not left as "connecting…" for ever.
  {
    const { els } = boot(() => Promise.reject(new TypeError("Failed to fetch")));
    await tick();
    assert(/cannot reach/.test(els.get("server-status").textContent),
      `unreachable server is stated plainly: "${els.get("server-status").textContent}"`);
  }

  // 4. Success renders, and the overlay clears.
  {
    const { els } = boot((url) => (url === "/health" ? health(true) : {
      ok: true, status: 200, json: async () => payload(),
    }));
    await tick();
    chooseFile(els);
    els.get("go").dispatch("click");
    await tick(); await tick(); await tick();
    assert(els.get("busy").hidden === true, "overlay cleared after a successful request");
    assert(els.get("notation").textContent === "S R G P",
      `notation rendered: "${els.get("notation").textContent}"`);
    assert(els.get("sa-hz").textContent === "262.08 Hz", "Sa rendered");
    assert(els.get("result-panel").hidden === false, "result panel revealed");
  }

  // 5. A request that NEVER SETTLES must not leave the overlay up for ever.
  //    This is the bug the AbortController was added for: a rejected promise is
  //    caught either way, so a test that only simulates rejection proves
  //    nothing. Here fetch() simply never answers, which is what a wedged
  //    socket looks like from the page.
  {
    const shortDeadline = (s) => s.replace("timeoutMs = 300000", "timeoutMs = 120");
    const { els } = boot(
      (url) => (url === "/health" ? health(true) : new Promise(() => {})),
      shortDeadline
    );
    await tick();
    chooseFile(els);
    els.get("go").dispatch("click");
    await tick();
    assert(els.get("busy").hidden === false, "overlay is up while the request hangs");
    await new Promise((r) => setTimeout(r, 400));
    assert(els.get("busy").hidden === true, "overlay cleared once the deadline fires");
    assert(/too long/.test(els.get("error-text").textContent),
      `hung request explained in plain words: "${els.get("error-text").textContent.slice(0, 52)}…"`);
  }

  // 5b. The elapsed counter ticks, so a slow job reads as slow, not as frozen.
  {
    const { els } = boot(
      (url) => (url === "/health" ? health(true) : new Promise(() => {})),
      // Target the busy timer's interval specifically. There are two
      // `}, 1000);` in the file (the record-stopwatch and this timer) and
      // String.replace only touches the first.
      (s) => s.replace('$("busy-text").textContent = `${text || "Working\u2026"} (${s}s)`;\n    }, 1000);',
                       '$("busy-text").textContent = `${text || "Working\u2026"} (${s}s)`;\n    }, 30);')
    );
    await tick();
    chooseFile(els);
    els.get("go").dispatch("click");
    await new Promise((r) => setTimeout(r, 200));
    assert(/\(\d+s\)/.test(els.get("busy-text").textContent),
      `elapsed counter runs: "${els.get("busy-text").textContent}"`);
  }

  // 6. A server error detail is surfaced rather than swallowed.
  {
    const { els } = boot((url) => (url === "/health" ? health(true) : {
      ok: false, status: 422, json: async () => ({ detail: "could not detect Sa" }),
    }));
    await tick();
    chooseFile(els);
    els.get("go").dispatch("click");
    await tick(); await tick(); await tick();
    assert(els.get("error-text").textContent === "could not detect Sa",
      `server error detail shown verbatim: "${els.get("error-text").textContent}"`);
    assert(els.get("busy").hidden === true, "overlay cleared after a server error");
  }

  // 7. Retuning with no cached contour must warn, not hang.
  {
    const { els } = boot((url) => (url === "/health" ? health(true) : {
      ok: true, status: 200, json: async () => payload(),
    }));
    await tick();
    chooseFile(els);
    els.get("go").dispatch("click");
    await tick(); await tick(); await tick();
    els.get("sa-fix").value = "294";
    els.get("apply-sa").dispatch("click");
    await tick(); await tick();
    assert(els.get("busy").hidden === true, "overlay cleared after a retune attempt");
    assert(/re-upload/i.test(els.get("sa-warning").textContent),
      `missing contour is explained: "${els.get("sa-warning").textContent}"`);
  }

  console.log(failures ? `\n${failures} FAILURE(S)` : "\nall frontend checks passed");
  process.exit(failures ? 1 : 0);
}

main().catch((e) => {
  console.error("frontend_check crashed:", e);
  process.exit(1);
});
