# How to build, run and test Audio-to-Swar

Every command and output here was run against a clean clone of
`git@github.com:Hritik-Kumar-dev/swaras.git`.

---

## 1. Prerequisites

- **Python 3.11 or newer.** Check with `python3 --version`.
- Nothing else. Essentia is optional (see step 3).

You do **not** need the repository to use the tool — you can install it
straight from GitHub.

---

## 2. Install

```bash
git clone git@github.com:Hritik-Kumar-dev/swaras.git
cd swaras
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

That gives you three things:

| | |
| --- | --- |
| `swar` | console command, e.g. `swar transcribe song.wav` |
| `python -m swaras` | same thing, works without touching `PATH` |
| `pytest`, `matplotlib` | for the tests and the plots |

Confirm it works:

```bash
swar --help
```

### From a built wheel instead

```bash
pip install swaras                # from a package index
pip install 'swaras[essentia]'     # ...with the default pitch tracker
```

Both `swar` and `python -m swaras` are installed. On a clean box the default
detector will tell you what is missing and stop, rather than crashing:

```
error: Melodia F0 tracking needs Essentia, which is not installed. Either:
  pip install 'swaras[essentia]'   (Linux and macOS)
or use the fallback:  --detector pyin
```

---

## 3. Add Essentia (recommended)

Essentia supplies the default pitch tracker (Melodia) and a second opinion on
Sa. It is a native package with prebuilt wheels for **Linux and macOS only**.

```bash
pip install 'swaras[essentia]'
```

Check it landed:

```bash
python -c "import essentia.standard as es; print(hasattr(es,'PredominantPitchMelodia'))"
# True
```

On Windows, install Essentia inside WSL or a container. Without it everything
still works on librosa pYIN — just pass `--detector pyin`.

> Essentia prints one `[ INFO ] MusicExtractorSVM ...` line to **stderr** on
> import. Harmless: `stdout` stays clean, so `--plain` still pipes to a file
> without it.

---

## 4. Get the sample audio

`samples/` is not in the repository — two of the files are synthesised and
three are LibriSpeech excerpts fetched from the network. Rebuild them with:

```bash
python scripts/fetch_samples.py
```

```
wrote samples/sweep.wav
wrote samples/phrase_c4.wav
...
5 files in samples
```

Nothing in the test suite needs these. They are only used by the accuracy
reports in step 8, which skip whatever is missing.

**To use your own audio, skip this step entirely.**

---

## 5. Run the tests

```bash
pytest
```

```
354 passed, 1 warning in 8.94s
```

About 9 seconds, fully offline and deterministic. The one warning is a
third-party deprecation notice from Starlette about `httpx`, not from this code.

Useful subsets:

```bash
pytest tests/test_tonic.py -v        # one module
pytest -k shruti                     # by name
pytest -x -q                          # stop at the first failure
```

---

## 6. Transcribe a file

The bundled sample is a synthesised Sa-Re-Ga-Pa-Dha-Ni at C4, twice, so you
know exactly what the right answer is.

```bash
swar transcribe samples/phrase_c4.wav
```

```
Sa      : 262.08 Hz (histogram, confidence 1.00)
Notes   : 12
Notation: S R G P D N S R G P D N
Alts    : 294.0 Hz (conf 0.95), 329.8 Hz (conf 0.91)
Warning : essentia suggested 99.2 Hz, which does not match the histogram ranking (best 262.1 Hz)
Counts  : Sa x2, Re x2, Ga x2, Pa x2, Dha x2, Ni x2
Shruti  : mean |dev| 7.24c, worst 10.98c, 8 unclear
```

**That notation is correct.** The file really is Sa-Re-Ga-Pa-Dha-Ni twice at
261.63 Hz. This is the single best check that the whole pipeline works.

For scripting, take only the notation:

```bash
swar transcribe samples/phrase_c4.wav --plain
# S R G P D N S R G P D N
```

### The options

| Flag | What it does |
| --- | --- |
| `--plain` | Only the notation line. Use this in scripts. |
| `--sa 261.63` | Supply Sa; skips detection entirely. |
| `--detector pyin` | Use librosa pYIN instead of Melodia. |
| `--detailed` | Per-note timing, cents, deviation from nominal. |
| `--shruti` | The 22-shruti layer: nearest shruti and cents off it. |
| `--json` | Everything, for `jq` or another program. |
| `--out FILE` | Write to a file instead of the terminal. |
| `--help` | Full list. |

### The 22-shruti view

```bash
swar transcribe samples/phrase_c4.wav --shruti --detailed
```

```
  #  shruti   swar       var          dev  clarity  conf   start
  1  S0       Sa         shuddha     +2.0     45.0  0.96   0.046
  2  R4h      Re         high        -7.0     11.0  0.37   0.464
  3  G8h      Ga         high       -11.0     11.0  0.00   0.975
```

`dev` is the signed distance from the shruti: how far off the position the
note was actually sung. `conf` is 1.0 on a shruti and 0.0 exactly halfway to
the next. Note `G8h` at 0.00 — that Ga is genuinely ambiguous, sitting
exactly between the two Ga shrutis at 386 and 408 cents. The low confidence is
the honest answer, not a failure.

### Detect Sa on its own

```bash
swar tonic samples/phrase_c4.wav
```

```
Sa = 262.08 Hz  (confidence 1.00, method histogram)
  evidence: sa 0.88  pa 1.00  octave 0.00  methods histogram
  alt:   294.00 Hz  confidence 0.95  (histogram)
  alt:   329.82 Hz  confidence 0.91  (histogram)
  essentia raw: 99.24 Hz
  warning: essentia suggested 99.2 Hz, which does not match the histogram ranking
```

Use this when you suspect the Sa is wrong — the alternatives are the wrong
readings to consider, and the `evidence` line shows how the winner was chosen.

---

## 7. Use the web UI

```bash
swar serve
```

```
Serving the web UI on http://127.0.0.1:8000/
```

Open <http://127.0.0.1:8000/>. Then:

1. **Upload** a file, or **Record** from your microphone.
2. Press **Transcribe**. The notation appears with a **Copy** button.
3. **Check Sa.** The alternatives are buttons — click one to re-transcribe
   instantly, or type your own value and press **Re-transcribe**.
4. Toggle the **22-shruti** line and the **per-note table** as needed.

### Why correcting Sa is fast

The first request tracks pitch once and caches the result. Every correction
re-runs only what depends on Sa:

```
POST /transcribe   0.060 s
POST /retune       0.004 s
```

Options: `swar serve --port 9000`, `--host 0.0.0.0`, `--reload`.

> **Microphone needs a secure context.** `getUserMedia` only works on
> `https://` or `localhost`. From another machine you need TLS; the Upload tab
> works regardless.

---

## 8. Check the accuracy yourself

Four scripts print the measured numbers. Run from the repo root with `samples/`
present.

```bash
python -m tests.compare_trackers    # Stage 1: F0 error against known sweeps
python -m tests.tonic_accuracy      # Stage 2: Sa recovery
python -m tests.segment_accuracy    # Stages 3-4: round trip
python -m tests.shruti_accuracy     # Stage 5: the shruti grid
```

`tonic_accuracy` expects roughly:

```
  Sa Re Ga Pa                -1.0       -1.0       -1.0
  ascending scale            -1.0       -1.0       -1.0
  within 15 cents: 15/15 (100%)   median |error| 1.0c   max 1.0c
```

`segment_accuracy` should report **exact** for all five phrase shapes at all
three tonics with Sa supplied.

---

## 9. See the pitch contour

Plots are a separate entry point, `python -m swaras.plot`, not a flag on
`swar`. Useful for checking *why* a transcription looks wrong.

```bash
python -m swaras.plot samples/phrase_c4.wav --out out/contour.png
python -m swaras.plot samples/phrase_c4.wav --histogram --out out/sa.png
python -m swaras.plot samples/phrase_c4.wav --shruti-plot --out out/grid.png
```

| Flag | Draws |
| --- | --- |
| *(none)* | F0 contour in Hz, both detectors overlaid |
| `--sa 261.63` | Same, with the y-axis in cents above Sa |
| `--histogram` | Sa candidates over the cents histogram, with the Pa peak marked |
| `--shruti-plot` | Each note's measured pitch against the 22-shruti lines |
| `--detectors pyin` | Pick the tracker |

`--shruti-plot` is the most informative when a note looks wrong: a dot sitting
on a grid line is a confident match, one floating between two lines is an
ambiguous one, and the connector's length is the deviation.

---

## 10. Use it from Python

```python
from swaras.pipeline import transcribe_file

result = transcribe_file("samples/phrase_c4.wav")

print(result.transcription.text)          # S R G P D N S R G P D N
print(result.transcription.sa_hz)         # 262.08
print(result.transcription.shruti_text)   # S0 +2c R4h -7c G8h -11c ...
print(result.segmentation.diagnostics)    # counters, for debugging
```

### Supplying Sa, and reading the notes

```python
from swaras.pipeline import transcribe_file

r = transcribe_file("samples/phrase_c4.wav", sa_hz=261.63)
for note in r.transcription.notes[:3]:
    print(note.label, round(note.absolute_cents, 1), note.deviation_cents)
# S 5.0 5.0
# R 200.0 7.0
# G 400.0 3.0
```

Note the deviation is about 5 cents here, where the default detection gave 2.
Supplying `sa_hz=261.63` when the recording is really at 262.08 shifts every
note, which is exactly why `swar tonic` is worth running before you override
anything.

### Using the API from Python

```python
import requests

with open("samples/phrase_c4.wav", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/transcribe",
                      files={"file": f}, data={"detector": "melodia"})
data = r.json()
print(data["notation"])

# Correct Sa without re-uploading.
r2 = requests.post("http://127.0.0.1:8000/retune",
                   json={"contour_id": data["contour_id"], "sa_hz": 294.0})
print(r2.json()["notation"])
```

| Route | Purpose |
| --- | --- |
| `GET /` | the UI |
| `POST /transcribe` | multipart `file`, optional `sa_hz`, `detector` |
| `POST /retune` | JSON `{contour_id, sa_hz}` |
| `GET /health` | liveness, and which detectors are available |
| `DELETE /cache` | drop cached contours |

Interactive API docs are at <http://127.0.0.1:8000/docs> while `swar serve` runs.

---

## 11. Build a distributable package

```bash
pip install build
python -m build
```

```
Successfully built swaras-0.1.0.tar.gz and swaras-0.1.0-py3-none-any.whl
```

In `dist/`. Test the wheel in a throwaway environment before publishing:

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install 'dist/swaras-0.1.0-py3-none-any.whl[essentia]'
/tmp/check/bin/swar transcribe samples/phrase_c4.wav --plain
# S R G P D N S R G P D N
```

`dist/` is gitignored — a checked-in wheel goes stale the moment the source
changes.

---

## 12. If something is wrong

| Symptom | Cause and fix |
| --- | --- |
| `error: ... needs Essentia` | `pip install 'swaras[essentia]'`, or use `--detector pyin` |
| `no voiced pitch detected` | Silent input, or pitch outside 65–1100 Hz (roughly C2–C6) |
| `could not detect Sa` | Too little pitched material. Supply `--sa`. |
| `could not decode audio` | Unsupported container. Convert to wav: `ffmpeg -i in.m4a out.wav` |
| Notation looks like a different key | Sa was misdetected. `swar tonic FILE`, then re-run with `--sa HZ`. |
| One note split into several | Vibrato deeper than ~80 cents, or a slow meend. See "Known limitations" in the README. |
| Escaped `&` or wrong characters in the UI | Serve the page, don't open `index.html` from disk — it calls the API. |
| `getUserMedia` blocked | Microphone needs `https://` or `localhost`. |

### Sanity check

If `swar transcribe samples/phrase_c4.wav --plain` does not print exactly:

```
S R G P D N S R G P D N
```

something is wrong with the install, not with your audio. Work through steps 2
to 5.

---

## 13. When you add your own audio

What works well: a single unaccompanied voice or instrument, a few seconds to a
minute, pitch roughly C2 to C6, little reverb, no accompaniment.

What does not yet: a voice over instruments, a full film song, or a recording
with a second singer. The pipeline assumes one melodic line, which is the
insertion point for source separation — see "Extending the pipeline" in the
README.

Start by running `swar tonic` on the file before transcribing. If Sa looks
wrong, fix it with `--sa` and you will get a far better transcription than
fighting the detector.
