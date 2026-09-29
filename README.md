# Audio-to-Swar

Transcribe a recording of a single melodic voice or instrument as Indian
classical swar notation. Pitch identity only: no lyrics, no rhythm, no raga
detection. The output is tonic-relative, so Sa is always detected or supplied,
never hardcoded to a fixed frequency.

```
Ni Sa Ma Ma Pa Dha Ni
Pa Ni Sa Ga
```

## Status

All six stages are done. A recording transcribes end to end into swar
notation, from the command line or a browser.

| Stage | Module | Status |
| --- | --- | --- |
| 1 | `audio_io`, `pitch` | done |
| 2 | `tonic` | done |
| 3 | `segment`, `swar`, `format`, `cli` | done |
| 4 | octave + komal/tivra marking | done (covered by Level A) |
| 5 | `shruti` (22-shruti layer) | done |
| 6 | `api`, `web` | done |

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Essentia is optional but recommended. It is a native dependency and only ships
prebuilt wheels for Linux and macOS; on Windows use WSL or Docker. Without it
the pipeline falls back to librosa pYIN and a NumPy tonic detector.

```bash
pip install essentia   # TonicIndianArtMusic + PredominantPitchMelodia
```

### Why two F0 trackers

| Tracker | Backend | Good at |
| --- | --- | --- |
| `melodia` (default) | Essentia | Polyphonic audio, real songs, a voice over accompaniment |
| `pyin` | librosa | Dry solo voice; sharper on a clean single line |

Both implement the same `PitchDetector` interface, so either can be swapped for
CREPE later without touching downstream stages.

## Pipeline

```
audio ─▶ audio_io ─▶ pitch ─▶ normalize ─▶ segment ─▶ swar ─▶ format ─▶ text/JSON
                        │         ▲
                        └▶ tonic ─┘   (Sa is needed before cents can be computed)
```

Work left to right: decode to mono 22.05 kHz and trim silence, estimate a
continuous F0 contour, detect Sa, express every frame as cents relative to Sa,
group the contour into stable notes, then map notes to swaras. Nothing is
quantised per frame; the contour stays continuous until `segment` needs to
decide where one note ends and the next begins.

## Stage 1 usage

Plot the pitch contour of a file:

```bash
python -m swaras.plot samples/libri3.wav --out contour.png
```

Compare both trackers on the same file, with the y-axis in cents above Sa:

```bash
python -m swaras.plot samples/phrase_c4.wav --sa 261.63 --out compare.png
```

Sa is detected automatically and drawn on the axis. Add `--histogram` to see
the cents histogram with the Sa candidates marked:

```bash
python -m swaras.plot samples/phrase_c4.wav --histogram --out sa.png
```

Override Sa with `--sa 261.63`, or choose the detector for tonic detection with
`--detector pyin`.

Run the accuracy comparisons:

```bash
python -m tests.compare_trackers    # F0 trackers
python -m tests.tonic_accuracy      # Sa detection
```

Or drive the stages from Python:

```python
from swaras.audio_io import load_file
from swaras.pitch import get_detector

audio = load_file("samples/libri3.wav")
contour = get_detector("melodia").estimate(audio.samples, audio.sample_rate)

print(contour.n_frames, "frames")
print(contour.voiced_fraction, "voiced")
print(contour.voiced_f0()[:5])
```

### Stage 1 accuracy

Measured on the synthetic sweep 80 → 900 Hz, where the true F0 is known:

| Tracker | median error | p90 | voiced |
| --- | --- | --- | --- |
| `melodia` | 16.2 cents | 20.3 cents | 99.2% |
| `pyin` | 11.9 cents | 16.8 cents | 100% |

On steady sine tones across the singing range, Melodia tracks within 0.1 cents.
pYIN has a systematic bias, not noise: about +9.2 cents below 250 Hz, +4.2
cents at 261 Hz, and about -0.8 cents above 293 Hz. For tonic detection and
22-shruti deviation, where a 9-cent bias is a meaningful fraction of a shruti,
Melodia's accuracy is the deciding factor and it is the default.

On the three real recordings, the two trackers agree within 30 cents for
60–100% of co-voiced frames (median disagreement 0.8–21 cents). They differ
mainly in how much of the file each considers voiced: on `libri1`, pYIN marks
66% of frames voiced against Melodia's 33%. pYIN is more willing to assign a
pitch in breathy or noisy speech.

## Tonic detection (Stage 2)

Sa cannot be read off the audio: nothing in the recording distinguishes "this
is Sa" from "this is Pa, and the tonic was below". So the stage returns a
ranked list with confidences and never claims certainty, and the UI is meant to
let you correct Sa and re-run.

Two methods feed the ranking:

- **Cents histogram** (always available). The voiced F0 is binned at 1 cent,
  smoothed, and every candidate Sa in 55–800 Hz is scored on four terms: the
  peak at Sa, the peak at Pa (+702c), the peaks at the octave above *and*
  below, and a lowest-note prior. A dense scan of the whole range is scored,
  not just histogram peaks, so a valid Sa sitting in a trough between two
  strong peaks is not missed.
- **Essentia `TonicIndianArtMusic`** (optional). Accurate on real music with a
  tanpura-like drone. On a bare melodic line it clamps to the bottom of its
  search range, so its answer is treated as corroboration: agreement boosts a
  candidate, disagreement is reported as a warning and never wins.

The octave term sums peaks above *and* below, which makes the score symmetric
under octave displacement on purpose. Sa and Sa' then score near-identically,
and `collapse_octave_equivalents` resolves the tie in favour of the lower one,
keeping the upper visible as an alternative. That is the single most common
tonic error, so it is handled explicitly rather than left to chance.

### Stage 2 accuracy

15 synthetic phrases (5 shapes × 3 tonics at 220 / 261.63 / 293.66 Hz),
tolerance 15 cents:

| Detector | within 15 cents | median abs error | max |
| --- | --- | --- | --- |
| `pyin` | 15/15 (100%) | 1.0 cents | 1.0 cents |
| `melodia` | 15/15 (100%) | 0.0 cents | 1.0 cents |

Also verified across a C3–B5 transposition sweep (12 tonics, Sa-Re-Ga-Pa):
all within 1.1 cents.

On the synthetic `phrase_c4` (Sa = 261.63 Hz) the detector returns 262.08 Hz,
with 294.0 and 329.8 Hz as the alternatives — Re and Ga, which are the plausible
wrong readings. Essentia's 99.2 Hz answer is reported as a warning, since the
recording has no drone.

## Transcribing (Stages 3 and 4)

```bash
python -m swaras transcribe recording.wav
```

```
Sa      : 262.08 Hz (histogram, confidence 1.00)
Notes   : 12
Notation: S R G P D N S R G P D N
Counts  : Sa x2, Re x2, Ga x2, Pa x2, Dha x2, Ni x2
```

| Flag | Effect |
| --- | --- |
| `--sa 261.63` | Supply Sa and skip detection entirely |
| `--detailed` | Per-note timing, cents and deviation from nominal |
| `--shruti` | The 22-shruti layer: nearest shruti and cents deviation |
| `--json` | The full result, ready for `jq` |
| `--plain` | Only the notation line |
| `--out FILE` | Write to a file |
| `--detector pyin` | Use librosa pYIN instead of Melodia |

There is also `python -m swaras tonic recording.wav`, which reports Sa and its
candidates without transcribing.

Octave is carried by a notation marker rather than a word: a dot below (`Ni.`)
for mandra, a tick above (`Sa'`) for taar, and a separate integer field in the
JSON. Komal and tivra are distinct positions, so `Re komal`, `Re` and
`Ma tivra` all appear in the output.

## The web UI and API (Stage 6)

```bash
python -m swaras serve          # http://127.0.0.1:8000/
```

Upload a file or record from the microphone, then read the notation, copy it,
and correct Sa if the detection was wrong.

### Correcting Sa is instant

Tonic errors are usually octave errors, so the user will correct Sa often.
That is why the API has two endpoints:

| Endpoint | Does | Cost |
| --- | --- | --- |
| `POST /transcribe` | decode, track pitch, detect Sa, segment | the full pipeline |
| `POST /retune` | re-runs everything *after* pitch tracking | milliseconds |

`/transcribe` returns a `contour_id` alongside the result and caches the pitch
contour against it. Clicking an alternative Sa, or typing a corrected one, posts
only the id and the new frequency:

```
POST /transcribe   0.060 s   S R G P D N S R G P D N     Sa 262.08 Hz
POST /retune       0.004 s   n. S R M P D n. S R M P D   Sa 294.00 Hz
```

Fifteen times faster is not the point; the point is that the correction does
not re-run the pitch tracker, which is the expensive part and would be
unchanged by a different Sa. The cache is an LRU of 8 contours keyed on a hash
of the upload bytes, so the same file uploaded twice hits it and a different
file can never collide onto someone else's contour.

### Endpoints

| Route | Purpose |
| --- | --- |
| `GET /` | the single-page UI |
| `POST /transcribe` | multipart `file`, optional `sa_hz`, `detector` |
| `POST /retune` | JSON `{contour_id, sa_hz}` |
| `GET /health` | liveness, and which detectors are available |
| `DELETE /cache` | drop every cached contour |

Uploads are capped at 50 MB and rejected with a 413 rather than being read into
memory. Undecodable audio returns 422 with the decoder's message; a missing
optional backend returns 503, not 500, so the UI can tell "try the other
detector" apart from "your file is broken".

### In the UI

Sa is shown with its confidence and its alternatives as buttons, each showing
its own confidence. Correcting Sa by clicking clears the stale alternatives,
since after a manual correction they are no longer meaningful. The 22-shruti
line and a per-note table (start, duration, cents, deviation, matched shruti,
shruti deviation, confidence) are behind two toggles. If the contour has
expired the UI says so and asks for a re-upload rather than failing silently.

The test suite cross-checks that every element id `app.js` reaches for exists
in `index.html` and that every URL it posts to is a real route, which catches
the likeliest front-end bug without needing a browser in the loop.

## The 22-shruti layer (Stage 5)

```bash
python -m swaras transcribe recording.wav --shruti --detailed
```

```
  #  shruti   swar       var          dev  clarity  conf   start
  1  S0       Sa         shuddha     +2.0     45.0  0.96   0.046
  2  R4h      Re         high        -7.0     11.0  0.37   0.464
  3  G8h      Ga         high       -11.0     11.0  0.00   0.975
```

The shruti layer is not a second set of labels. A shruti is a pitch position
and a performance sits *near* one, so every result carries the signed distance
from the position it matched. That is the honest answer, and it is why this
stage exists alongside `swar.py` rather than replacing it: `swar` answers "which
of the 12 positions", `shruti` answers "how finely can you place it, and how
far off are you".

**Every match carries a confidence**, and it is not decoration. It is 1.0 on a
shruti and falls to 0.0 exactly halfway to the next one, so a note sitting in
the middle of a 70-cent gap is reported as an uncertain choice:

| sung | shruti | dev | conf | why |
| --- | --- | --- | --- | --- |
| 0 | `S0` | +0.0 | 1.00 | exactly on Sa |
| +8 | `S0` | +8.0 | 0.82 | 8 cents sharp |
| -8 | `S0` | -8.0 | 0.82 | 8 cents flat |
| 100 | `R1l` | +10.0 | 0.09 | between the two Re komal shrutis (90/112) |
| 397 | `G7l` | +11.0 | 0.00 | exactly midway between the two Ga shrutis (386/408) |
| 702 | `P13` | +0.0 | 1.00 | exactly on Pa |

The `clarity` column is the distance at which the match would flip. It is
half the *smaller* neighbouring gap, because that is the boundary that arrives
first: Ga-high sits 22 cents above Ga-low but 90 below Ma, so a note 11 cents
below it is already ambiguous while one 45 below is not. Sa and Pa, which are
singletons flanked by 90-cent gaps, have a clarity of 45.

### Seeing it

```bash
python -m swaras.plot recording.wav --shruti-plot --out out/notes.png
```

draws the measured pitch of every note against the 22 grid lines, with a
connector down to the shruti it matched so the connector's length *is* the
deviation, and colour for the confidence. On `phrase_c4` the Ga notes come out
dark red and sit exactly on the midpoint between the 386 and 408 lines, which
is what an ambiguous match should look like.

### What it shows on real audio

`phrase_c4` was synthesised at 0 / 200 / 400 / 700 / 900 / 1100 cents, that is
at equal temperament, and the table is just intonation. The layer reports the
difference rather than hiding it:

```
S0 +2c  R4h -7c  G8h -11c  P13 -5c  D17h -9c  N20l +9c
```

mean 7.2 cents, which is essentially the just-versus-equal-temperament
difference for those degrees. On the three speech recordings the mean absolute
deviation is 7 to 12 cents with one outlier at 25, but those have no musical
tonic, so the numbers mean nothing there beyond tracking stability.

### Round-trip accuracy

Synthesise a known sequence, transcribe it, compare. 5 phrase shapes x 3 tonics
(220 / 261.63 / 293.66 Hz), Sa supplied so this measures segmentation and
mapping alone:

| Detector | exact |
| --- | --- |
| `pyin` | 15/15 |
| `melodia` | 15/15 |

Octave marking across three registers is exact: `S. P. S P S' N'`.

With Sa auto-detected, the three phrases whose lowest note is Sa still come out
exact. The two that dip below Sa ("with mandra", "komal and tivra") drop to
about 50%, because tonic detection reads the lowest note as Sa and the octave
reference moves with it. That is the octave ambiguity the UI's Sa correction
exists to resolve, not a segmentation fault.

On the bundled `phrase_c4` sample (true Sa = 261.63 Hz) the pipeline returns
`S R G P D N S R G P D N` against the synthesized `Sa Re Ga Pa Dha Ni` twice
over, with per-note deviations of 0 to +7 cents.

## Design notes

- **Nothing is quantised per frame.** The contour stays as continuous float
  cents until Stage 3 decides note boundaries.
- **Frame length is 1024 samples** (46 ms), chosen by measurement. At 2048
  samples pYIN showed 26 cents of lag on a fast glissando versus 11 cents at
  1024, while 1024 still contains two full periods of the lowest supported
  pitch (65 Hz needs 679 samples).
- **Octave errors are corrected, not snapped.** `correct_octave_jumps` removes
  a near-integer multiple of 1200 cents when the residual is small, which
  fixes sub-harmonic errors while leaving real octave leaps intact.
- **The contour keeps its raw track.** `PitchContour.f0` is what the detector
  produced; smoothing and octave correction are opt-in via
  `smoothed_f0()`, so later stages can choose their own filtering.
- **Reported Sa alternatives are forced apart.** The maximum of a broad
  histogram peak moves continuously with the audio, so the raw top 3 landed
  within a few cents of each other (79.0, 74.0, 72.2 Hz on `libri1`) — three
  readings of one peak, which is useless to a user choosing a correction.
  Candidates within 45 cents of a stronger one are now suppressed.
- **Melodia's pitch-continuity threshold is loosened to 100 cents.** Essentia's
  default of 27.6 cents is tighter than vibrato and than an ordinary melodic
  leap; on a Sa-Pa-Sa'-Ni' phrase at 220 Hz it left only 19% of frames voiced
  and the whole phrase was discarded. A continuous sweep is unaffected.
- **Segmentation is greedy against a running median, not a threshold on
  frame-to-frame change.** A step threshold cannot find a slow meend: a
  Sa-to-Pa glide over 30 frames moves about 24 cents per frame, never crossing
  a 60-cent threshold, so the whole phrase stayed one region, was unstable, and
  was discarded, yielding *no* notes at all. Growing regions while frames stay
  within tolerance of the region's own median handles a step, a meend and
  vibrato with one rule, and the median is what collapses andolan to the centre.
- **The pre-filter width is set by the vibrato rate.** At a 43 Hz frame rate a
  5 Hz vibrato has a 4.3-frame half-period, so the previous 5-frame median could
  not touch it and a 25-cent vibrato fragmented a phrase into 16 notes instead
  of 12. A width of 9 covers a full period; anything from 7 to 15 gives the
  same answer.
- **A meend is recognised by shape, not magnitude.** A ramp gets chopped into
  slices that are each within 40 cents of their own median, so no magnitude
  test separates them from short notes. The distance a slice's values cover,
  compared with the range they span, does: a drift travels one way and covers
  its range once, while a note with vibrato covers its range several times
  over. This has to be measured on the raw contour, because the pre-filter
  flattens a slow ramp into something that looks settled.

## Tuning table

`src/swaras/config/shruti_table.json` holds the tuning data, so the musicology
is data rather than code. It currently carries the 22-shruti just-intonation
scale and the 12 swara positions derived from it. See
[docs/shruti_table.md](docs/shruti_table.md) for the mapping and the
assumptions behind it.

## Tests

```bash
python -m pytest
```

354 tests, all passing. The suite is offline and deterministic: every fixture
is generated by `tests/synthetic.py`, so the suite needs no audio files at all.

The audio under `samples/` is **not committed**. Two of those files are
synthesised and three are LibriSpeech excerpts fetched over the network, which
should not sit in the repository. Rebuild them with:

```bash
python scripts/fetch_samples.py
```

No test depends on them; the accuracy report scripts skip any that are
missing, and the tables quoted above are reproducible without them.

| File | Covers |
| --- | --- |
| `test_audio_io.py` | mono/stereo decode, resampling, peak normalisation, silence trimming, WAV/OGG, error paths |
| `test_cents.py` | cents arithmetic, octave folding with index tracking |
| `test_pitch.py` | pure helpers, pYIN and Melodia on synthetic audio, frame-grid agreement, detector registry |
| `test_pitch_accuracy.py` | measured error against known F0 across the singing range, vibrato, glides |
| `test_tonic.py` | histogram construction, scoring, octave collapse, the 15-cent requirement across tonics and phrase shapes, Essentia integration, manual override |
| `test_tuning.py` | the 22-shruti table contents, the shruti-to-swara mapping, and loader validation |
| `test_segment.py` | normalization, all 12 swara positions, octave markers, segmentation (meends, vibrato, blips, rests, octave spikes), the round trip, output formatting |
| `test_shruti.py` | all 22 shrutis reachable and self-matching, signed deviation, the confidence V-curve, clarity margins, octave handling, output |
| `test_api.py` | both endpoints, the contour cache (LRU, TTL, content keying), upload limits, error codes, and the front-end wiring between `app.js` and `index.html` |

Three measurement scripts print the accuracy tables quoted above:

```bash
python -m tests.compare_trackers     # Stage 1, F0
python -m tests.tonic_accuracy       # Stage 2, Sa
python -m tests.segment_accuracy     # Stages 3-4, round trip
python -m tests.shruti_accuracy     # Stage 5, the shruti layer
```

## Known limitations

- **Single voice, dry recording.** Stage 1 tracks one melodic line. A voice
  over accompaniment or in a full song will confuse both trackers; a
  source-separation step belongs before `pitch` (see below).
- **F0 range is capped** at 65–1100 Hz (roughly C2 to C6), so a very low male
  Sa or a very high Sa' is out of reach.
- **pYIN bias** of up to +9 cents below 250 Hz is not corrected by default. It
  is documented in `PitchParams.pyin_bias_correction_cents` and left at 0.0
  rather than guessed at.
- **Tonic detection is only as good as the material.** The histogram method
  assumes the melody outlines the tonic, which a khayal does and free
  improvisation may not. On the three real speech recordings it returns a
  confident but musically meaningless answer, because spoken English has no
  tonic at all. Treat the confidence as "the histogram agreed with itself",
  not as a musical guarantee, and expect to use the manual override.
- **Essentia's tonic algorithm is unvalidated here.** It needs a tanpura-like
  drone to work, and no such recording is in the test set, so its agreement
  path is exercised only on synthetic audio where it disagrees. Worth
  re-measuring on real khayal before relying on it.
- **A glide longer than about a second is not recognised as one.** Meends up to
  40 frames (0.9 s) transcribe correctly as a transition between the two
  notes. Beyond that the drift slices are long enough that the pre-filter
  flattens them, and they come out as spurious notes. A 1.4 s portamento is
  unusual between notes, so this is unlikely to bite, but it is a real limit.
- **Vibrato deeper than about 80 cents fragments a note.** Region growth allows
  a frame to wander 40 cents from the running median, and a singer rocking a
  note by more than that between cycles splits it. Tests cover up to 70 cents.
- **The 22-shruti layer, the API and the web UI are not built yet.** Stages 5
  and 6 remain.

## Extending the pipeline

The design leaves one deliberate insertion point. A vocal-separation step
should run between `audio_io` and `pitch`, so that the object handed to the
detectors is a single voice:

```
audio ─▶ audio_io ─▶ [separation, later] ─▶ pitch ─▶ normalize ─▶ segment ─▶ ...
```

`PitchDetector` is the only interface the pitch stage depends on, so swapping
in CREPE is a subclass with one `estimate` method and no other changes.
