# Fretcast

Detect which note — string *and* fret, not just pitch — you're playing on an electric
guitar, from a clean DI signal, in close to real time. Eventually overlaid on an AR view
of the fretboard through a webcam. This repo holds two attempts:

- **V01** (`src/`, Expo/React Native): mobile-first prototype. Camera + MediaPipe hand
  tracking + planar tracking on the fretboard itself. Failed — see
  [Why V01's vision approach was abandoned](#why-v01s-vision-approach-was-abandoned).
  Kept for reference, not under active development.
- **V02** (`core/`): audio-first rebuild, in progress. This README documents V02's
  pipeline end to end.

Full original design rationale: `d:\gta 5 mods\fretcast-v02-plan.md`.

---

## The pipeline, end to end

```
                    ┌─────────────────────────── OFFLINE (training) ───────────────────────────┐
                    │                                                                            │
GuitarSet (mirdata) ─┤                                                                            │
EGDB (Chen et al.)  ─┼──► CQT features (.pt) ──► GuitarTabCRNN (bidirectional) ──► best_model.pth │
   real DI audio     │      + frame labels           trained on GuitarSet,             (fase 1-2) │
   + per-string MIDI │                                fine-tuned on +EGDB                         │
                    │                                       │                                    │
                    │                          warm-start CNN, fresh causal RNN                   │
                    │                                       ▼                                    │
                    │                          GuitarTabCRNN (unidirectional                      │
                    │                          + ~70ms lookahead) ──► best_model.pth (fase 3)     │
                    └────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────── LIVE (inference) ─────────────────────────────┐
                    │                                                                             │
Focusrite (44.1kHz) ─┤ WASAPI       resample     rolling raw-  │  CQT   │ causal   │ per-string  │ ─► note events
  or a WAV file     ─┼─ exclusive ─► 2:1 ──────► audio buffer ─┼───────►│ CRNN ───►│ segmenter   │    (time, string,
                    │  capture     (scipy)       (~450ms,      │        │ (frame-  │ (stateful   │     fret, pitch)
                    │                             hop-aligned) │        │  wise)   │  across      │       │
                    │                                                              │  pushes)    │       ▼
                    └─────────────────────────────────────────────────────────────────────────── PySide6 UI
```

Two CQT computations exist and must never be confused: the **offline** one
(`data_processing/preparation.py`, whole clip at once, used for training data and for
scoring a fixed file) and the **streaming** one (`streaming_infer.py`, recomputed over a
bounded rolling buffer every push). They use identical parameters — same sample rate,
hop length, bin count — so a model trained on one decodes correctly on the other.

---

## The model

`external/trimplexx/python/model/architecture.py`'s `GuitarTabCRNN`, used unmodified:

```
CQT spectrogram (168 bins, 24 bins/octave, fmin=E2/82.41Hz)
        │
        ▼
TabCNN: 5× [Conv2d → BatchNorm → ReLU → MaxPool]   (channels 32→64→128→128→128)
        │
        ▼
GRU (hidden=768, layers=2, dropout=0.5)
   fase 1-2: bidirectional (sees the whole clip, can't run causally)
   fase 3:   unidirectional (causal — only needs the past + a small lookahead)
        │
        ├──► onset_fc  → 6 strings × sigmoid          "did a string just get hit?"
        └──► fret_fc   → 6 strings × 22-way softmax   fret 0-20, or "silence"
```

**Sample rate 22.05kHz, hop length 512 samples (~23ms/frame)** — trimplexx's own
config, kept as-is throughout so every fase's numbers stay comparable.

**String numbering, two conventions, one conversion point.** Fretcast's own code
(`core/fretcast/tab.py`) counts strings 1 (high e) to 6 (low E) — the way tab notation
reads. trimplexx's model counts string index 0 (low E) to 5 (high e) — the way
`OPEN_STRING_PITCHES_MIDI` is indexed. `tab.py`'s `fretcast_string_to_trimplexx` /
`trimplexx_string_to_fretcast` (both literally `6 - x`, the mapping is its own inverse)
is the single conversion point; every script imports from there rather than
re-deriving it.

---

## Fase 1 — baseline (GuitarSet)

`core/scripts/download_guitarset.py` fetches the mic+mix audio subset (skips the
hexaphonic-pickup audio trimplexx's own pipeline never reads — 1.4GB instead of 8.2GB).
`core/scripts/train_baseline.py` runs trimplexx's *own* training pipeline
(`training/pipeline.py`'s `run_training_loop`) unchanged, with their reported best
hyperparameters (`hyperparam_set_v1.json[0]`, "run72": GRU hidden=768, bidirectional,
LR 3e-4, onset/fret combined loss).

**Result:** early-stopped at epoch 92/300 (patience 25). Val TDR F1 ≈ 0.81 — in line
with trimplexx's own reported 0.857 on GuitarSet itself. `run_baseline.py`'s
`transcribe()`/`transcribe_with_model()` wire this checkpoint to Fretcast's own
`TabEvent`/eval harness (`core/fretcast/eval.py`: onset F1, pitch/string/fret/position
accuracy, disambiguation rate).

## Fase 2 — domain adaptation (EGDB)

GuitarSet is recorded via a hexaphonic pickup + room mic on an acoustic-electric —
not the same signal chain as a real DI box into an audio interface. To measure that
gap without waiting on manually-recorded takes, **EGDB** (Chen et al., ISMIR 2022,
*"Towards Automatic Transcription of Polyphonic Electric Guitar Music"*) was used
instead: 240 tabs, one guitarist, hexaphonic pickup, clean DI signal + amp-sim
variants, per-string MIDI ground truth (one MIDI track per string; track index maps
directly onto Fretcast's own string numbering — confirmed empirically against the
fret range, not documented anywhere).

`core/scripts/download_egdb_files.py` fetches it from EGDB's public Google Drive
folder — one file at a time with a short delay, because Drive's bulk
`download_folder()` throttles hard after ~60 files (a real, reproducible limit, not a
config issue). `core/fretcast/egdb.py` parses the label MIDIs (`load_egdb_label` for
eval; `load_egdb_notes_for_training` — with note *duration*, needed for training labels
— pairs note_on/note_off per string track, monophonic per string so a simple two-slot
state machine suffices).

**Held-out set:** ids 100–119 are *never* used for training (see
`prepare_egdb_for_training.py`'s `HELD_OUT_EVAL_IDS`) — every fase-2/3 checkpoint below
is scored against the same 20 tracks, so the numbers are comparable across fases.

**Baseline vs. EGDB (domain gap):**

| metric | GuitarSet-trained baseline | fine-tuned on GuitarSet+EGDB |
|---|---|---|
| onset F1 | 73.0% | 74.3% |
| pitch acc | 53.6% | 61.1% |
| string acc | 45.1% | 57.7% |
| fret acc | 53.0% | 62.8% |
| **position acc** (string+fret) | 38.3% | **51.6%** |
| disambiguation rate | 75.5% | 87.5% |

`prepare_egdb_for_training.py` converts EGDB into trimplexx's own on-disk feature
format (`egdb_<id>_features.pt`/`_labels.pt`, dropped straight into the same
`processed_guitarset_data/{train,validation}/` directories GuitarSet already
populated — no new `Dataset` class needed). `finetune_egdb.py` loads the fase-1
checkpoint as *initialization* (not from scratch) via trimplexx's own
`model_utils.load_best_model`, continues training at a 10× lower LR (3e-5) on the
combined set. `enable_audio_augmentations=False` is required for both splits —
`egdb_<id>` ids don't exist in mirdata's GuitarSet index, so with augmentations on they'd
silently get dropped from training.

Fase 2's own target (≥90% string/fret, ≥97% pitch) isn't hit yet — the fine-tune closed
most of the gap but not all of it. More/better domain data is the obvious next lever if
this needs revisiting; it wasn't blocking, so fase 3 went ahead on top of it.

## Fase 3 — streaming (causal model)

A bidirectional GRU needs the *entire* clip — including audio that hasn't happened yet
in a live performance. `core/scripts/streaming_model.py` retrains it unidirectional
with a fixed, small lookahead (`LOOKAHEAD_FRAMES = 3` → 69.7ms at hop=512/sr=22050,
inside the plan's 50–70ms budget): training labels are shifted forward by that many
frames (`create_frame_level_labels_with_lookahead`), so a causal model reading up to
frame *t* is being asked for the label at frame *t − lookahead* — the lookahead is baked
into training, not bolted on at inference.

Only the CNN transfers cleanly from the fase-2 checkpoint (`rnn_bidirectional` doesn't
affect it at all); the RNN and onset/fret heads have different weight shapes
(bidirectional doubles the heads' input width) and train from a fresh init
(`build_unidirectional_model_warm_started`, `train_streaming.py`).

**Result: no accuracy loss vs. the bidirectional model — slightly better, if anything**
(same held-out ids 100–119):

| metric | finetuned (bidirectional) | streaming (causal + lookahead) |
|---|---|---|
| onset F1 | 74.3% | 74.0% |
| position acc | 51.6% | **56.2%** |
| disambiguation rate | 87.5% | **93.0%** |

### The streaming decoder: a real bug, found and fixed

`streaming_infer.py`'s `StreamingTranscriber` feeds small audio blocks in (`push_audio`)
and must decide, incrementally, when a note has actually ended. The first version
re-ran trimplexx's own `note_conversion_utils.frames_to_notes_for_eval` on the whole
rolling buffer *from scratch* every push — that function force-closes any still-active
note at the last frame it's given. On a growing buffer, "the last frame" is wherever
that particular push happened to stop, not a real note boundary: a sustained note got
"rediscovered" as a brand-new note on the next push, right after the previous push's
artificial cutoff. Symptom: ~2× too many notes, with near-duplicate (string, fret)
pairs a consistent 70–160ms apart — not random jitter, so a time-based dedup /
trailing-margin filter didn't help (tried margins from 0.15s to 0.5s; the confusion
comes from an *already-expired* previous push's forced closure, not the current one's
edge).

**Fix:** `PerStringSegmenter` — an incremental version of the same state machine, but
its "note currently sustaining" state persists *across* `push_audio` calls. A note is
only considered finished on a genuine re-onset, fret change, or silence classification
— never because a particular push's buffer happened to end there. Frames too close to
the buffer's trailing edge (not yet past the model's own lookahead) are simply left
unprocessed until a later push has enough context. Result: note count went from ~2×
over-detected down to matching the offline (whole-file) decode within a few percent —
validated with `validate_streaming.py` against `transcribe_with_model()` on several
EGDB tracks (88–92% streaming-vs-offline onset F1 agreement).

A second, unrelated problem then showed up on longer tracks: recomputing the full CQT +
model forward pass over the whole rolling buffer *every* ~23ms push is wasteful (>95%
of that audio was already processed the previous push) — one track timed out after 2
minutes. Fixed by decoupling audio arrival from recompute cadence
(`recompute_hop_frames`): audio can arrive in whatever tiny blocks the hardware
delivers, but the expensive CQT+model step only runs once enough new audio has piled up
(default: every 4 frames ≈ 93ms). Same track: 29 seconds, slightly *better* agreement.

### Real audio capture — WASAPI, not ASIO

The plan's original assumption was ASIO input. This machine has **no ASIO backend at
all** — `sounddevice.query_hostapis()` lists only MME/DirectSound/WASAPI/WDM-KS, even
with a Focusrite interface plugged in. The standard pip `sounddevice`/PortAudio wheel
doesn't ship ASIO support (Steinberg's ASIO SDK license doesn't permit
redistributing a binary built against it). **WASAPI exclusive mode**
(`sd.WasapiSettings(exclusive=True)`) is the substitute — bypasses the Windows audio
mixer directly, comparable latency to ASIO without a custom PortAudio build.

WASAPI exclusive mode needs the device's own *native* sample rate — the Focusrite
reports 44100Hz (`sd.query_devices()`), not the 22050Hz the model expects.
`live_capture.py` / `core/gui/audio_worker.py` capture at 44100 and downsample 2:1
(`scipy.signal.resample_poly` — exact integer ratio, cheap) before handing audio to
`StreamingTranscriber`.

### Open, not blocking: the CQT window is long for low strings

`librosa.filters.wavelet_lengths` on the current CQT config shows the low-E filter
needs a **420ms** analysis window — on its own, nearly 3× the plan's whole <150ms
latency budget, before model inference or the lookahead are even added:

| open string | frequency | CQT window needed |
|---|---|---|
| low E | 82Hz | 420ms |
| A | 110Hz | 315ms |
| D | 147Hz | 236ms |
| G | 196Hz | 177ms |
| B | 247Hz | 140ms |
| high e | 330Hz | 105ms |

`probe_filter_scale.py` tested whether shrinking `filter_scale` (linearly shortens
every filter's window) fixes this cheaply, *without* retraining: no — at the point the
window reaches ~150ms (filter_scale≈0.35), position accuracy on the fine-tuned
checkpoint collapses from 51.6% to 15.3%. Important caveat this measures: robustness to
a train/test *mismatch* (the model never saw this filter_scale during training), not
the ceiling a matched retrain at that filter_scale would reach — those are different
numbers, and the second one is still untested. A harder limit also showed up: below
filter_scale≈0.3, librosa raises a Nyquist-violation error at the highest CQT bins.
Leading candidate mitigation (not built yet): an asynchronous UI — a fast, cheap
onset/energy-based "flash" the instant a string is hit, with the exact fret number
"clicking in" 40–80ms later once the CQT/model has resolved it. Not blocking further
work; just an open, measured trade-off.

---

## The desktop app (`core/gui/`)

Originally planned as a websocket bridge from this Python backend to the old V01
Expo/RN app (`core/scripts/websocket_server.py` — written, then abandoned mid-build).
Reconsidered: the entire backend is already Python, the eventual camera/ArUco overlay
work fits far more naturally in Python/OpenCV than in React Native, and Dario's other
project (ClearCast) is already a PySide6 desktop app — same tools, no new stack, and no
Electron/Tauri needed for a native Windows+Mac app. `websocket_server.py` is left in
place, unused; nothing currently imports it.

```
core/gui/
  audio_worker.py   QThread: runs the whole capture -> StreamingTranscriber chain,
                    emits a Qt signal (note_detected) per confirmed note. Qt's
                    signal/slot mechanism is already thread-safe across threads --
                    no IPC layer needed now that everything's one process.
                    Two sources: --source live (real WASAPI capture) or
                    --source wav (replays a file with real-time-like pacing --
                    lets the whole app be exercised without a guitar plugged in).
  main_window.py    QMainWindow: status label, start/stop button, a table of
                    confirmed notes (newest on top). Deliberately no camera
                    preview or overlay yet -- validates the plumbing first.
  app.py            Entrypoint.
```

```
python -m core.gui.app --source wav --wav-path D:/FretcastData/egdb/audio_DI/104.wav
python -m core.gui.app --source live
```

Verified end to end (model load → WAV playback with real-time pacing → notes arriving
via Qt signals, matching `validate_streaming.py`'s numbers for the same file) by driving
`AudioWorker` directly under a `QCoreApplication` event loop, headlessly — the agent
that built this can't click a "Start" button or play a guitar, so that's as far as it
verified. **Actually playing into `--source live` and confirming real notes show up is
still open** — needs a human at the interface.

**Not built yet:** camera capture, ArUco marker detection, homography-based fretboard
overlay — fase 5's AR half. Needs a real guitar + webcam to build and calibrate
against, so it waits for that to be available. `guitarNeckDetector.ts`/`homography.ts`/
`fretGrid.ts` in `src/vision/` (V01, TypeScript) have the relevant math already worked
out and would port to Python/OpenCV fairly directly (`cv2.findHomography`,
`cv2.aruco`) if reused.

---

## Why V01's vision approach was abandoned

V01 tried to track the fretboard itself frame-to-frame (planar/homography tracking on
natural features, i.e. the fret lines) plus MediaPipe hand landmarks. It didn't hold up:

- **Repetitive texture.** A fretboard is ~22 near-identical wooden cells with metal
  fret lines — to a tracker, fret 5 looks like fret 7. A couple of dropped/slipped
  frames and the tracked position is off by several frets with no way to self-correct.
- **Heavy occlusion.** The thing being tracked (the frets) is continuously covered by
  the thing that moves (the fretting hand).
- **No fixed reference.** The guitar moves with the player's body; without a rigid
  external anchor, small tilts/breathing/strum motion accumulate as drift a one-time
  calibration can't survive.

V02's planned fix (not yet built, waits on fase 5): **ArUco markers** on the headstock
and body — 2–3 fixed, individually-identifiable markers, pose-solved fresh every frame
with no temporal memory, so drift is structurally impossible and repetitive-texture
confusion doesn't apply (markers are unique by design). `guitarNeckDetector.ts`'s
YOLO-based per-frame corner detection is a markerless fallback already scaffolded in
V01 if ArUco proves impractical, but markers are the primary plan.

---

## Repo layout

```
core/
  fretcast/            pure-stdlib foundation (no install needed)
    tab.py             pitch <-> (string, fret) math, candidate positions,
                        fretcast_string_to_trimplexx / trimplexx_string_to_fretcast
    labels.py          load ground-truth exercises / save-load predictions
    eval.py            scoring: onset F1, pitch/string/fret/position acc, disambig rate
    egdb.py            EGDB MIDI label parser (eval + training-label variants)
  exercises/           scripted Phase-0 takes + ground-truth CSVs + protocol
                       (Dario's own DI recordings -- still not made; EGDB substituted
                       for the domain-gap measurement in the meantime)
  scripts/
    run_baseline.py              model seam: load_model / transcribe_with_model /
                                 transcribe_array_with_model (used by everything below)
    train_baseline.py            fase 1: reproduce trimplexx's GuitarSet baseline
    download_guitarset.py        fase 1: fetch the GuitarSet subset trimplexx needs
    download_egdb_files.py       fase 2: fetch EGDB audio/labels (rate-limit-safe)
    prepare_egdb_for_training.py convert EGDB into trimplexx's on-disk feature format
    finetune_egdb.py             fase 2: continue training on GuitarSet+EGDB
    eval_egdb.py                 domain-gap measurement against EGDB's held-out ids
    probe_filter_scale.py        CQT-window-latency probe
    streaming_model.py           fase 3: unidirectional+lookahead label shift, CNN
                                 warm-start
    train_streaming.py           fase 3: train the causal model
    streaming_infer.py           StreamingTranscriber -- stateful incremental decode
    validate_streaming.py        checks streaming output agrees with offline decode
    live_capture.py              real WASAPI-exclusive audio capture (console-only)
    websocket_server.py          unused -- superseded by core/gui/, see above
  gui/                 PySide6 desktop app (see "The desktop app" above)
  tests/
    test_tab_eval.py   dependency-free self-tests for the math and the eval harness
  requirements.txt     model-side deps (CUDA venv); the eval harness itself needs none
external/
  trimplexx/           the cloned upstream model repo (gitignored, clone it yourself)
src/                   V01, Expo/React Native -- see "Why V01's vision approach was
                       abandoned"; kept for reference, not developed further
```

## Setup

```
# model side (RTX box)
python -m venv .venv && .venv\Scripts\activate
# install the CUDA torch build from pytorch.org, then:
pip install -r core/requirements.txt   # after uncommenting the model deps
git clone https://github.com/trimplexx/music-transcription external/trimplexx

# verify the eval harness needs none of that (pure stdlib):
python core/tests/test_tab_eval.py
python core/scripts/run_baseline.py --exercise ex3_devil_pairs --dry-run   # scores 100%
```

## Roadmap

1. **Fase 0** — scripted eval-harness exercises. Done.
2. **Fase 1** — GuitarSet baseline. Done.
3. **Fase 2** — EGDB domain adaptation. Done (fase-2's own ≥90% target not yet hit).
4. **Fase 3** — causal streaming model + real audio capture. Done; CQT-window latency
   open but not blocking.
5. **PySide6 desktop app**, audio-only. Done; needs a human to actually play into it.
6. **Next** — camera + ArUco + fretboard overlay (fase 5's AR half). Needs a real
   guitar + webcam to build against.
