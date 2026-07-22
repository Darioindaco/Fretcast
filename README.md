# Fretcast

Real-time-ish guitar tab transcription: detect which note/string/fret you're playing
from a clean DI signal, eventually overlaid on an AR view of the fretboard.

## Status

- **V01** (`src/`, Expo/React Native): the original mobile-first attempt. Failed on
  MediaPipe finger-tracking + planar fretboard tracking (repetitive fret-line textures,
  occlusion, no fixed reference — see `core/gui/`'s README below for why V02 dropped
  this route entirely). Kept as reference; not under active development.
- **V02** (`core/`): audio-first rebuild, in progress. A PyTorch CRNN
  (unidirectional GRU + a fixed ~70ms lookahead, so it can run causally frame-by-frame)
  estimates string/fret from a CQT spectrogram of the DI signal, fine-tuned on GuitarSet
  + EGDB (Chen et al., ISMIR 2022 — real DI-recorded electric guitar with per-string
  MIDI ground truth). A PySide6 desktop app (`core/gui/`) captures live audio (WASAPI
  exclusive mode — no ASIO backend available via pip's sounddevice/PortAudio build on
  this machine) and streams detected notes to a live UI. No camera/AR overlay yet —
  that's the next milestone, and needs a real guitar + webcam to build against.

Full architecture rationale: `d:\gta 5 mods\fretcast-v02-plan.md`. Package-level detail:
`core/README.md` (the Python ML/audio side) and `core/gui/__init__.py` (the desktop app).

## Quick start (V02 desktop app)

```
python -m core.gui.app --source wav --wav-path <some.wav>   # no guitar needed, replays a file
python -m core.gui.app --source live                         # real audio via your interface
```

Needs the model-side venv set up — see `core/README.md`.
