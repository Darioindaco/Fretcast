# Fretcast V02 — core

Audio-first guitar tab transcription. The `src/` Expo app (V01) becomes the overlay
frontend; this Python package is the new brain. Full rationale in
`d:\gta 5 mods\fretcast-v02-plan.md`.

## Layout

```
core/
  fretcast/          pure-stdlib foundation (runs anywhere, no install)
    tab.py           pitch <-> (string, fret) math + candidate positions
    labels.py        load ground-truth exercises / save-load predictions
    eval.py          scoring: pitch / string / fret / position / disambiguation
    egdb.py          EGDB (Chen et al.) MIDI label parser -- eval + training-label variants
  exercises/         scripted Phase 0 takes + ground-truth CSVs + protocol
  scripts/
    run_baseline.py          model seam (transcribe/transcribe_with_model/load_model)
    train_baseline.py        fase 1: reproduce trimplexx's GuitarSet baseline
    download_guitarset.py    fase 1: fetch the GuitarSet subset trimplexx needs
    download_egdb_files.py   fase 2: fetch EGDB audio/labels (rate-limit-safe, one at a time)
    prepare_egdb_for_training.py  convert EGDB into trimplexx's on-disk feature format
    finetune_egdb.py         fase 2: continue training on GuitarSet+EGDB
    eval_egdb.py             domain-gap measurement against EGDB's held-out ids
    probe_filter_scale.py    CQT-window-latency probe (see the plan doc, section 1)
    streaming_model.py       fase 3: unidirectional+lookahead label shift, CNN warm-start
    train_streaming.py       fase 3: train the causal model
    streaming_infer.py       StreamingTranscriber -- stateful incremental decode
    validate_streaming.py    checks streaming output agrees with offline whole-file decode
    live_capture.py          real WASAPI-exclusive audio capture (console-only)
  gui/               PySide6 desktop app -- see gui/__init__.py
  tests/
    test_tab_eval.py self-tests for the math and the harness
  requirements.txt   model-side deps (CUDA venv); the harness needs none
```

## Verify the harness (no model, no audio needed)

```
python core/tests/test_tab_eval.py            # math + eval self-tests
python core/scripts/run_baseline.py --exercise ex3_devil_pairs --dry-run
```

The dry run scores a perfect prediction — every metric should read 100%.

## Roadmap

1. **Fase 0** — scripted eval-harness exercises (`exercises/phase0_protocol.md`). Done.
2. **Fase 1** — baseline: cloned `trimplexx/music-transcription`, trained its reported
   best config on GuitarSet (val TDR F1 ~0.81, in line with their own 0.857). Done.
3. **Fase 2** — domain adaptation: measured the gap against EGDB (Chen et al.) DI audio
   (baseline 38.3% position acc), fine-tuned on GuitarSet+EGDB combined (51.6%). Done,
   though fase 2's own target (>=90% string/fret) isn't hit yet -- more/better data is
   the obvious next lever if this needs revisiting.
4. **Fase 3** — streaming: unidirectional + ~70ms-lookahead retrain (no accuracy loss
   vs. the bidirectional model, 56.2% position acc on the same held-out set), a
   stateful incremental decoder (`streaming_infer.py`, fixed a real duplicate-note bug
   from naively re-segmenting a growing buffer every push), and real WASAPI-exclusive
   audio capture. Model + streaming plumbing done; the CQT-window/latency question
   (`probe_filter_scale.py`) is open but not blocking.
5. **PySide6 desktop app** (`core/gui/`) — audio-only live note viewer wired up. Done;
   needs real playing to validate actual recognition (the agent can't play guitar).
6. **Next**: camera + ArUco marker detection + fretboard overlay (needs a real guitar
   + webcam to build against) -- fase 5 of the original plan.

## Model-side setup (run on the RTX box)

```
python -m venv .venv && .venv\Scripts\activate
# install the CUDA torch build from pytorch.org, then:
pip install -r core/requirements.txt   # after uncommenting the model deps
git clone https://github.com/trimplexx/music-transcription external/trimplexx
```
