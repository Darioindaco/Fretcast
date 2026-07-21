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
  exercises/         scripted Phase 0 takes + ground-truth CSVs + protocol
  scripts/
    run_baseline.py  run a model on a take and score it (model seam = transcribe())
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

## Roadmap (this milestone = bootstrap + fase 0-1)

1. **Fase 0** — record the three exercises to dry DI at 80 BPM
   (`exercises/phase0_protocol.md`). Ground truth is the tab; no annotation.
2. **Fase 1** — clone `trimplexx/music-transcription`, load its pretrained GuitarSet
   checkpoint, and implement `transcribe()` in `run_baseline.py`. Score against
   GuitarSet (in-domain) and your DI takes (out-of-domain). The gap between the two
   is the number fase 2 (domain adaptation) has to close.

## Model-side setup (run on the RTX box)

```
python -m venv .venv && .venv\Scripts\activate
# install the CUDA torch build from pytorch.org, then:
pip install -r core/requirements.txt   # after uncommenting the model deps
git clone https://github.com/trimplexx/music-transcription external/trimplexx
```
