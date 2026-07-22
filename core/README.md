# Fretcast V02 — core

The Python side: audio-first guitar tab transcription. Full pipeline explanation,
file-by-file layout, and the numbers behind every fase live in the
**[top-level README](../README.md)** — this file is just the quick "how do I run
something" reference.

## Verify the eval harness (no model, no audio, no venv needed)

```
python core/tests/test_tab_eval.py            # math + eval self-tests
python core/scripts/run_baseline.py --exercise ex3_devil_pairs --dry-run
```

The dry run scores a perfect prediction — every metric should read 100%.

## Model-side setup (run on the RTX box)

```
python -m venv .venv && .venv\Scripts\activate
# install the CUDA torch build from pytorch.org, then:
pip install -r core/requirements.txt   # after uncommenting the model deps
git clone https://github.com/trimplexx/music-transcription external/trimplexx
```

## Run the desktop app

```
python -m core.gui.app --source wav --wav-path <some.wav>   # no guitar needed
python -m core.gui.app --source live                          # real audio via your interface
```
