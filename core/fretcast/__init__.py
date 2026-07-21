"""Fretcast V02 core — audio-first guitar tab transcription.

Pure-Python, dependency-free foundation:
  - tab.py     fretboard math + candidate positions (disambiguation core)
  - labels.py  load/save ground-truth tab labels (scripted exercises)
  - eval.py    evaluation harness: per-note pitch / string / fret accuracy

The heavy model side (PyTorch CRNN, librosa/CQT, ASIO capture) plugs in on top
via scripts/run_baseline.py and keeps its dependencies out of this package so the
eval harness always runs with a bare Python install.
"""

__all__ = ["tab", "labels", "eval"]
