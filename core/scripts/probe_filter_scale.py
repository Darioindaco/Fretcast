"""Fase-3 CQT-latency probe: how much does a shorter CQT filter_scale cost, without retraining?

The low E string's CQT filter needs a ~420ms analysis window at the current config
(24 bins/octave, filter_scale=1.0) -- see the fase-3 discussion this script grew out of.
`filter_scale` scales every filter's window length linearly (filter_scale=0.5 halves it),
at the cost of more spectral leakage in the resulting CQT.

IMPORTANT CAVEAT this script exists to make explicit: the model was trained on
filter_scale=1.0 features. Running it at a lower filter_scale here measures the model's
ROBUSTNESS TO A TRAIN/TEST MISMATCH, not the accuracy a model trained from scratch at
that filter_scale would reach -- those are different numbers. A bad result here doesn't
prove shorter windows won't work (a matched retrain might do fine); a good result here
is a real, free win (no retrain needed). Either way, this is a cheap first signal before
deciding whether a full retrain-at-shorter-filter_scale (or a bins-per-octave reduction,
which DOES require retraining the CNN from scratch, unlike this) is worth doing.

Usage:
    python core/scripts/probe_filter_scale.py --filter-scales 1.0 0.75 0.5 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fretcast.egdb import load_egdb_label  # noqa: E402
from fretcast.eval import evaluate  # noqa: E402
from run_baseline import DEFAULT_CHECKPOINT, DEFAULT_RUN_CONFIG, load_model, transcribe_with_model  # noqa: E402
from prepare_egdb_for_training import HELD_OUT_EVAL_IDS  # noqa: E402

TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"
sys.path.insert(0, str(TRIMPLEXX_PY))
import librosa  # noqa: E402
import config as tx_config  # noqa: E402


def low_e_window_ms(filter_scale: float) -> float:
    freqs = librosa.cqt_frequencies(
        n_bins=tx_config.N_BINS_CQT, fmin=tx_config.FMIN_CQT, bins_per_octave=tx_config.BINS_PER_OCTAVE_CQT
    )
    lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=tx_config.SAMPLE_RATE, filter_scale=filter_scale)
    return lengths[0] / tx_config.SAMPLE_RATE * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("D:/FretcastData/egdb"))
    ap.add_argument("--filter-scales", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    args = ap.parse_args()

    audio_dir = args.data_dir / "audio_DI"
    label_dir = args.data_dir / "audio_label"
    ids = sorted(
        int(p.stem) for p in label_dir.glob("*.midi")
        if (audio_dir / f"{p.stem}.wav").exists() and int(p.stem) in HELD_OUT_EVAL_IDS
    )
    if not ids:
        print("no held-out (audio, label) pairs found")
        return 1
    refs = {tid: load_egdb_label(label_dir / f"{tid}.midi") for tid in ids}

    model_handle = load_model(args.checkpoint, args.run_config)

    print(f"{'filter_scale':>12}  {'low-E window':>13}  {'onset F1':>9}  {'position acc':>13}  {'disambig':>9}")
    for fs in args.filter_scales:
        results = []
        for tid in ids:
            pred = transcribe_with_model(audio_dir / f"{tid}.wav", model_handle, filter_scale=fs)
            results.append(evaluate(refs[tid], pred, onset_tol=args.onset_tol))
        avg = lambda attr: sum(getattr(r, attr) for r in results) / len(results)  # noqa: E731
        print(
            f"{fs:12.2f}  {low_e_window_ms(fs):11.1f}ms  {avg('onset_f1'):8.1%}  "
            f"{avg('position_acc'):12.1%}  {avg('disambiguation_rate'):8.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
