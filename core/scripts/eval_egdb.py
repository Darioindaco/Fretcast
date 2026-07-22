"""Fase 1 domain-gap measurement against EGDB's real DI recordings.

EGDB (Chen et al., ISMIR 2022) ships a clean DI signal per tab with per-string MIDI
ground truth -- exactly the (audio, string, fret) pairing the eval harness needs,
downloaded ahead of Dario's own phase0_protocol.md takes (see core/fretcast/egdb.py
for the label parser). Its hex-pickup DI isn't identical to a plain humbucker/
single-coil through Dario's own interface, but it's a much closer domain than
GuitarSet's mic recording -- a useful early read before he records anything himself.

Usage:
    python core/scripts/eval_egdb.py --data-dir D:/FretcastData/egdb
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("D:/FretcastData/egdb"))
    ap.add_argument("--limit", type=int, default=None, help="only score the first N selected tracks")
    ap.add_argument(
        "--all",
        action="store_true",
        help="score every downloaded track instead of just the held-out eval ids (100-119). "
        "Only meaningful for a model that never trained on EGDB -- once a checkpoint has been "
        "fine-tuned on prepare_egdb_for_training.py's train/validation split, --all silently "
        "measures memorized data, not the domain gap.",
    )
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    args = ap.parse_args()

    audio_dir = args.data_dir / "audio_DI"
    label_dir = args.data_dir / "audio_label"
    ids = sorted(int(p.stem) for p in label_dir.glob("*.midi") if (audio_dir / f"{p.stem}.wav").exists())
    if not args.all:
        ids = [i for i in ids if i in HELD_OUT_EVAL_IDS]
    if not ids:
        print(f"no (audio, label) pairs found under {args.data_dir}"
              + ("" if args.all else " within the held-out eval ids"))
        return 1
    if args.limit:
        ids = ids[: args.limit]

    model_handle = load_model(args.checkpoint, args.run_config)

    results = []
    for track_id in ids:
        ref = load_egdb_label(label_dir / f"{track_id}.midi")
        pred = transcribe_with_model(audio_dir / f"{track_id}.wav", model_handle)
        r = evaluate(ref, pred, onset_tol=args.onset_tol)
        results.append(r)
        print(
            f"[{track_id:>3}] onset F1 {r.onset_f1:6.1%}  position {r.position_acc:6.1%}  "
            f"disambig {r.disambiguation_rate:6.1%}  ({len(ref)} ref notes)"
        )

    def avg(attr: str) -> float:
        return sum(getattr(r, attr) for r in results) / len(results)

    print("-" * 60)
    print(f"scored {len(results)} EGDB tracks (of {len(ids)} available pairs)")
    print(f"mean onset F1       {avg('onset_f1'):.1%}")
    print(f"mean pitch acc      {avg('pitch_acc'):.1%}")
    print(f"mean string acc     {avg('string_acc'):.1%}")
    print(f"mean fret acc       {avg('fret_acc'):.1%}")
    print(f"mean position acc  {avg('position_acc'):.1%}")
    print(f"mean disambig rate  {avg('disambiguation_rate'):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
