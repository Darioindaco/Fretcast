"""Sanity check: does StreamingTranscriber (simulated live, small blocks) agree with the
already-validated offline whole-file transcribe on the same audio?

This does NOT re-measure accuracy against ground truth from scratch (eval_egdb.py
already did that for the whole file). It only checks that feeding audio in small blocks
through the rolling-buffer streaming path reproduces (closely) what the offline path
already produces -- i.e. that the streaming plumbing itself is correct, not a new
accuracy claim.

Usage:
    python core/scripts/validate_streaming.py --track-id 104
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fretcast.egdb import load_egdb_label  # noqa: E402
from fretcast.eval import evaluate  # noqa: E402
from run_baseline import load_model, transcribe_with_model  # noqa: E402
from streaming_infer import simulate_live_wav  # noqa: E402

STREAMING_ARTIFACTS = Path("D:/FretcastData/artifacts/streaming_v1")
DATA_DIR = Path("D:/FretcastData/egdb")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-id", type=int, default=104)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--onset-tol", type=float, default=0.05)
    args = ap.parse_args()

    wav_path = DATA_DIR / "audio_DI" / f"{args.track_id}.wav"
    midi_path = DATA_DIR / "audio_label" / f"{args.track_id}.midi"
    ref = load_egdb_label(midi_path)
    print(f"track {args.track_id}: {len(ref)} ground-truth notes")

    model_handle = load_model(STREAMING_ARTIFACTS / "best_model.pth", STREAMING_ARTIFACTS / "run_configuration.json")

    offline_pred = transcribe_with_model(wav_path, model_handle)
    print(f"offline (whole file):    {len(offline_pred)} notes predicted")

    streaming_pred, _transcriber = simulate_live_wav(wav_path, model_handle, block_size=args.block_size)
    print(f"streaming (simulated live, block={args.block_size} samples): {len(streaming_pred)} notes predicted")

    print("\n--- offline vs ground truth ---")
    print(evaluate(ref, offline_pred, onset_tol=args.onset_tol).summary())
    print("\n--- streaming vs ground truth ---")
    print(evaluate(ref, streaming_pred, onset_tol=args.onset_tol).summary())
    print("\n--- streaming vs offline (agreement check, not accuracy) ---")
    agreement = evaluate(offline_pred, streaming_pred, onset_tol=args.onset_tol)
    print(agreement.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
