"""Run a tab-transcription model on one exercise WAV and score it against ground truth.

This is the fase 1 harness. The model adapter is deliberately a single seam
(`transcribe`) so swapping trimplexx for FretNet, a finetuned checkpoint, or a hex-pickup
decoder later is a one-function change and every metric stays comparable.

Usage:
    python core/scripts/run_baseline.py --exercise ex1_chromatic --wav takes/ex1_chromatic.wav

Before a real model exists, pass --dry-run to score a perfect synthetic prediction and
confirm the pipeline end-to-end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python core/scripts/run_baseline.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fretcast.eval import evaluate  # noqa: E402
from fretcast.labels import load_exercise, load_predictions, save_predictions  # noqa: E402
from fretcast.tab import TabEvent  # noqa: E402

EXERCISE_DIR = Path(__file__).resolve().parents[1] / "exercises"


def transcribe(wav_path: Path) -> list[TabEvent]:
    """Model seam. Load audio, run the CRNN, return predicted tab events.

    TODO (fase 1): plug in trimplexx/music-transcription here.
        1. clone the repo and load its pretrained GuitarSet checkpoint
        2. wav -> CQT frames (its own preprocessing)
        3. model -> per-frame (string, fret) logits -> onset-gated note events
        4. map each note to a TabEvent(time_sec, string, fret)
    Keep everything else in this file untouched so fase-1 numbers stay comparable
    to fase-2 (finetuned) and beyond.
    """
    raise NotImplementedError(
        "No model wired yet. Run with --dry-run to validate the pipeline, or implement "
        "transcribe() against trimplexx (see the TODO in this function)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exercise", required=True, help="basename in core/exercises, e.g. ex1_chromatic")
    ap.add_argument("--wav", help="path to the recorded take")
    ap.add_argument("--out", help="where to write predictions CSV (default: <wav>.pred.csv)")
    ap.add_argument("--onset-tol", type=float, default=0.05, help="onset match tolerance (s)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="skip the model; score a perfect prediction to test the harness",
    )
    args = ap.parse_args()

    ref_csv = EXERCISE_DIR / f"{args.exercise}.csv"
    ref, bpm = load_exercise(ref_csv)
    print(f"exercise : {args.exercise}  ({len(ref)} notes @ {bpm} bpm)")

    if args.dry_run:
        # perfect prediction: exactly the ground truth -> every metric should read 100%
        pred = list(ref)
    else:
        if not args.wav:
            ap.error("--wav is required unless --dry-run")
        pred = transcribe(Path(args.wav))
        out = Path(args.out) if args.out else Path(args.wav).with_suffix(".pred.csv")
        save_predictions(out, pred)
        pred = load_predictions(out)  # round-trip, proves the file is usable downstream
        print(f"wrote    : {out}")

    result = evaluate(ref, pred, onset_tol=args.onset_tol)
    print("-" * 60)
    print(result.summary())
    if result.ambiguity_failures:
        print("-" * 60)
        print("ambiguity failures (right pitch, wrong position):")
        for r, p in result.ambiguity_failures:
            print(f"  {r.name:>4}  played s{r.string} f{r.fret}  ->  guessed s{p.string} f{p.fret}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
