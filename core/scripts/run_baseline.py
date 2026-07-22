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
from fretcast.tab import TabEvent, trimplexx_string_to_fretcast  # noqa: E402

EXERCISE_DIR = Path(__file__).resolve().parents[1] / "exercises"
TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"

DEFAULT_ARTIFACTS_DIR = (
    Path("D:/FretcastData/artifacts/baseline/run_0_Baseline_run72_re-run_with_new_aug_augEnabled")
)
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACTS_DIR / "best_model.pth"
DEFAULT_RUN_CONFIG = DEFAULT_ARTIFACTS_DIR / "run_configuration.json"


def load_model(checkpoint: Path = DEFAULT_CHECKPOINT, run_config: Path = DEFAULT_RUN_CONFIG):
    """Load the trained trimplexx checkpoint once. Returns a handle for `transcribe_with_model`.

    Reuses trimplexx's own checkpoint loader (model.utils.load_best_model) so the
    reconstructed architecture always matches what was actually trained. If the run
    config carries a `lookahead_frames` key (written by train_streaming.py for a
    unidirectional/streaming model, see streaming_model.py), `transcribe_with_model`
    automatically corrects decoded onset times for it -- callers don't need to know
    whether a given checkpoint is the fase-1/2 bidirectional model or a fase-3
    streaming one.
    """
    import json

    sys.path.insert(0, str(TRIMPLEXX_PY))
    import torch  # noqa: E402
    import config as tx_config  # noqa: E402
    from model import architecture, utils as model_utils  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_utils.load_best_model(
        architecture.GuitarTabCRNN, str(checkpoint), str(run_config), device
    )
    if model is None:
        raise RuntimeError(f"failed to load model from {checkpoint} / {run_config}")

    with open(run_config, encoding="utf-8") as f:
        lookahead_frames = json.load(f).get("lookahead_frames", 0)

    return model, device, tx_config, lookahead_frames


def transcribe_array_with_model(audio, model_handle, filter_scale: float = 1.0) -> list[TabEvent]:
    """Model seam, array-based. Run raw mono audio (np.ndarray, already at tx_config.SAMPLE_RATE)
    through an already-loaded model, return predicted tab events.

    Reuses trimplexx's own CQT frontend (data_processing.preparation) and frame-to-note
    decoder (training.note_conversion_utils.frames_to_notes_for_eval) so fase-1 numbers
    are directly comparable to their own reported GuitarSet metrics. The only translation
    layer here is string numbering: trimplexx counts 0=low E..5=high e, Fretcast counts
    1=high E..6=low E (see fretcast/tab.py).

    `filter_scale` overrides librosa's default (1.0, what the model was trained on) --
    only meant for the fase-3 CQT-window-latency probe (probe_filter_scale.py): a value
    below 1.0 shortens each CQT filter's analysis window (shorter low-frequency latency)
    at the cost of spectral leakage the model never saw during training. Measures
    robustness to that train/test mismatch, not the accuracy ceiling a matched retrain
    at this filter_scale would reach -- see that script's docstring.

    This is the array-based half of what used to be transcribe_with_model (still below,
    now a thin file-loading wrapper around this) -- factored out so streaming_infer.py's
    rolling audio buffer can call it directly without writing to disk first.
    """
    model, device, tx_config, lookahead_frames = model_handle
    import torch  # noqa: E402
    import librosa  # noqa: E402
    import numpy as np  # noqa: E402
    from training import note_conversion_utils  # noqa: E402

    cqt = librosa.cqt(
        y=audio,
        sr=tx_config.SAMPLE_RATE,
        hop_length=tx_config.HOP_LENGTH,
        fmin=tx_config.FMIN_CQT,
        n_bins=tx_config.N_BINS_CQT,
        bins_per_octave=tx_config.BINS_PER_OCTAVE_CQT,
        filter_scale=filter_scale,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        onset_logits, fret_logits = model(features)
        onset_binary = (torch.sigmoid(onset_logits) > 0.5).float()[0].cpu()
        fret_pred_indices = torch.argmax(fret_logits, dim=-1)[0].cpu()

    notes = note_conversion_utils.frames_to_notes_for_eval(
        onset_binary,
        fret_pred_indices,
        tx_config.HOP_LENGTH,
        tx_config.SAMPLE_RATE,
        tx_config.MAX_FRETS,
    )
    lookahead_seconds = lookahead_frames * tx_config.HOP_LENGTH / tx_config.SAMPLE_RATE
    return [
        TabEvent(
            time=max(0.0, n["start_time"] - lookahead_seconds),
            string=trimplexx_string_to_fretcast(n["string"]),
            fret=n["fret"],
        )
        for n in notes
    ]


def transcribe_with_model(wav_path: Path, model_handle, filter_scale: float = 1.0) -> list[TabEvent]:
    """File-based wrapper around `transcribe_array_with_model` (see its docstring for
    what actually happens). Kept as a separate name since eval_egdb.py and every fase-1/2
    script already call this one directly.
    """
    import librosa  # noqa: E402

    _, _, tx_config, _ = model_handle
    audio, _ = librosa.load(str(wav_path), sr=tx_config.SAMPLE_RATE, mono=True)
    return transcribe_array_with_model(audio, model_handle, filter_scale=filter_scale)


def transcribe(
    wav_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    run_config: Path = DEFAULT_RUN_CONFIG,
) -> list[TabEvent]:
    """Single-file convenience wrapper: load the model and transcribe one WAV.

    For scoring many files (see eval_egdb.py), call `load_model` once and reuse it
    with `transcribe_with_model` instead -- this wrapper reloads the checkpoint from
    disk on every call.
    """
    return transcribe_with_model(wav_path, load_model(checkpoint, run_config))


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
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="model .pth to load")
    ap.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG, help="run_configuration.json to load")
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
        pred = transcribe(Path(args.wav), checkpoint=args.checkpoint, run_config=args.run_config)
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
