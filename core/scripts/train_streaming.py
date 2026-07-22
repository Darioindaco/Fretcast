"""Fase 3, first pass: train a causal (unidirectional) + limited-lookahead model.

Starts from the fase-2 fine-tuned checkpoint's CNN weights (warm start -- see
streaming_model.build_unidirectional_model_warm_started for why only the CNN transfers)
and trains the RNN + onset/fret heads from scratch, since a unidirectional GRU has
different weight shapes than the bidirectional one fase 1/2 used. Labels are shifted by
LOOKAHEAD_FRAMES (streaming_model.py) so the causal model is allowed a small, fixed
amount of future context -- exactly the plan's "unidirectioneel + beperkte lookahead"
requirement for a model that can run frame-by-frame in real time.

Same combined GuitarSet+EGDB train/validation split prepare_egdb_for_training.py already
wrote (features are architecture-independent, only the label transform changes here),
same held-out ids 100-119 excluded.

This script only produces and offline-validates the causal model -- it does not touch
ASIO capture, sliding-window CQT, or the live overlay. Those are the rest of fase 3 and
depend on this model actually holding up accuracy-wise first (plan section 7: "een model
dat offline niet werkt, debug je in real-time nooit").

Usage:
    python core/scripts/train_streaming.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FINETUNED_ARTIFACTS_DIR = Path("D:/FretcastData/artifacts/finetune_egdb_v1")
FINETUNED_CHECKPOINT = FINETUNED_ARTIFACTS_DIR / "best_model.pth"
FINETUNED_RUN_CONFIG = FINETUNED_ARTIFACTS_DIR / "run_configuration.json"

OUTPUT_BASE_DIR = "D:/FretcastData/processed_guitarset_data"
ARTIFACTS_DIR = "D:/FretcastData/artifacts/streaming_v1"

STREAMING_LR = 1e-4  # higher than the 3e-5 fase-2 finetune -- RNN+heads are fresh, not continuing
STREAMING_NUM_EPOCHS = 150
STREAMING_EARLY_STOPPING_PATIENCE = 20  # between fase-1's 25 (from scratch) and fase-2's 10 (pure continuation)

TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"
sys.path.insert(0, str(TRIMPLEXX_PY))

import torch  # noqa: E402
from torch import optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from data_processing import dataset, batching  # noqa: E402
from training import loss_functions, pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streaming_model import (  # noqa: E402
    LOOKAHEAD_FRAMES,
    LOOKAHEAD_SECONDS,
    build_unidirectional_model_warm_started,
    create_frame_level_labels_with_lookahead,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"lookahead: {LOOKAHEAD_FRAMES} frames = {LOOKAHEAD_SECONDS * 1000:.1f} ms")

    with open(FINETUNED_RUN_CONFIG, encoding="utf-8") as f:
        finetuned_run_config = json.load(f)
    hyperparams = dict(finetuned_run_config["hyperparameters_tuned"])
    hyperparams["RNN_BIDIRECTIONAL"] = False
    hyperparams["LEARNING_RATE_INIT"] = STREAMING_LR
    hyperparams["run_description"] = "streaming_v1_unidirectional_lookahead"

    print(f"warm-starting CNN from {FINETUNED_CHECKPOINT}, fresh unidirectional RNN+heads...")
    model = build_unidirectional_model_warm_started(FINETUNED_CHECKPOINT, finetuned_run_config, device)
    model.train()

    def lookahead_label_transform(raw_annotation_tensor, feature_map_tensor, hop, sr, max_fret):
        return create_frame_level_labels_with_lookahead(raw_annotation_tensor, feature_map_tensor, hop, sr, max_fret)

    common_params = {
        **config.DATASET_COMMON_PARAMS,
        "label_transform_function": lookahead_label_transform,
    }
    train_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR,
        data_split_name="train",
        enable_audio_augmentations=False,  # egdb_<id> ids aren't real mirdata tracks, see finetune_egdb.py
        enable_specaugment=True,
        **common_params,
    )
    val_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR,
        data_split_name="validation",
        enable_audio_augmentations=False,
        **common_params,
    )
    print(f"train: {len(train_ds)} tracks (GuitarSet + EGDB)")
    print(f"validation: {len(val_ds)} tracks (GuitarSet + EGDB)")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=batching.collate_fn_pad,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE_DEFAULT,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=batching.collate_fn_pad,
    )

    onset_pos_weight = (
        torch.tensor([hyperparams["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device)
        if hyperparams.get("ONSET_POS_WEIGHT_MANUAL_VALUE", -1) > 0
        else None
    )
    criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pos_weight,
        onset_loss_weight=hyperparams["ONSET_LOSS_WEIGHT"],
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams["LEARNING_RATE_INIT"],
        weight_decay=hyperparams["WEIGHT_DECAY"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=hyperparams["SCHEDULER_FACTOR"],
        patience=hyperparams["SCHEDULER_PATIENCE"],
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    log_file_path = os.path.join(ARTIFACTS_DIR, "training_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(
            f"streaming (unidirectional, lookahead={LOOKAHEAD_FRAMES} frames = "
            f"{LOOKAHEAD_SECONDS*1000:.1f}ms), CNN warm-started from {FINETUNED_CHECKPOINT}\n"
            f"hyperparameters: {json.dumps(hyperparams, indent=2)}\n"
        )

    run_config_for_json = dict(finetuned_run_config)
    run_config_for_json["hyperparameters_tuned"] = hyperparams
    run_config_for_json["warm_started_cnn_from"] = str(FINETUNED_CHECKPOINT)
    run_config_for_json["lookahead_frames"] = LOOKAHEAD_FRAMES
    with open(os.path.join(ARTIFACTS_DIR, "run_configuration.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_for_json, f, indent=4)

    run_training_config = {
        "NUM_EPOCHS": STREAMING_NUM_EPOCHS,
        "ARTIFACTS_DIR": ARTIFACTS_DIR,
        "LOG_FILE_PATH": log_file_path,
        "CHECKPOINT_METRIC": config.CHECKPOINT_METRIC_DEFAULT,
        "EARLY_STOPPING_PATIENCE": STREAMING_EARLY_STOPPING_PATIENCE,
    }

    print("\n--- training (streaming, unidirectional + lookahead) ---")
    pipeline.run_training_loop(
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        run_training_config,
        config,
    )
    print("\n--- done ---")


if __name__ == "__main__":
    main()
