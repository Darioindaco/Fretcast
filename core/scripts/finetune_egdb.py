"""Fase 2, first pass: fine-tune the GuitarSet-trained baseline on real DI audio (EGDB).

Loads the existing best_model.pth as INITIALIZATION (not a fresh random model -- that's
the whole point of fine-tuning) via trimplexx's own model_utils.load_best_model, then
continues training with a much lower learning rate on the combined GuitarSet + EGDB
processed data that prepare_egdb_for_training.py already wrote into
processed_guitarset_data/{train,validation} (same directory, egdb_<id> ids appended to
the existing *_ids.txt files -- GuitarSetTabDataset reads them identically either way).

`enable_audio_augmentations=False` for BOTH splits: egdb_<id> ids aren't real mirdata
GuitarSet track ids, so the live audio-augmentation path (which resolves each id's audio
file through mirdata) would silently drop every EGDB row from training. Precomputed CQT
features + spec-augment (time/freq masking, independent of that flag) are used instead --
good enough for a first fine-tune pass; real EGDB-domain audio augmentation is future work
if this round doesn't close enough of the gap.

Reuses trimplexx's own run_training_loop unchanged (just fed a pretrained model instead
of process_single_hyperparameter_run's forced fresh init), so training mechanics/logging/
checkpointing/early-stopping match the original baseline run exactly.

Usage:
    python core/scripts/finetune_egdb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASELINE_ARTIFACTS_DIR = Path(
    "D:/FretcastData/artifacts/baseline/run_0_Baseline_run72_re-run_with_new_aug_augEnabled"
)
BASELINE_CHECKPOINT = BASELINE_ARTIFACTS_DIR / "best_model.pth"
BASELINE_RUN_CONFIG = BASELINE_ARTIFACTS_DIR / "run_configuration.json"

OUTPUT_BASE_DIR = "D:/FretcastData/processed_guitarset_data"
ARTIFACTS_DIR = "D:/FretcastData/artifacts/finetune_egdb_v1"

FINETUNE_LR = 3e-5  # ~1/10th of the original 3e-4 -- continuation, not from-scratch training
FINETUNE_NUM_EPOCHS = 60
FINETUNE_EARLY_STOPPING_PATIENCE = 10  # fine-tuning converges faster than from-scratch (patience 25)

TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"
sys.path.insert(0, str(TRIMPLEXX_PY))

import torch  # noqa: E402
from torch import optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from data_processing import dataset, batching  # noqa: E402
from model import architecture, utils as model_utils  # noqa: E402
from training import loss_functions, pipeline  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(BASELINE_RUN_CONFIG, encoding="utf-8") as f:
        baseline_run_config = json.load(f)
    hyperparams = dict(baseline_run_config["hyperparameters_tuned"])
    hyperparams["LEARNING_RATE_INIT"] = FINETUNE_LR
    hyperparams["run_description"] = "egdb_finetune_v1"

    print("loading pretrained baseline as fine-tune initialization...")
    model = model_utils.load_best_model(
        architecture.GuitarTabCRNN, str(BASELINE_CHECKPOINT), str(BASELINE_RUN_CONFIG), device
    )
    if model is None:
        raise RuntimeError("failed to load baseline checkpoint for fine-tuning")
    model.train()

    common_params = {
        **config.DATASET_COMMON_PARAMS,
        "label_transform_function": dataset.create_frame_level_labels,
    }
    # enable_audio_augmentations=False for train too -- see module docstring
    train_ds = dataset.GuitarSetTabDataset(
        processed_data_base_dir=OUTPUT_BASE_DIR,
        data_split_name="train",
        enable_audio_augmentations=False,
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
        mode="max",  # CHECKPOINT_METRIC_DEFAULT is val_tdr_f1 (higher is better)
        factor=hyperparams["SCHEDULER_FACTOR"],
        patience=hyperparams["SCHEDULER_PATIENCE"],
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    log_file_path = os.path.join(ARTIFACTS_DIR, "training_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"fine-tune from {BASELINE_CHECKPOINT}\nhyperparameters: {json.dumps(hyperparams, indent=2)}\n")

    run_config_for_json = dict(baseline_run_config)
    run_config_for_json["hyperparameters_tuned"] = hyperparams
    run_config_for_json["finetuned_from"] = str(BASELINE_CHECKPOINT)
    with open(os.path.join(ARTIFACTS_DIR, "run_configuration.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_for_json, f, indent=4)

    run_training_config = {
        "NUM_EPOCHS": FINETUNE_NUM_EPOCHS,
        "ARTIFACTS_DIR": ARTIFACTS_DIR,
        "LOG_FILE_PATH": log_file_path,
        "CHECKPOINT_METRIC": config.CHECKPOINT_METRIC_DEFAULT,
        "EARLY_STOPPING_PATIENCE": FINETUNE_EARLY_STOPPING_PATIENCE,
    }

    print("\n--- fine-tuning ---")
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
