"""Fase 1 baseline: reproduce trimplexx's best reported GuitarSet run.

Mirrors external/trimplexx/python/guitar_ATM.ipynb cells 2/4/6/10, but as a plain
script (no Jupyter) and running ONLY hyperparam_set_v1.json[0] -- the
"Baseline_run72_re-run_with_new_aug" config, i.e. the RNN_HIDDEN_SIZE=768 GRU setup
config.py documents as the paper's best (TDR F1=0.8569). The JSON also lists 4 more
hyperparameter variants; those are a sweep for later, not part of this baseline run.

All data/artifacts live under D:\FretcastData (the small C: drive stays untouched):
    D:\FretcastData\mir_datasets_storage      GuitarSet (see download_guitarset.py)
    D:\FretcastData\processed_guitarset_data  cached CQT features + labels
    D:\FretcastData\artifacts\baseline        checkpoints, logs, run_configuration.json

Usage:
    python core/scripts/train_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"
sys.path.insert(0, str(TRIMPLEXX_PY))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402  (trimplexx's config.py)
from data_processing import preparation, dataset, batching  # noqa: E402
from training.pipeline import process_single_hyperparameter_run  # noqa: E402

DATA_HOME = "D:/FretcastData/mir_datasets_storage"
OUTPUT_BASE_DIR = "D:/FretcastData/processed_guitarset_data"
ARTIFACTS_DIR = "D:/FretcastData/artifacts/baseline"
HYPERPARAMS_FILE = TRIMPLEXX_PY / "hyperparam_set_v1.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device found; this will be extremely slow.")

    os.makedirs(DATA_HOME, exist_ok=True)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(HYPERPARAMS_FILE, encoding="utf-8") as f:
        param_grid = json.load(f)
    hyperparams = param_grid[0]
    print(f"hyperparameter set: {hyperparams['run_description']!r}")

    # ---- stage 1: split + preprocess (skips tracks already cached on disk) ----
    print("\n--- stage 1: track splits + CQT preprocessing ---")
    track_splits = preparation.prepare_track_splits(
        data_home=DATA_HOME,
        problematic_files_list=config.PROBLEMATIC_FILES,
        test_split_fraction=config.TEST_SPLIT_SIZE,
        validation_split_fraction=config.VALIDATION_SPLIT_SIZE,
        seed=config.RANDOM_SEED,
        output_dir_for_ids=OUTPUT_BASE_DIR,
    )
    if any(len(ids) > 0 for ids in track_splits.values()):
        preparation.preprocess_guitarset_data(
            guitarset_data_home=DATA_HOME,
            processed_output_base_dir=OUTPUT_BASE_DIR,
            track_ids_map=track_splits,
            audio_sample_rate=config.SAMPLE_RATE,
            audio_hop_length=config.HOP_LENGTH,
            audio_n_cqt_bins=config.N_BINS_CQT,
            audio_cqt_bins_per_octave=config.BINS_PER_OCTAVE_CQT,
            audio_cqt_fmin=config.FMIN_CQT,
        )

    # ---- stage 2: datasets + dataloaders ----
    print("\n--- stage 2: datasets ---")
    train_common = {
        **config.DATASET_COMMON_PARAMS,
        **config.DATASET_TRAIN_AUGMENTATION_PARAMS,
        "label_transform_function": dataset.create_frame_level_labels,
        "guitarset_data_home": DATA_HOME,
    }
    eval_common = {
        **config.DATASET_COMMON_PARAMS,
        **config.DATASET_EVAL_AUGMENTATION_PARAMS,
        "label_transform_function": dataset.create_frame_level_labels,
        "guitarset_data_home": DATA_HOME,
    }

    def make_loader(split: str, params: dict, shuffle: bool) -> DataLoader | None:
        if not track_splits.get(split):
            return None
        ds = dataset.GuitarSetTabDataset(
            processed_data_base_dir=OUTPUT_BASE_DIR, data_split_name=split, **params
        )
        if len(ds) == 0:
            return None
        print(f"  {split}: {len(ds)} samples")
        return DataLoader(
            ds,
            batch_size=config.BATCH_SIZE_DEFAULT,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=batching.collate_fn_pad,
        )

    train_loader = make_loader("train", train_common, shuffle=True)
    validation_loader = make_loader("validation", eval_common, shuffle=False)
    test_loader = make_loader("test", eval_common, shuffle=False)

    if train_loader is None or validation_loader is None:
        raise RuntimeError("train/validation set is empty -- did preprocessing succeed?")

    # ---- stage 3: train the single baseline config ----
    print("\n--- stage 3: training (this is the multi-hour part) ---")
    summary = process_single_hyperparameter_run(
        run_id=0,
        hyperparams_combo=hyperparams,
        current_augmentation_params=config.DATASET_TRAIN_AUGMENTATION_PARAMS,
        config_obj=config,
        main_artifacts_dir=ARTIFACTS_DIR,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )

    print("\n--- done ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
