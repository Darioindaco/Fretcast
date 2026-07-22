"""Convert downloaded EGDB (audio_DI + audio_label) pairs into trimplexx's own
processed-data format so GuitarSetTabDataset can read them unchanged during fine-tuning.

Writes `egdb_<id>_features.pt` / `egdb_<id>_labels.pt` straight into the SAME
processed_guitarset_data/{train,validation} directories train_baseline.py already
populated for GuitarSet, and appends `egdb_<id>` lines to that directory's
`{split}_ids.txt` files -- so a fine-tune run can just point at the existing dirs and
get both datasets for free, no new Dataset class needed.

A fixed held-out set of track ids (HELD_OUT_EVAL_IDS) is never written to train or
validation here -- those are the tracks eval_egdb.py already measured the untouched
GuitarSet-trained baseline against (73.0% onset F1 / 38.3% position acc, 20-track
pilot). Keeping them out of fine-tuning is what lets a later eval_egdb.py run on the
same ids be a fair before/after comparison instead of measuring memorized data.

Idempotent: re-running only processes ids that don't already have both output files
and an ids.txt entry, so it's safe to run again as more of the 240 EGDB tracks finish
downloading in the background.

Usage:
    python core/scripts/prepare_egdb_for_training.py --val-fraction 0.15
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"
sys.path.insert(0, str(TRIMPLEXX_PY))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import librosa  # noqa: E402
import config as tx_config  # noqa: E402

from fretcast.egdb import load_egdb_notes_for_training  # noqa: E402

EGDB_DIR = Path("D:/FretcastData/egdb")
PROCESSED_DIR = Path("D:/FretcastData/processed_guitarset_data")

# tracks 100-119: already scored as the fase-1 domain-gap pilot (eval_egdb.py). Never
# train on these so that number stays a legitimate before/after comparison.
HELD_OUT_EVAL_IDS = set(range(100, 120))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=tx_config.RANDOM_SEED)
    args = ap.parse_args()

    audio_dir = EGDB_DIR / "audio_DI"
    label_dir = EGDB_DIR / "audio_label"

    available_ids = sorted(
        int(p.stem)
        for p in label_dir.glob("*.midi")
        if (audio_dir / f"{p.stem}.wav").exists() and int(p.stem) not in HELD_OUT_EVAL_IDS
    )
    print(f"{len(available_ids)} EGDB (audio, label) pairs available for training/validation "
          f"({len(HELD_OUT_EVAL_IDS)} held out for eval)")
    if not available_ids:
        return 1

    rng = random.Random(args.seed)
    shuffled = available_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_ids = set(shuffled[:n_val])
    split_of = {tid: ("validation" if tid in val_ids else "train") for tid in available_ids}

    stats = {"train": {"processed": 0, "skipped": 0}, "validation": {"processed": 0, "skipped": 0}}

    for split in ("train", "validation"):
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    ids_file_lines: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        ids_path = PROCESSED_DIR / f"{split}_ids.txt"
        existing = set(ids_path.read_text(encoding="utf-8").splitlines()) if ids_path.exists() else set()
        ids_file_lines[split] = existing

    for track_id in available_ids:
        split = split_of[track_id]
        base = f"egdb_{track_id}"
        features_path = PROCESSED_DIR / split / f"{base}_features.pt"
        labels_path = PROCESSED_DIR / split / f"{base}_labels.pt"
        already_listed = base in ids_file_lines[split]

        if features_path.exists() and labels_path.exists() and already_listed:
            stats[split]["skipped"] += 1
            continue

        notes = load_egdb_notes_for_training(label_dir / f"{track_id}.midi")
        if not notes:
            print(f"  egdb_{track_id}: no notes parsed, skipping")
            continue

        audio, _ = librosa.load(str(audio_dir / f"{track_id}.wav"), sr=tx_config.SAMPLE_RATE, mono=True)
        cqt = librosa.cqt(
            y=audio,
            sr=tx_config.SAMPLE_RATE,
            hop_length=tx_config.HOP_LENGTH,
            fmin=tx_config.FMIN_CQT,
            n_bins=tx_config.N_BINS_CQT,
            bins_per_octave=tx_config.BINS_PER_OCTAVE_CQT,
        )
        log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        torch.save(torch.tensor(log_cqt, dtype=torch.float32), features_path)
        torch.save(torch.tensor(np.array(notes, dtype=np.float32)), labels_path)
        ids_file_lines[split].add(base)
        stats[split]["processed"] += 1

    for split in ("train", "validation"):
        ids_path = PROCESSED_DIR / f"{split}_ids.txt"
        # rewrite whole file de-duplicated (it was seeded from the existing content
        # above) rather than blind-append, since this script gets re-run as more EGDB
        # tracks finish downloading
        ids_path.write_text("\n".join(sorted(ids_file_lines[split])) + "\n", encoding="utf-8")

    print("-" * 60)
    for split in ("train", "validation"):
        n_egdb_total = sum(1 for l in ids_file_lines[split] if l.startswith("egdb_"))
        print(f"{split}: {stats[split]['processed']} newly processed, {stats[split]['skipped']} already done, "
              f"{n_egdb_total} egdb ids total ({len(ids_file_lines[split])} incl. GuitarSet) in {split}_ids.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
