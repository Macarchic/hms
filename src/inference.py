"""
Inference entry point. Loads best checkpoint and produces submission.csv.

Usage:
    python src/inference.py                              # kfuji, auto-detect best ckpt
    python src/inference.py --modality kfuji
    python src/inference.py --ckpt checkpoints/kfuji/epoch=05-val_loss=0.2399.ckpt
    python src/inference.py --out my_submission.csv
"""
import argparse
import glob
import importlib
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.logger import get_logger
from src.utils import CKPT_DIR, DATA_DIR, LOGS_DIR

VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
OUT_COLS = ["eeg_id"] + VOTE_COLS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", default="kfuji")
    p.add_argument("--ckpt", default=None,
                   help="Path to .ckpt file. Auto-selects lowest val_loss if omitted.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="submission.csv")
    return p.parse_args()


def find_best_ckpt(ckpt_dir: str) -> str:
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    def val_loss(path):
        try:
            return float(os.path.basename(path).split("val_loss=")[1].removesuffix(".ckpt"))
        except Exception:
            return float("inf")

    best = min(ckpts, key=val_loss)
    return best


def main():
    args = parse_args()

    name = args.modality.capitalize()
    cfg_cls = getattr(importlib.import_module(f"src.models.{args.modality}.config"), f"{name}Config")
    ds_cls = getattr(importlib.import_module(f"src.models.{args.modality}.dataset"), "HMSDataset")
    mod_cls = getattr(importlib.import_module(f"src.models.{args.modality}.model"), f"{name}Module")

    log_dir = os.path.join(LOGS_DIR, args.modality)
    os.makedirs(log_dir, exist_ok=True)
    log = get_logger(f"{args.modality}_infer", log_dir)

    ckpt_path = args.ckpt or find_best_ckpt(os.path.join(CKPT_DIR, args.modality))
    log.info(f"checkpoint: {ckpt_path}")

    cfg = cfg_cls()
    model = mod_cls.load_from_checkpoint(ckpt_path, config=cfg)
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    log.info(f"device: {device}")

    # test.csv has no vote columns and no eeg_label_offset_seconds
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_df["eeg_label_offset_seconds"] = 0
    for col in VOTE_COLS:
        test_df[col] = 0

    eeg_dir = os.path.join(DATA_DIR, "test_eegs")
    test_ds = ds_cls(test_df, eeg_dir, cfg)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_probs = []
    with torch.no_grad():
        for x, _ in test_dl:
            probs = F.softmax(model(x.to(device)), dim=1).cpu()
            all_probs.append(probs)

    probs = torch.cat(all_probs, dim=0).numpy()
    sub = pd.DataFrame(probs, columns=VOTE_COLS)
    sub.insert(0, "eeg_id", test_df["eeg_id"].values)
    sub.to_csv(args.out, index=False)
    log.info(f"Saved {len(sub)} rows → {args.out}")


if __name__ == "__main__":
    main()
