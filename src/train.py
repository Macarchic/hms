"""
Training entry point.

Usage:
    python src/train.py                          # kfuji, fold 0, single stage
    python src/train.py --modality kfuji --fold 1
    python src/train.py --modality kfuji --two_stage
    python src/train.py --modality kfuji --epochs 20 --batch_size 8

Adding a new modality:
    1. Create src/models/<name>/config.py  with class <Name>Config
    2. Create src/models/<name>/dataset.py with class HMSDataset
    3. Create src/models/<name>/model.py   with class <Name>Module
    Then pass --modality <name>.
"""
import argparse
import importlib
import os

import torch
import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.callbacks import EpochMetricsLogger, PredictionLogger
from src.logger import get_logger
from src.utils import CKPT_DIR, DATA_DIR, LOGS_DIR

VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", default="kfuji")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fold", type=int, default=0, help="validation fold index 0-4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--two_stage", action="store_true",
                   help="stage1=all votes>1, stage2=votes>=10 with lr/10")
    p.add_argument("--stage1_epochs", type=int, default=5)
    p.add_argument("--train_size", type=int, default=None, help="clip train set to N rows (debug)")
    p.add_argument("--val_size", type=int, default=None, help="clip val set to N rows (debug)")
    return p.parse_args()


def load_modality(modality: str):
    """Import Config, HMSDataset, Module from src/models/<modality>/."""
    name = modality.capitalize()
    cfg_cls = getattr(importlib.import_module(f"src.models.{modality}.config"), f"{name}Config")
    ds_cls = getattr(importlib.import_module(f"src.models.{modality}.dataset"), "HMSDataset")
    mod_cls = getattr(importlib.import_module(f"src.models.{modality}.model"), f"{name}Module")
    return cfg_cls, ds_cls, mod_cls


def fold_split(df: pd.DataFrame, fold: int, n_folds: int = 5):
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(df, groups=df["patient_id"].values))
    train_idx, val_idx = splits[fold]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def make_loaders(ds_cls, cfg, train_df, val_df, eeg_dir, num_workers):
    train_ds = ds_cls(train_df, eeg_dir, cfg, training=True)
    val_ds = ds_cls(val_df, eeg_dir, cfg, training=False)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kw)
    return train_dl, val_dl


def build_trainer(cfg, log_dir, ckpt_dir, max_epochs, patience=7):
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        EpochMetricsLogger(log_path=os.path.join(log_dir, "metrics.csv")),
        PredictionLogger(log_path=os.path.join(log_dir, "preds.csv")),
    ]
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    return trainer, ckpt_cb


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    cfg_cls, ds_cls, mod_cls = load_modality(args.modality)
    torch.serialization.add_safe_globals([cfg_cls])
    cfg = cfg_cls()

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers

    log_dir = os.path.join(LOGS_DIR, args.modality)
    ckpt_dir = os.path.join(CKPT_DIR, args.modality)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = get_logger(args.modality, log_dir)
    log.info(f"modality={args.modality}  fold={args.fold}  seed={args.seed}  two_stage={args.two_stage}")

    eeg_dir = os.path.join(DATA_DIR, "train_eegs")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["total_votes"] = df[VOTE_COLS].sum(axis=1)

    train_all, val_df = fold_split(df, args.fold)
    if args.val_size is not None:
        val_df = val_df.head(args.val_size).reset_index(drop=True)

    if args.two_stage:
        # --- Stage 1: all data with votes > 1 ---
        stage1_df = train_all[train_all["total_votes"] > 1]
        if args.train_size is not None:
            stage1_df = stage1_df.head(args.train_size).reset_index(drop=True)
        log.info(f"Stage 1: {len(stage1_df):,} train  {len(val_df):,} val  epochs={args.stage1_epochs}")

        train_dl, val_dl = make_loaders(ds_cls, cfg, stage1_df, val_df, eeg_dir, args.num_workers)
        model = mod_cls(cfg)
        trainer, ckpt_cb = build_trainer(cfg, log_dir, ckpt_dir, max_epochs=args.stage1_epochs)
        trainer.fit(model, train_dl, val_dl)

        # --- Stage 2: quality data, lower LR ---
        stage2_epochs = cfg.epochs - args.stage1_epochs
        cfg.lr = cfg.lr / 10
        stage2_df = train_all[train_all["total_votes"] >= 10]
        if args.train_size is not None:
            stage2_df = stage2_df.head(args.train_size).reset_index(drop=True)
        log.info(f"Stage 2: {len(stage2_df):,} train  lr={cfg.lr}  epochs={stage2_epochs}")

        train_dl, val_dl = make_loaders(ds_cls, cfg, stage2_df, val_df, eeg_dir, args.num_workers)
        model = mod_cls.load_from_checkpoint(ckpt_cb.best_model_path, config=cfg)
        trainer2, _ = build_trainer(cfg, log_dir, ckpt_dir, max_epochs=stage2_epochs)
        trainer2.fit(model, train_dl, val_dl)
    else:
        train_df = train_all[train_all["total_votes"] > 1]
        if args.train_size is not None:
            train_df = train_df.head(args.train_size).reset_index(drop=True)
        log.info(f"Single stage: {len(train_df):,} train  {len(val_df):,} val  epochs={cfg.epochs}")

        train_dl, val_dl = make_loaders(ds_cls, cfg, train_df, val_df, eeg_dir, args.num_workers)
        model = mod_cls(cfg)
        trainer, _ = build_trainer(cfg, log_dir, ckpt_dir, max_epochs=cfg.epochs)
        trainer.fit(model, train_dl, val_dl)

    log.info("Done.")


if __name__ == "__main__":
    main()
