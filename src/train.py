"""
Usage:
    python src/train.py --modality eeg
    python src/train.py --modality spec
    python src/train.py --modality eeg  --backbone efficientnet_b2
    python src/train.py --modality eeg  --backbone efficientnet_b4 --batch_size 8
    python src/train.py --modality eeg  --mixup_alpha 0 --label_smooth 0   # ablation
    python src/train.py --modality spec --epochs 1  --batch_size 4         # sanity check
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import EpochMetricsLogger, PredictionLogger
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from dataset import HMSSpectrogramDataset, HMSEEGDataset, VOTE_COLS
from spec_model import SpectrogramModel
from eeg_model import EEGModel
from utils import get_device, DATA_DIR, CKPT_DIR, LOGS_DIR, CACHE_DIR


# ── Lightning module ──────────────────────────────────────────────────────────

class HMSModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float, freeze_epochs: int = 5,
                 weight_decay: float = 3e-4, label_smooth: float = 0.1,
                 mixup_alpha: float = 0.4):
        super().__init__()
        self.model        = model
        self.lr           = lr
        self.freeze_epochs = freeze_epochs
        self.weight_decay = weight_decay
        self.label_smooth = label_smooth
        self.mixup_alpha  = mixup_alpha
        self.save_hyperparameters(ignore=['model'])

        if freeze_epochs > 0 and hasattr(model, 'backbone'):
            for p in model.backbone.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        if self.freeze_epochs > 0 and self.current_epoch == self.freeze_epochs:
            if hasattr(self.model, 'backbone'):
                for p in self.model.backbone.parameters():
                    p.requires_grad = True
                print(f'\n  [epoch {self.current_epoch}] backbone unfrozen')

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.kl_div(F.log_softmax(logits, dim=1), y, reduction='batchmean')
        acc = (logits.argmax(1) == y.argmax(1)).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        x, y = batch

        if self.label_smooth > 0:
            y = (1 - self.label_smooth) * y + self.label_smooth / y.size(1)

        if self.mixup_alpha > 0 and x.size(0) > 1:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

        logits = self(x)
        loss = F.kl_div(F.log_softmax(logits, dim=1), y, reduction='batchmean')
        acc = (logits.argmax(1) == y.argmax(1)).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc',  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc',  acc,  on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.model, 'backbone') and hasattr(self.model, 'head'):
            param_groups = [
                {'params': self.model.backbone.parameters(), 'lr': self.lr * 0.1},
                {'params': self.model.head.parameters(),     'lr': self.lr},
            ]
        else:
            param_groups = list(self.parameters())
        opt = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}


# ── helpers ───────────────────────────────────────────────────────────────────

def _next_run_name(base_dir: str, modality: str) -> str:
    """Returns next versioned run name: {modality}_v{N}_{YYYY-MM-DD}."""
    versions = []
    if os.path.isdir(base_dir):
        prefix = f'{modality}_v'
        for d in os.listdir(base_dir):
            if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d)):
                tail = d[len(prefix):]       # e.g. '3_2026-05-07'
                n_str = tail.split('_')[0]
                if n_str.isdigit():
                    versions.append(int(n_str))
    next_v = max(versions, default=0) + 1
    return f'{modality}_v{next_v}_{datetime.now().strftime("%Y-%m-%d")}'


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality',      choices=['spec', 'eeg'], default="eeg")
    parser.add_argument('--backbone',      type=str,   default='efficientnet_b0',
                        help='timm backbone: efficientnet_b0 / efficientnet_b2 / efficientnet_b4')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--drop_rate',     type=float, default=0.3)
    parser.add_argument('--weight_decay',  type=float, default=3e-4)
    parser.add_argument('--label_smooth',  type=float, default=0.1,
                        help='label smoothing alpha (0 = disabled)')
    parser.add_argument('--mixup_alpha',   type=float, default=0.4,
                        help='MixUp Beta alpha (0 = disabled)')
    parser.add_argument('--freeze_epochs', type=int,   default=5,
                        help='freeze backbone for first N epochs, then unfreeze (0 = no freeze)')
    parser.add_argument('--min_votes',     type=int,   default=10,
                        help='drop samples with total votes below this threshold (0 = keep all)')
    parser.add_argument('--num_workers',   type=int,   default=0)
    parser.add_argument('--train_size',    type=int,   default=None)
    parser.add_argument('--val_size',      type=int,   default=None)
    args = parser.parse_args()

    # ── data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    if args.min_votes > 0:
        total_votes = df[VOTE_COLS].sum(axis=1)
        before = len(df)
        df = df[total_votes >= args.min_votes].reset_index(drop=True)
        print(f'vote filter >= {args.min_votes}: {before} → {len(df)} samples ({before - len(df)} removed)')

    if args.modality == 'spec':
        df = df.drop_duplicates(subset=['spectrogram_id', 'spectrogram_sub_id']).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=['eeg_id', 'eeg_sub_id']).reset_index(drop=True)

    # StratifiedGroupKFold: балансує класи + не допускає витоку пацієнтів
    # n_splits=5 → fold розміром 20% → рівно 80/20 split
    dominant = df[VOTE_COLS].values.argmax(axis=1)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df, dominant, groups=df['patient_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    if args.train_size:
        train_df = train_df.iloc[:args.train_size]
    if args.val_size:
        val_df = val_df.iloc[:args.val_size]

    LABEL_NAMES = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    train_dist = pd.Series(train_df[VOTE_COLS].values.argmax(axis=1)).value_counts().sort_index()
    val_dist   = pd.Series(val_df[VOTE_COLS].values.argmax(axis=1)).value_counts().sort_index()
    print(f'train: {len(train_df)}  val: {len(val_df)}')
    print('class distribution (dominant label):')
    print(f'  {"class":<10}' + '  '.join(f'{n:<8}' for n in LABEL_NAMES))
    print(f'  {"train":<10}' + '  '.join(f'{train_dist.get(i, 0):<8}' for i in range(6)))
    print(f'  {"val":<10}' + '  '.join(f'{val_dist.get(i, 0):<8}' for i in range(6)))

    if args.modality == 'spec':
        spec_dir  = os.path.join(DATA_DIR, 'train_spectrograms')
        train_ds  = HMSSpectrogramDataset(train_df, spec_dir, augment=True)
        val_ds    = HMSSpectrogramDataset(val_df,   spec_dir, augment=False)
        model     = SpectrogramModel(pretrained=True, drop_rate=args.drop_rate, backbone=args.backbone)
    else:
        eeg_dir   = os.path.join(DATA_DIR, 'train_eegs')
        cache_dir = CACHE_DIR if os.path.isdir(CACHE_DIR) else None
        if cache_dir:
            print(f'EEG cache знайдено: {cache_dir}')
        else:
            print('EEG cache не знайдено — запустіть src/precompute.py для прискорення')
        train_ds = HMSEEGDataset(train_df, eeg_dir, augment=True,  cache_dir=cache_dir)
        val_ds   = HMSEEGDataset(val_df,   eeg_dir, augment=False, cache_dir=cache_dir)
        model    = EEGModel(drop_rate=args.drop_rate, backbone=args.backbone)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    # ── run directories ───────────────────────────────────────────────────────
    run_name = _next_run_name(LOGS_DIR, args.modality)
    logs_run = os.path.join(LOGS_DIR, run_name)
    ckpt_run = os.path.join(CKPT_DIR, run_name)
    os.makedirs(logs_run, exist_ok=True)
    os.makedirs(ckpt_run, exist_ok=True)
    print(f'run             : {run_name}')

    # ── Lightning module + callbacks ──────────────────────────────────────────
    lit = HMSModule(model, lr=args.lr, freeze_epochs=args.freeze_epochs,
                    weight_decay=args.weight_decay, label_smooth=args.label_smooth,
                    mixup_alpha=args.mixup_alpha)

    callbacks = [
        EpochMetricsLogger(log_path=os.path.join(logs_run, 'metrics.csv')),
        PredictionLogger(log_path=os.path.join(logs_run, 'preds.csv')),
        ModelCheckpoint(
            dirpath=ckpt_run,
            filename='best',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=False,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            mode='min',
            verbose=False,
        ),
    ]

    # ── accelerator ───────────────────────────────────────────────────────────
    device = get_device()
    if device.type == 'mps':
        accelerator, devices = 'mps', 1
    elif device.type == 'cuda':
        accelerator, devices = 'gpu', 1
    else:
        accelerator, devices = 'cpu', 1

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=False,
        log_every_n_steps=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, train_loader, val_loader)

    ckpt_cb      = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    metrics_cb   = next(cb for cb in callbacks if isinstance(cb, EpochMetricsLogger))
    best = ckpt_cb.best_model_path

    summary = {
        'modality':      args.modality,
        'run':           run_name,
        'best_epoch':    metrics_cb.best_epoch,
        'best_val_loss': round(metrics_cb.best_val_loss, 6),
        'total_epochs':  trainer.current_epoch + 1,
        'stopped_early': trainer.current_epoch + 1 < args.epochs,
        'checkpoint':    best,
    }
    summary_path = os.path.join(logs_run, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nbest checkpoint : {best}')
    print(f'best epoch      : {metrics_cb.best_epoch}  (val_loss={metrics_cb.best_val_loss:.6f})')
    print(f'logs            : {logs_run}/')
    print(f'run summary     : {summary_path}')


if __name__ == '__main__':
    main()
