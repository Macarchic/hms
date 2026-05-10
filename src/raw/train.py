"""
Raw-EEG training script (no STFT, no spectrogram).

Usage (from project root):
    python src/raw/train.py
    python src/raw/train.py --backbone efficientnet_b2 --epochs 30
    python src/raw/train.py --backbone efficientnet_b5 --batch_size 8 --epochs 50
    python src/raw/train.py --epochs 1 --batch_size 4   # sanity check
"""

import argparse
import json
import os
import sys

# make src/ importable for callbacks / utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from datetime import datetime

from callbacks import EpochMetricsLogger, PredictionLogger
from utils import get_device, DATA_DIR, CKPT_DIR, LOGS_DIR
from raw.dataset import HMSRawEEGDataset, VOTE_COLS
from raw.model import RawEEGModel


class RawEEGModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float,
                 freeze_epochs: int = 3,
                 weight_decay: float = 1e-4,
                 label_smooth: float = 0.05,
                 mixup_alpha: float = 0.4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.freeze_epochs = freeze_epochs
        self.weight_decay = weight_decay
        self.label_smooth = label_smooth
        self.mixup_alpha = mixup_alpha
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
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.model, 'backbone') and hasattr(self.model, 'head'):
            param_groups = [
                {'params': self.model.backbone.parameters(), 'lr': self.lr * 0.1},
                {'params': [*self.model.avg_pool.parameters(),
                             *self.model.max_pool.parameters(),
                             *self.model.head.parameters()], 'lr': self.lr},
            ]
        else:
            param_groups = list(self.parameters())
        opt = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}


def _next_run_name(base_dir: str) -> str:
    versions = []
    if os.path.isdir(base_dir):
        prefix = 'raw_v'
        for d in os.listdir(base_dir):
            if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d)):
                tail = d[len(prefix):]
                n_str = tail.split('_')[0]
                if n_str.isdigit():
                    versions.append(int(n_str))
    next_v = max(versions, default=0) + 1
    return f'raw_v{next_v}_{datetime.now().strftime("%Y-%m-%d")}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='efficientnet_b2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smooth', type=float, default=0.05)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--freeze_epochs', type=int, default=0)
    parser.add_argument('--min_votes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    if args.min_votes > 0:
        total_votes = df[VOTE_COLS].sum(axis=1)
        before = len(df)
        df = df[total_votes >= args.min_votes].reset_index(drop=True)
        print(f'vote filter >= {args.min_votes}: {before} → {len(df)} rows')

    df = df.drop_duplicates(subset=['eeg_id', 'eeg_sub_id']).reset_index(drop=True)

    dominant = df[VOTE_COLS].values.argmax(axis=1)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df, dominant, groups=df['patient_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    if args.train_size:
        train_df = train_df.iloc[:args.train_size]
    if args.val_size:
        val_df = val_df.iloc[:args.val_size]

    LABEL_NAMES = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    train_dist = pd.Series(train_df[VOTE_COLS].values.argmax(axis=1)).value_counts().sort_index()
    val_dist = pd.Series(val_df[VOTE_COLS].values.argmax(axis=1)).value_counts().sort_index()
    print(f'train: {len(train_df)}  val: {len(val_df)}')
    print('class distribution (dominant label):')
    print(f'  {"class":<10}' + '  '.join(f'{n:<8}' for n in LABEL_NAMES))
    print(f'  {"train":<10}' + '  '.join(f'{train_dist.get(i, 0):<8}' for i in range(6)))
    print(f'  {"val":<10}' + '  '.join(f'{val_dist.get(i, 0):<8}' for i in range(6)))

    eeg_dir = os.path.join(DATA_DIR, 'train_eegs')
    train_ds = HMSRawEEGDataset(train_df, eeg_dir, augment=True)
    val_ds = HMSRawEEGDataset(val_df, eeg_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)

    run_name = _next_run_name(LOGS_DIR)
    logs_run = os.path.join(LOGS_DIR, run_name)
    ckpt_run = os.path.join(CKPT_DIR, run_name)
    os.makedirs(logs_run, exist_ok=True)
    os.makedirs(ckpt_run, exist_ok=True)
    print(f'run: {run_name}')

    model = RawEEGModel(drop_rate=args.drop_rate, backbone=args.backbone)
    lit = RawEEGModule(model, lr=args.lr, freeze_epochs=args.freeze_epochs,
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

    ckpt_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    metrics_cb = next(cb for cb in callbacks if isinstance(cb, EpochMetricsLogger))
    best = ckpt_cb.best_model_path

    summary = {
        'modality': 'raw',
        'backbone': args.backbone,
        'run': run_name,
        'best_epoch': metrics_cb.best_epoch,
        'best_val_loss': round(metrics_cb.best_val_loss, 6),
        'total_epochs': trainer.current_epoch + 1,
        'stopped_early': trainer.current_epoch + 1 < args.epochs,
        'checkpoint': best,
    }
    summary_path = os.path.join(logs_run, 'run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nbest checkpoint : {best}')
    print(f'best epoch      : {metrics_cb.best_epoch}  (val_loss={metrics_cb.best_val_loss:.6f})')
    print(f'logs            : {logs_run}/')


if __name__ == '__main__':
    main()
