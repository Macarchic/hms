"""
Usage:
    python src/train.py --modality spec
    python src/train.py --modality eeg
    python src/train.py --modality eeg  --epochs 30 --batch_size 16 --lr 1e-3
    python src/train.py --modality spec --epochs 1  --batch_size 4             # sanity check
    python src/train.py --modality eeg  --train_size 500 --val_size 100        # subset
"""

import argparse
import json
import os

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
    def __init__(self, model: torch.nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.kl_div(F.log_softmax(logits, dim=1), y, reduction='batchmean')
        acc = (logits.argmax(1) == y.argmax(1)).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc',  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc',  acc,  on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality',    choices=['spec', 'eeg'], required=True)
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=0)
    parser.add_argument('--train_size',  type=int,   default=None)
    parser.add_argument('--val_size',    type=int,   default=None)
    args = parser.parse_args()

    # ── data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

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
        model     = SpectrogramModel(pretrained=True)
    else:
        eeg_dir   = os.path.join(DATA_DIR, 'train_eegs')
        cache_dir = CACHE_DIR if os.path.isdir(CACHE_DIR) else None
        if cache_dir:
            print(f'EEG cache знайдено: {cache_dir}')
        else:
            print('EEG cache не знайдено — запустіть src/precompute.py для прискорення')
        train_ds = HMSEEGDataset(train_df, eeg_dir, augment=True,  cache_dir=cache_dir)
        val_ds   = HMSEEGDataset(val_df,   eeg_dir, augment=False, cache_dir=cache_dir)
        model    = EEGModel()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    # ── Lightning module + callbacks ──────────────────────────────────────────
    lit = HMSModule(model, lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    preds_log   = os.path.join(LOGS_DIR, f'{args.modality}_preds.csv')
    metrics_log = os.path.join(LOGS_DIR, f'{args.modality}_metrics.csv')
    callbacks = [
        EpochMetricsLogger(log_path=metrics_log),
        PredictionLogger(log_path=preds_log),
        ModelCheckpoint(
            dirpath=CKPT_DIR,
            filename=f'{args.modality}_best',
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
    )

    trainer.fit(lit, train_loader, val_loader)

    ckpt_cb      = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    metrics_cb   = next(cb for cb in callbacks if isinstance(cb, EpochMetricsLogger))
    best = ckpt_cb.best_model_path

    summary = {
        'modality':            args.modality,
        'best_epoch':          metrics_cb.best_epoch,
        'best_val_loss':       round(metrics_cb.best_val_loss, 6),
        'total_epochs':        trainer.current_epoch + 1,
        'stopped_early':       trainer.current_epoch + 1 < args.epochs,
        'checkpoint':          best,
    }
    summary_path = os.path.join(LOGS_DIR, f'{args.modality}_run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nbest checkpoint : {best}')
    print(f'best epoch      : {metrics_cb.best_epoch}  (val_loss={metrics_cb.best_val_loss:.6f})')
    print(f'metrics         : {metrics_log}')
    print(f'run summary     : {summary_path}')


if __name__ == '__main__':
    main()
