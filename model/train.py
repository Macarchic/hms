import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model.logger import get_logger
from model.dataset import VOTE_COLS, CLASS_NAMES, build_df_unique, make_folds, EEGDataset

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
CKPT_ROOT = ROOT_DIR / 'checkpoints'
LOGS_ROOT = ROOT_DIR / 'logs'


def next_version() -> int:
    existing = [d for d in CKPT_ROOT.glob('model_v*') if d.is_dir()]
    if not existing:
        return 1
    return max(int(d.name[7:]) for d in existing) + 1


class EEGModel(nn.Module):
    def __init__(self, backbone: str, dropout: float):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.num_features, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.drop(x)
        return F.log_softmax(self.fc(x), dim=1)  # (B, 6) log-probs


def build_model(backbone: str, dropout: float = 0.5) -> nn.Module:
    return EEGModel(backbone, dropout)


def kldiv_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.kl_div(log_probs, targets, reduction='batchmean')


def run_epoch(model, loader, optimizer, device, train: bool,
              desc: str = '') -> float:
    model.train(train)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with ctx:
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)
            log_probs = model(imgs)
            loss = kldiv_loss(log_probs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            n += len(imgs)
            bar.set_postfix(loss=f'{total_loss / n:.4f}')
    return total_loss / n


def make_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    counts = train_df['expert_consensus'].value_counts()
    weights = train_df['expert_consensus'].map(lambda c: 1.0 / counts[c]).to_numpy(copy=True)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def predict_val(model, val_df: pd.DataFrame, eeg_dir: Path,
                batch_size: int, device, fold: int, epoch: int) -> pd.DataFrame:
    ds = EEGDataset(val_df, eeg_dir, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            all_probs.append(model(imgs.to(device)).exp().cpu().numpy())
    probs = np.vstack(all_probs)

    result = val_df[['eeg_id'] + VOTE_COLS].copy().reset_index(drop=True)
    result.columns = ['eeg_id'] + [f'true_{c}' for c in CLASS_NAMES]
    for i, c in enumerate(CLASS_NAMES):
        result[f'pred_{c}'] = probs[:, i]
    result.insert(0, 'epoch', epoch)
    result.insert(0, 'fold', fold)
    return result


def train_fold(args, df: pd.DataFrame, fold: int, device,
               ckpt_dir: Path, log_dir: Path, log) -> float:
    train_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)
    eeg_dir = DATA_DIR / 'train_eegs'

    train_ds = EEGDataset(train_df, eeg_dir, augment=True)
    val_ds = EEGDataset(val_df, eeg_dir, augment=False)
    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, args.dropout).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr / 10},
        {'params': list(model.drop.parameters()) + list(model.fc.parameters()), 'lr': args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    best_epoch = 0
    best_ckpt = None
    patience_count = 0

    metrics_path = log_dir / 'metrics.csv'
    write_header = not metrics_path.exists()
    metrics_file = metrics_path.open('a', newline='')
    writer = csv.DictWriter(metrics_file, fieldnames=['fold', 'epoch', 'train_loss', 'val_loss'])
    if write_header:
        writer.writeheader()

    log.info(f'fold {fold} | train={len(train_df)} val={len(val_df)}')

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                               desc=f'f{fold} e{epoch:03d} train')
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                             desc=f'f{fold} e{epoch:03d} val  ')
        scheduler.step()

        writer.writerow({'fold': fold, 'epoch': epoch,
                         'train_loss': round(train_loss, 6),
                         'val_loss': round(val_loss, 6)})
        metrics_file.flush()

        preds_df = predict_val(model, val_df, eeg_dir, args.batch_size, device, fold, epoch)
        preds_path = log_dir / 'preds.csv'
        preds_df.to_csv(preds_path, mode='a', header=not preds_path.exists(), index=False)

        improved = val_loss < best_val
        if improved:
            if best_ckpt and best_ckpt.exists():
                best_ckpt.unlink()
            best_val = val_loss
            best_epoch = epoch
            best_ckpt = ckpt_dir / f'fold{fold}_epoch{epoch:03d}_val{val_loss:.4f}.ckpt'
            torch.save(model.state_dict(), best_ckpt)
            patience_count = 0
        else:
            patience_count += 1

        log.info(
            f'fold {fold} | epoch {epoch:03d}/{args.epochs} '
            f'train={train_loss:.4f} val={val_loss:.4f} '
            f'best={best_val:.4f} {"★" if improved else ""}'
        )

        if patience_count >= args.patience:
            log.info(f'fold {fold} | early stopping at epoch {epoch}')
            break

    metrics_file.close()
    log.info(f'fold {fold} | best epoch={best_epoch} val={best_val:.4f} → {best_ckpt.name}')
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='convnext_atto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=None,
                        help='train single fold; omit to train all')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    version = f'model_v{next_version()}'
    ckpt_dir = CKPT_ROOT / version
    log_dir = LOGS_ROOT / version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(__name__, str(log_dir))
    log.info(f'version={version}  backbone={args.backbone}')

    device = (
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('cpu')
    )
    log.info(f'device={device}')

    df = build_df_unique(DATA_DIR / 'train.csv')
    df = make_folds(df, n_splits=args.n_folds, seed=args.seed)

    folds = [args.fold] if args.fold is not None else list(range(args.n_folds))
    scores = []
    for fold in folds:
        score = train_fold(args, df, fold, device, ckpt_dir, log_dir, log)
        scores.append(score)

    log.info(f'CV: {[round(s, 4) for s in scores]}  mean={np.mean(scores):.4f}')


if __name__ == '__main__':
    main()
