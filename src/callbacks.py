import csv
import os

import torch
import torch.nn.functional as F
import lightning as L

LABELS = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']


class EpochMetricsLogger(L.Callback):
    """Writes one clean row per epoch to {modality}_metrics.csv."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def on_fit_start(self, trainer, pl_module):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        m = trainer.callback_metrics
        param_groups = trainer.optimizers[0].param_groups
        lr_head = param_groups[-1]['lr']
        lr_bb   = param_groups[0]['lr'] if len(param_groups) > 1 else lr_head

        val_loss = m.get('val_loss')
        if val_loss is not None and val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.best_epoch = trainer.current_epoch

        row = {
            'epoch':      trainer.current_epoch,
            'train_loss': round(m['train_loss'].item(), 6) if 'train_loss' in m else '',
            'train_acc':  round(m['train_acc'].item(), 4)  if 'train_acc'  in m else '',
            'val_loss':   round(m['val_loss'].item(), 6)   if 'val_loss'   in m else '',
            'val_acc':    round(m['val_acc'].item(), 4)    if 'val_acc'    in m else '',
            'lr':         f'{lr_head:.8f}',
            'lr_bb':      f'{lr_bb:.8f}',
        }

        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'lr_bb']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class PredictionLogger(L.Callback):
    """
    Після кожної епохи зберігає в CSV:
      - середній передбачений розподіл по класах (pred_*)
      - середній реальний розподіл по класах (true_*)
      - точність по кожному класу (як часто модель правильно вибирає топ-клас)

    Файл: logs/{name}_preds.csv
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def on_fit_start(self, trainer, pl_module):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking:
            return
        x, y = batch
        with torch.no_grad():
            logits = pl_module(x.to(pl_module.device))
            probs = F.softmax(logits, dim=1).cpu()
        self._preds.append(probs)
        self._targets.append(y.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._preds:
            self._preds.clear()
            self._targets.clear()
            return

        epoch = trainer.current_epoch
        preds   = torch.cat(self._preds,   dim=0)  # (N, 6)
        targets = torch.cat(self._targets, dim=0)  # (N, 6)

        pred_mean   = preds.mean(dim=0)    # (6,) — середній передбачений розподіл
        target_mean = targets.mean(dim=0)  # (6,) — середній реальний розподіл

        # per-class accuracy: чи модель обрала той самий топ-клас що й мітка
        pred_cls   = preds.argmax(dim=1)
        target_cls = targets.argmax(dim=1)
        per_class_acc = {}
        for i, label in enumerate(LABELS):
            mask = target_cls == i
            if mask.sum() > 0:
                per_class_acc[label] = (pred_cls[mask] == i).float().mean().item()
            else:
                per_class_acc[label] = float('nan')

        row = {'epoch': epoch}
        for i, label in enumerate(LABELS):
            row[f'pred_{label}']   = round(pred_mean[i].item(), 4)
            row[f'true_{label}']   = round(target_mean[i].item(), 4)
            row[f'acc_{label}']    = round(per_class_acc[label], 4) if per_class_acc[label] == per_class_acc[label] else ''

        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # коротке резюме в термінал
        pred_str = '  '.join(f'{l}={v:.3f}' for l, v in zip(LABELS, pred_mean.tolist()))
        true_str = '  '.join(f'{l}={v:.3f}' for l, v in zip(LABELS, target_mean.tolist()))
        print(f'\n  [epoch {epoch}] pred:   {pred_str}')
        print(f'  [epoch {epoch}] target: {true_str}')

        self._preds.clear()
        self._targets.clear()
