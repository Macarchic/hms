import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.kfuji.config import KfujiConfig
from adan_pytorch import Adan


class KfujiModule(L.LightningModule):
    """
    MaxViT fine-tuned on stacked CWT scalograms.
    Loss: KLDivLoss (matches competition metric).
    Optimizer: Adan.
    """

    def __init__(self, config: KfujiConfig):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _step(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(F.log_softmax(logits, dim=1), y)
        acc = (logits.argmax(1) == y.argmax(1)).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = Adan(self.parameters(), lr=self.cfg.lr)
        sched = CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]
