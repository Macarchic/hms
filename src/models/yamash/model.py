import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from adan_pytorch import Adan

from src.models.yamash.augment import mixup_batch
from src.models.yamash.config import YamashConfig


def entmax(logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Entmax-alpha: p*_i = max(0, (α-1)*(z_i - τ))^{1/(α-1)}, τ found by bisection.
    alpha=1 → softmax, alpha→∞ → argmax.
    Computed in float64 to handle high exponents (e.g. 1/0.03 = 33).
    """
    if alpha <= 1.001:
        return torch.softmax(logits, dim=-1)

    a1 = alpha - 1.0
    exp = 1.0 / a1

    z = logits.double() - logits.max(dim=-1, keepdim=True).values.double()

    # lo: tau where all terms active → sum >> 1
    # hi: tau = z_max = 0 → all terms = 0 → sum = 0
    lo = z.min(dim=-1, keepdim=True).values - 1.0 / a1
    hi = z.max(dim=-1, keepdim=True).values  # = 0.0 after max-shift

    for _ in range(64):
        mid = (lo + hi) / 2
        p = (a1 * (z - mid)).clamp(min=0).pow(exp)
        s = p.sum(dim=-1, keepdim=True)
        lo = torch.where(s > 1.0, mid, lo)
        hi = torch.where(s <= 1.0, mid, hi)

    tau = (lo + hi) / 2
    p = (a1 * (z - tau)).clamp(min=0).pow(exp)
    return (p / (p.sum(dim=-1, keepdim=True) + 1e-12)).float()


class YamashModule(L.LightningModule):
    """
    Convnext-atto fine-tuned on stacked raw EEG crops (yamash approach).
    Training: log_softmax + KLDivLoss + Mixup.
    Inference: entmax for sharper predicted distributions.
    """

    def __init__(self, config: YamashConfig):
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

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss = self.loss_fn(F.log_softmax(logits, dim=1), y)
        acc = (logits.argmax(1) == y.argmax(1)).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = mixup_batch(x, y)
        logits = self(x)
        loss, acc = self._compute_loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, acc = self._compute_loss(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return entmax(self(x), self.cfg.entmax_alpha)

    def configure_optimizers(self):
        opt = Adan(self.parameters(), lr=self.cfg.lr)
        sched = CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]
