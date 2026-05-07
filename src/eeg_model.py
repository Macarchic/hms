import timm
import torch
import torch.nn as nn


class EEGModel(nn.Module):
    """
    EfficientNet-B0 на STFT-спектрограмах з 16 біполярних leads.

    Input:  (B, 16, F, T) — 16 leads × 13 freq bins × 198 time steps
    Output: (B, 6) logits

    Reshape: (B, 16, F, T) → (B, 3, 16*F, T) — стек leads по висоті,
    трипліковано по каналах для ImageNet-backbone.
    """

    def __init__(self, num_classes: int = 6, drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True, in_chans=3,
            num_classes=0, global_pool='',
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape               # (B, 16, 13, 198)
        x = x.reshape(B, C * F, T)         # (B, 208, 198)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, 208, 198)

        x = self.backbone(x)               # (B, 1280, h, w)
        x = self.pool(x).flatten(1)        # (B, 1280)
        return self.head(x)                # (B, 6)
