import timm
import torch
import torch.nn as nn


class EEGModel(nn.Module):
    """
    EfficientNet на STFT-спектрограмах з 16 біполярних leads.

    Input:  (B, 16, F, T) — 16 leads × 26 freq bins (0–40 Hz) × 395 time steps
    Output: (B, 6) logits

    Reshape: (B, 16, F, T) → (B, 1, 16*F, T) — стек leads по висоті, single channel.
    Feature map after 32× downsampling: ~(13, 12) for B0.
    """

    def __init__(self, num_classes: int = 6, drop_rate: float = 0.3,
                 backbone: str = 'efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=1,
            num_classes=0, global_pool='',
        )
        n_features = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape               # (B, 16, 13, 198)
        x = x.reshape(B, C * F, T)         # (B, 208, 198)
        x = x.unsqueeze(1)                 # (B, 1, 208, 198)

        x = self.backbone(x)               # (B, 1280, h, w)
        x = self.pool(x).flatten(1)        # (B, 1280)
        return self.head(x)                # (B, 6)
