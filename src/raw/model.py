import timm
import torch
import torch.nn as nn


class RawEEGModel(nn.Module):
    """
    EfficientNet on raw EEG reshaped as a single-channel image.

    Input:  (B, 1, 160, 200)
    Output: (B, 6) logits

    Feature map after 32× downsampling: ~(5, 6).
    Head uses avg-pool + max-pool concatenation for richer spatial summary.
    """

    def __init__(self, num_classes: int = 6, drop_rate: float = 0.3,
                 backbone: str = 'efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=1,
            num_classes=0, global_pool='',
        )
        n_features = self.backbone.num_features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(n_features * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                          # (B, C, h, w)
        avg = self.avg_pool(feat).flatten(1)             # (B, C)
        mx = self.max_pool(feat).flatten(1)              # (B, C)
        return self.head(torch.cat([avg, mx], dim=1))    # (B, 6)
