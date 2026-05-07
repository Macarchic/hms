import timm
import torch
import torch.nn as nn


class SpectrogramModel(nn.Module):
    """
    EfficientNet-B0 on 4-channel spectrogram input.

    Input:  (B, 4, 100, 300)
    Output: (B, 6) logits
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            in_chans=4,        # 4 EEG chain groups instead of 3 RGB
            num_classes=0,     # remove default classifier
            global_pool='avg',
        )
        n_features = self.backbone.num_features  # 1280 for B0

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 100, 300)
        features = self.backbone(x)   # (B, 1280)
        return self.head(features)    # (B, 6)
