import timm
import torch
import torch.nn as nn


class SpectrogramModel(nn.Module):
    """
    EfficientNet-B0 on 4-channel spectrogram input.

    Input:  (B, 4, 100, 300)
    Output: (B, 6) logits
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, drop_rate: float = 0.2,
                 backbone: str = 'efficientnet_b0'):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=4,
            num_classes=0,
            global_pool='avg',
        )
        n_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 100, 300)
        features = self.backbone(x)   # (B, 1280)
        return self.head(features)    # (B, 6)
