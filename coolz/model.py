import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    2D-CNN on raw EEG reshaped into a (3, 160, 1000) image.

    Input:  (B, 16, 10000)  — 16-channel double-banana, 50 sec @ 200 Hz
    Reshape:
        view(B, 16, 1000, 10)          # 1000 "macro" steps × 10 micro-steps
        permute(0, 1, 3, 2)            # → (B, 16, 10, 1000)
        reshape(B, 16*10=160, 1000)    # channel × micro-step → "height"
        unsqueeze(1).expand(3, …)      # replicate to 3-channel image
    → (B, 3, 160, 1000) into any timm 2D backbone
    """

    def __init__(
        self,
        backbone: str = 'efficientnet_b5',
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.net = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool='',
        )
        n_feat = self.net.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_feat, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, 16, 1000, 10).permute(0, 1, 3, 2).reshape(B, 160, 1000)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        x = self.net(x)
        x = self.pool(x).view(B, -1)
        x = self.drop(x)
        return F.log_softmax(self.fc(x), dim=1)
