"""XYMasking and Mixup augmentations for 2D EEG images."""
import random

import torch


def xy_masking(
    image: torch.Tensor,
    num_masks_x: int = 2,
    num_masks_y: int = 2,
    mask_ratio_x: float = 0.1,
    mask_ratio_y: float = 0.1,
) -> torch.Tensor:
    """
    Zero out random horizontal (time) and vertical (channel/freq) strips.
    image: (C, H, W)
    """
    image = image.clone()
    _, H, W = image.shape
    sx = max(1, int(W * mask_ratio_x))
    sy = max(1, int(H * mask_ratio_y))
    for _ in range(num_masks_x):
        x0 = random.randint(0, max(0, W - sx))
        image[:, :, x0:x0 + sx] = 0.0
    for _ in range(num_masks_y):
        y0 = random.randint(0, max(0, H - sy))
        image[:, y0:y0 + sy, :] = 0.0
    return image


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup within a batch. Shuffle samples and mix with random lambda.
    x: (B, C, H, W)
    y: (B, n_classes) soft labels
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix
