"""Paul wavelet CWT for EEG scalogram computation.

Reference: kfuji's HMS solution, following suguuuuu's G2Net CWT approach.
Paul wavelet Fourier transform:
    ψ̂(ω) = H(ω) · (2^m / sqrt(m·(2m)!)) · ω^m · exp(-ω)
Center frequency: f = m / (2π·s)  →  scale s = m / (2π·f)
"""
from math import factorial, sqrt

import numpy as np
import torch


def build_freqs(lower: float, upper: float, n: int) -> np.ndarray:
    """Log-spaced frequency array from lower to upper Hz."""
    return np.geomspace(lower, upper, n)


def make_psi_matrix(
    freqs: np.ndarray,
    n_samples: int,
    fs: float,
    m: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Precompute (n_scales, N) Paul wavelet FT matrix for batched CWT.
    Call once per dataset, reuse for every signal.
    """
    norm = (2**m) / sqrt(m * factorial(2 * m))
    omega = 2 * torch.pi * torch.fft.fftfreq(n_samples, d=1.0 / fs).to(device)
    scales = torch.from_numpy(m / (2 * torch.pi * freqs)).float().to(device)
    scaled_omega = scales.unsqueeze(1) * omega.unsqueeze(0)  # (n_scales, N)
    psi = torch.zeros_like(scaled_omega)
    mask = scaled_omega > 0
    so = scaled_omega[mask]
    psi[mask] = norm * so.pow(m) * torch.exp(-so)
    return psi  # (n_scales, N), real


def paul_scalogram(
    x: torch.Tensor,
    psi: torch.Tensor,
    stride: int = 16,
    border_crop: int = 1,
) -> torch.Tensor:
    """
    Compute CWT power scalogram for one 1-D signal.

    x:    (N,)           signal, float32
    psi:  (n_scales, N)  precomputed Paul wavelet FT matrix (from make_psi_matrix)
    returns: (n_scales, T) instantaneous power, T ≈ N // stride
    """
    x_ft = torch.fft.fft(x.float())
    W = torch.fft.ifft(psi * x_ft.unsqueeze(0), dim=1)  # (n_scales, N) complex
    power = W.real.pow(2) + W.imag.pow(2)               # (n_scales, N)
    power = power[:, ::stride]
    if border_crop > 0 and power.shape[1] > 2 * border_crop:
        power = power[:, border_crop:-border_crop]
    return power  # (n_scales, T)
