"""
Raw-EEG dataset (no STFT).

Approach mirrors the 2nd-place HMS solution:
  16 bipolar leads × 2000 samples (10 s @ 200 Hz)
  → reshape (16, 2000) → (1, 160, 200)
    each row = 1 second of one lead
  → single-channel image fed into EfficientNet

Usage:
    from raw.dataset import HMSRawEEGDataset
"""

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset

VOTE_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

_EEG_ALL_COLS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
                 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

EEG_FS = 200
EEG_SECONDS = 10
EEG_SAMPLES = EEG_FS * EEG_SECONDS  # 2000

_LL = ['Fp1', 'F7', 'T3', 'T5', 'O1']
_RL = ['Fp2', 'F8', 'T4', 'T6', 'O2']
_LP = ['Fp1', 'F3', 'C3', 'P3', 'O1']
_RP = ['Fp2', 'F4', 'C4', 'P4', 'O2']
_CHAINS = [_LL, _RL, _LP, _RP]

N_LEADS = 16
# Image layout: (1, 160, 200)
#   rows = N_LEADS * EEG_SECONDS = 16 leads × 10 one-second rows per lead
#   cols = EEG_FS = 200 samples per second
IMG_H = N_LEADS * EEG_SECONDS  # 160
IMG_W = EEG_FS                  # 200

_BUTTER_SOS = butter(N=4, Wn=[0.5, 40.0], btype='bandpass', fs=EEG_FS, output='sos')


def _soft_label(row: pd.Series) -> np.ndarray:
    votes = row[VOTE_COLS].values.astype(np.float32)
    total = votes.sum()
    if total == 0:
        return np.ones(6, dtype=np.float32) / 6
    return votes / total


class HMSRawEEGDataset(Dataset):
    """
    Returns:
        img   : float32 tensor (1, 160, 200)
                row layout: leads 0-15, each split into 10 one-second rows
        label : float32 tensor (6,)
    """

    def __init__(self, df: pd.DataFrame, eeg_dir: str, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = int(row['eeg_id'])

        raw = pd.read_parquet(f'{self.eeg_dir}/{eeg_id}.parquet')
        data = raw[_EEG_ALL_COLS].values.astype(np.float32)  # (N, 20)
        col_idx = {c: i for i, c in enumerate(_EEG_ALL_COLS)}
        file_len = len(data)

        # 10-second window centered on labeled segment
        offset_sec = float(row.get('eeg_label_offset_seconds', 0))
        center = int((offset_sec + 5.0) * EEG_FS)
        start = center - EEG_SAMPLES // 2
        end = start + EEG_SAMPLES
        if start < 0:
            start, end = 0, EEG_SAMPLES
        elif end > file_len:
            start = max(0, file_len - EEG_SAMPLES)
            end = file_len
        segment = data[start:end]

        if len(segment) < EEG_SAMPLES:
            pad = np.zeros((EEG_SAMPLES - len(segment), segment.shape[1]), dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=0)

        # 16 bipolar derivations
        leads = []
        for chain in _CHAINS:
            for a, b in zip(chain[:-1], chain[1:]):
                leads.append(segment[:, col_idx[a]] - segment[:, col_idx[b]])
        eeg = np.stack(leads, axis=0)  # (16, 2000)

        # NaN fill → clip → bandpass → z-score per lead
        for i in range(eeg.shape[0]):
            ch = eeg[i]
            nan_mask = np.isnan(ch)
            if nan_mask.any():
                ch[nan_mask] = ch[~nan_mask].mean() if (~nan_mask).any() else 0.0
        np.clip(eeg, -1024.0, 1024.0, out=eeg)
        eeg = sosfilt(_BUTTER_SOS, eeg, axis=1).astype(np.float32)
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-6
        eeg = (eeg - mean) / std  # (16, 2000)

        if self.augment:
            eeg = _augment(eeg)

        # (16, 2000) → (1, 160, 200): contiguous reshape preserves temporal order
        # rows 0-9 = lead 0 (second 0..9), rows 10-19 = lead 1, etc.
        img = eeg.reshape(IMG_H, IMG_W)[np.newaxis]  # (1, 160, 200)

        label = _soft_label(row)
        return torch.from_numpy(img), torch.from_numpy(label)


def _augment(eeg: np.ndarray) -> np.ndarray:
    # eeg: (16, 2000), raw normalised signal before reshape

    # random polarity flip per lead — bipolar derivation polarity is arbitrary
    if np.random.rand() < 0.5:
        signs = np.random.choice([-1.0, 1.0], size=(eeg.shape[0], 1)).astype(np.float32)
        eeg = eeg * signs

    # time reversal — LPD/GPD are symmetric periodic patterns
    if np.random.rand() < 0.5:
        eeg = eeg[:, ::-1].copy()

    # amplitude jitter — simulates electrode impedance variation
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.7, 1.3)
        eeg = (eeg * scale).astype(np.float32)

    # additive Gaussian noise
    if np.random.rand() < 0.5:
        eeg = eeg + (np.random.randn(*eeg.shape) * 0.02).astype(np.float32)

    # channel dropout — one lead zeroed (e.g. bad electrode)
    if np.random.rand() < 0.3:
        ch = np.random.randint(0, eeg.shape[0])
        eeg[ch] = 0.0

    return eeg
