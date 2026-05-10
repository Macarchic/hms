import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfilt, stft as scipy_stft
from torch.utils.data import Dataset

VOTE_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

SPEC_GROUPS = ['LL', 'RL', 'LP', 'RP']
SPEC_FREQS = 100  # freq bins per group
SPEC_TIMES = 300  # time steps in full 10-min spectrogram

# All columns present in EEG parquet files
EEG_ALL_COLS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
                'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

EEG_FS = 200              # Hz
EEG_WINDOW_SAMPLES = 10_000  # 50 seconds at 200 Hz

# Bipolar montage chains: 4 chains × 5 electrodes → 4 differences each → 16 leads
_LL = ['Fp1', 'F7', 'T3', 'T5', 'O1']
_RL = ['Fp2', 'F8', 'T4', 'T6', 'O2']
_LP = ['Fp1', 'F3', 'C3', 'P3', 'O1']
_RP = ['Fp2', 'F4', 'C4', 'P4', 'O2']
_CHAINS = [_LL, _RL, _LP, _RP]

# Butterworth bandpass 0.5–40 Hz (includes beta band 13–30 Hz)
_BUTTER_SOS = butter(N=5, Wn=[0.5, 40.0], btype='bandpass', fs=EEG_FS, output='sos')

# STFT parameters
EEG_N_FFT = 128    # window = 128/200 = 640 ms
EEG_HOP = 25       # step   =  25/200 = 125 ms (doubled time resolution)

# Frequency axis of STFT and mask for 0–40 Hz
_STFT_FREQS = np.fft.rfftfreq(EEG_N_FFT, d=1.0 / EEG_FS)   # (65,)
_FREQ_MASK = _STFT_FREQS <= 40.0                              # delta/theta/alpha/beta
EEG_FREQ_BINS = int(_FREQ_MASK.sum())                         # 26 bins: 0–39 Hz
EEG_TIME_STEPS = (EEG_WINDOW_SAMPLES - EEG_N_FFT) // EEG_HOP + 1  # 395 steps


def _soft_label(row: pd.Series) -> np.ndarray:
    votes = row[VOTE_COLS].values.astype(np.float32)
    total = votes.sum()
    if total == 0:
        return np.ones(6, dtype=np.float32) / 6
    return votes / total


class HMSSpectrogramDataset(Dataset):
    """
    Returns:
        spec  : float32 tensor (4, 100, 300) — full 10-min spectrogram, 4 chain groups
        label : float32 tensor (6,)           — soft vote distribution
    """

    def __init__(self, df: pd.DataFrame, spec_dir: str, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.spec_dir = spec_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_id = int(row['spectrogram_id'])

        raw = pd.read_parquet(f'{self.spec_dir}/{spec_id}.parquet')
        arrays = []
        for g in SPEC_GROUPS:
            cols = [c for c in raw.columns if c.startswith(f'{g}_')]
            arr = raw[cols].values.T.astype(np.float32)  # (100, T)
            arrays.append(arr)
        spec = np.stack(arrays, axis=0)  # (4, 100, T)

        # each time step = 2 sec; extract SPEC_TIMES steps starting at label offset
        offset_sec = float(row.get('spectrogram_label_offset_seconds', 0))
        start = int(offset_sec / 2)
        t = spec.shape[2]
        start = min(start, max(0, t - SPEC_TIMES))
        end = start + SPEC_TIMES
        if end <= t:
            spec = spec[:, :, start:end]
        else:
            chunk = spec[:, :, start:t]
            pad = np.zeros((spec.shape[0], spec.shape[1], SPEC_TIMES - chunk.shape[2]), dtype=np.float32)
            spec = np.concatenate([chunk, pad], axis=2)

        np.nan_to_num(spec, copy=False)

        if self.augment:
            spec = _augment_spec(spec)

        label = _soft_label(row)
        return torch.from_numpy(spec), torch.from_numpy(label)


class HMSEEGDataset(Dataset):
    """
    Returns:
        spec  : float32 tensor (16, EEG_FREQ_BINS, EEG_TIME_STEPS)
                16 bipolar leads × 26 freq bins (0–40 Hz) × 395 time steps
        label : float32 tensor (6,)

    cache_dir: якщо вказано і файл {eeg_id}_{eeg_sub_id}.npy існує —
               читає готовий STFT з диску (швидко) замість повного pipeline.
               Запустіть src/precompute.py щоб заповнити кеш.
    """

    def __init__(self, df: pd.DataFrame, eeg_dir: str, augment: bool = False,
                 cache_dir: str | None = None):
        self.df = df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.augment = augment
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = int(row['eeg_id'])
        sub_id = int(row.get('eeg_sub_id', 0))

        # ── швидкий шлях: читаємо з кешу ────────────────────────────────────
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, f'{eeg_id}_{sub_id}.npy')
            if os.path.exists(cache_file):
                spec = np.load(cache_file).astype(np.float32)
                if spec.shape == (16, EEG_FREQ_BINS, EEG_TIME_STEPS):
                    if self.augment:
                        spec = _augment_eeg(spec)
                    return torch.from_numpy(spec), torch.from_numpy(_soft_label(row))
                # shape mismatch → stale cache, fall through to recompute

        # ── повний pipeline якщо кешу немає ─────────────────────────────────
        raw = pd.read_parquet(f'{self.eeg_dir}/{eeg_id}.parquet')
        data = raw[EEG_ALL_COLS].values.astype(np.float32)  # (N, 20)
        col_idx = {c: i for i, c in enumerate(EEG_ALL_COLS)}

        # ── 50-second window centered on the labeled 10s segment ─────────────
        offset_sec = float(row.get('eeg_label_offset_seconds', 0))
        file_len = len(data)
        center = int((offset_sec + 5.0) * EEG_FS)
        start = center - EEG_WINDOW_SAMPLES // 2
        end = start + EEG_WINDOW_SAMPLES
        if start < 0:
            start, end = 0, EEG_WINDOW_SAMPLES
        elif end > file_len:
            start = max(0, file_len - EEG_WINDOW_SAMPLES)
            end = file_len
        segment = data[start:end]

        if len(segment) < EEG_WINDOW_SAMPLES:
            pad = np.zeros((EEG_WINDOW_SAMPLES - len(segment), segment.shape[1]),
                           dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=0)

        # ── brain leads: 16 bipolar derivations ─────────────────────────────
        leads = []
        for chain in _CHAINS:
            for a, b in zip(chain[:-1], chain[1:]):
                leads.append(segment[:, col_idx[a]] - segment[:, col_idx[b]])
        eeg = np.stack(leads, axis=0)   # (16, 10000)

        # ── NaN fill → clip → bandpass → z-score ────────────────────────────
        for i in range(eeg.shape[0]):
            ch = eeg[i]
            nan_mask = np.isnan(ch)
            if nan_mask.any():
                mean_val = ch[~nan_mask].mean() if (~nan_mask).any() else 0.0
                ch[nan_mask] = mean_val
        np.clip(eeg, -1024.0, 1024.0, out=eeg)
        eeg = sosfilt(_BUTTER_SOS, eeg, axis=1).astype(np.float32)
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-6
        eeg = (eeg - mean) / std

        # ── STFT → magnitude → log scale ────────────────────────────────────
        specs = []
        for ch in range(eeg.shape[0]):
            _, _, Zxx = scipy_stft(eeg[ch], fs=EEG_FS, nperseg=EEG_N_FFT,
                                   noverlap=EEG_N_FFT - EEG_HOP,
                                   boundary=None, padded=False)
            mag = np.abs(Zxx)[_FREQ_MASK, :EEG_TIME_STEPS]  # (26, 395)
            specs.append(mag)
        spec = np.stack(specs, axis=0).astype(np.float32)   # (16, 26, 395)
        spec = np.log1p(spec)

        if self.augment:
            spec = _augment_eeg(spec)

        label = _soft_label(row)
        return torch.from_numpy(spec), torch.from_numpy(label)


# ── augmentations ────────────────────────────────────────────────────────────

def _augment_spec(spec: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        spec = spec[:, :, ::-1].copy()
    if np.random.rand() < 0.5:
        f0 = np.random.randint(0, SPEC_FREQS - 10)
        spec[:, f0:f0 + np.random.randint(1, 11), :] = 0
    return spec


def _augment_eeg(spec: np.ndarray) -> np.ndarray:
    # spec: (16, F, T)
    # time mask — up to 20% of window
    if np.random.rand() < 0.5:
        t0 = np.random.randint(0, spec.shape[2])
        t_w = np.random.randint(1, max(2, spec.shape[2] // 5))
        spec[:, :, t0:t0 + t_w] = 0.0
    # freq mask
    if np.random.rand() < 0.5:
        f0 = np.random.randint(0, spec.shape[1])
        f_w = np.random.randint(1, max(2, spec.shape[1] // 3))
        spec[:, f0:f0 + f_w, :] = 0.0
    # channel dropout — zero out one random lead
    if np.random.rand() < 0.3:
        ch = np.random.randint(0, spec.shape[0])
        spec[ch] = 0.0
    return spec
