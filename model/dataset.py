import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

FS = 200
WIN_SAMPLES = 10_000
TARGET_SIZE = 512
CROP_LENGTHS = [2000, 5000, 10_000]  # each crop → one RGB channel
BANDPASS_LO = 0.5
BANDPASS_HI = 40.0
VOTE_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
CLASS_NAMES = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
NEEDED_COLS = list(dict.fromkeys(c for pair in BIPOLAR_PAIRS for c in pair))
LABEL_SMOOTHING = 0.02


def build_df_unique(csv_path: str | Path) -> pd.DataFrame:
    """
    Aggregate train.csv to one row per eeg_id.
    offset           = median sub_id offset (central window)
    vote cols        = mean votes normalised to sum=1 (soft labels)
    expert_consensus = argmax of soft labels
    """
    df = pd.read_csv(csv_path)
    df_unique = (
        df.groupby('eeg_id', sort=False)
        .agg(
            patient_id=('patient_id', 'first'),
            eeg_label_offset_seconds=('eeg_label_offset_seconds', 'median'),
            seizure_vote=('seizure_vote', 'mean'),
            lpd_vote=('lpd_vote', 'mean'),
            gpd_vote=('gpd_vote', 'mean'),
            lrda_vote=('lrda_vote', 'mean'),
            grda_vote=('grda_vote', 'mean'),
            other_vote=('other_vote', 'mean'),
            n_subs=('eeg_sub_id', 'count'),
        )
        .reset_index()
    )
    totals = df_unique[VOTE_COLS].sum(axis=1)
    df_unique[VOTE_COLS] = df_unique[VOTE_COLS].div(totals, axis=0)
    df_unique['expert_consensus'] = (
        df_unique[VOTE_COLS].values.argmax(axis=1)
    )
    df_unique['expert_consensus'] = df_unique['expert_consensus'].map(
        dict(enumerate(CLASS_NAMES))
    )
    return df_unique


def build_df_train(csv_path: str | Path) -> pd.DataFrame:
    """Raw train.csv rows with per-row normalised soft labels."""
    df = pd.read_csv(csv_path)
    totals = df[VOTE_COLS].sum(axis=1)
    df[VOTE_COLS] = df[VOTE_COLS].div(totals, axis=0)
    return df


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Add a 'fold' column (0..n_splits-1) using StratifiedGroupKFold.
    Groups = patient_id  →  no patient appears in both train and val.
    Stratify = expert_consensus  →  balanced class distribution per fold.
    """
    df = df.copy()
    df['fold'] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, y=df['expert_consensus'], groups=df['patient_id'])
    ):
        df.loc[val_idx, 'fold'] = fold
    return df


def _load_eeg_window(eeg_id: int, offset_sec: float, eeg_dir: Path) -> np.ndarray:
    raw = pq.read_table(eeg_dir / f'{eeg_id}.parquet', columns=NEEDED_COLS).to_pandas()
    start = int(offset_sec * FS)
    window = raw.iloc[start: start + WIN_SAMPLES]
    window = window.interpolate(axis=0, limit_direction='both').fillna(0)
    return window.values.T.astype(np.float32)  # (len(NEEDED_COLS), 10000)


def _bipolar_montage(eeg: np.ndarray, columns: list) -> np.ndarray:
    ch_idx = {c: i for i, c in enumerate(columns)}
    out = np.zeros((18, eeg.shape[1]), dtype=np.float32)
    for k, (a, b) in enumerate(BIPOLAR_PAIRS):
        out[k] = eeg[ch_idx[a]] - eeg[ch_idx[b]]
    return out


def _bandpass(signals: np.ndarray) -> np.ndarray:
    sos = butter(5, [BANDPASS_LO, BANDPASS_HI], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, signals, axis=-1).astype(np.float32)


def _signals_to_image(signals: np.ndarray, center: int | None = None) -> np.ndarray:
    """
    Map each crop length to one RGB channel → (3, 512, 512).
    crop 2000  → channel 0  (fine detail,  10 sec)
    crop 5000  → channel 1  (mid scale,    25 sec)
    crop 10000 → channel 2  (full window,  50 sec)
    `center` pins the temporal anchor for ch0/ch1 (default = middle sample).
    """
    T = signals.shape[1]
    if center is None:
        center = T // 2
    channels = []
    for crop_len in CROP_LENGTHS:
        if crop_len >= T:
            crop = signals
        else:
            start = np.clip(center - crop_len // 2, 0, T - crop_len)
            crop = signals[:, start: start + crop_len]
        ch = cv2.resize(
            crop.astype(np.float32),
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        channels.append(ch)
    img = np.stack(channels)  # (3, 512, 512)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img


def _xy_masking(img: np.ndarray, num_x: int = 2, num_y: int = 2,
                ratio_x: float = 0.08, ratio_y: float = 0.08) -> np.ndarray:
    img = img.copy()
    _, H, W = img.shape
    sw = max(1, int(W * ratio_x))
    sh = max(1, int(H * ratio_y))
    for _ in range(num_x):
        x0 = np.random.randint(0, max(1, W - sw))
        img[:, :, x0:x0 + sw] = 0.0
    for _ in range(num_y):
        y0 = np.random.randint(0, max(1, H - sh))
        img[:, y0:y0 + sh, :] = 0.0
    return img


class EEGDataset(Dataset):
    """
    Returns (image, label):
      image : float32 tensor (3, 512, 512)
                ch0 = crop 2000 samples  (fine)
                ch1 = crop 5000 samples  (mid)
                ch2 = crop 10000 samples (full)
      label : float32 tensor (6,) — soft label probability distribution
    """

    def __init__(self, df: pd.DataFrame, eeg_dir: str | Path, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.eeg_dir = Path(eeg_dir)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        eeg = _load_eeg_window(int(row['eeg_id']), row['eeg_label_offset_seconds'], self.eeg_dir)
        bip = _bipolar_montage(eeg, NEEDED_COLS)
        bip = np.clip(bip, -1024.0, 1024.0) / 32.0
        filt = _bandpass(bip)

        if self.augment:
            # random temporal center within the valid range for all crop lengths
            max_crop = max(CROP_LENGTHS[:-1])  # 5000
            center = np.random.randint(max_crop // 2, WIN_SAMPLES - max_crop // 2)
            if np.random.rand() < 0.5:
                filt = filt[:, ::-1].copy()  # time reversal
        else:
            center = WIN_SAMPLES // 2

        img = _signals_to_image(filt, center=center)

        if self.augment:
            img = _xy_masking(img)

        label = row[VOTE_COLS].values.astype(np.float32)
        if self.augment:
            label += LABEL_SMOOTHING
            label /= label.sum()
        return torch.from_numpy(img), torch.from_numpy(label)
