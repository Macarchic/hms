import os
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.models.kfuji.config import KfujiConfig
from src.models.kfuji.cwt import build_freqs, make_psi_matrix, paul_scalogram

FS = 200
VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

# 18 bipolar pairs used by kfuji (raw EEG, no EKG)
BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]

# Unique EEG columns needed — deduped, preserving order
NEEDED_COLS = list(dict.fromkeys(c for pair in BIPOLAR_PAIRS for c in pair))


class HMSDataset(Dataset):
    """
    One item = one unique eeg_id.
    Training:   random sub_id per call  (augmentation via offset variety)
    Validation: always first sub_id     (deterministic)

    Filtering by vote count for 2-stage training is done externally:
        stage1_df = df[df[VOTE_COLS].sum(1) > 1]
        stage2_df = df[df[VOTE_COLS].sum(1) >= 10]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: str,
        config: KfujiConfig,
        training: bool = True,
    ):
        self.eeg_dir = eeg_dir
        self.cfg = config
        self.training = training

        # groups: {eeg_id: [row_dict, ...]}  — ordered by eeg_sub_id
        grouped = df.sort_values("eeg_sub_id").groupby("eeg_id")
        self.groups = {eid: grp.to_dict("records") for eid, grp in grouped}
        self.eeg_ids = list(self.groups.keys())

        freqs = build_freqs(config.lower_freq, config.upper_freq, config.n_scales)
        self.psi = make_psi_matrix(freqs, config.window_samples, FS, config.cwt_m)

    def __len__(self) -> int:
        return len(self.eeg_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        eeg_id = self.eeg_ids[idx]
        rows = self.groups[eeg_id]
        row = random.choice(rows) if self.training else rows[0]

        signals = self._load_signals(eeg_id, int(row["eeg_label_offset_seconds"]))
        image = self._build_image(signals)
        label = self._make_label(row)
        return image, label

    def _load_signals(self, eeg_id: int, offset_sec: int) -> np.ndarray:
        path = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
        eeg = pq.read_table(path, columns=NEEDED_COLS).to_pandas()

        start = offset_sec * FS
        chunk = eeg.iloc[start : start + self.cfg.window_samples]

        signals = np.stack(
            [chunk[a].values - chunk[b].values for a, b in BIPOLAR_PAIRS],
            axis=0,
        ).astype(np.float32)  # (18, window_samples)

        if signals.shape[1] < self.cfg.window_samples:
            pad = self.cfg.window_samples - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad)))

        signals = np.nan_to_num(signals, nan=0.0, posinf=1024.0, neginf=-1024.0)
        return np.clip(signals, -1024.0, 1024.0) / 32.0

    def _build_image(self, signals: np.ndarray) -> torch.Tensor:
        rows = []
        for ch in range(signals.shape[0]):
            x = torch.from_numpy(signals[ch])
            s = paul_scalogram(x, self.psi, self.cfg.cwt_stride, self.cfg.cwt_border_crop)
            rows.append(s)

        image = torch.cat(rows, dim=0).float()  # (18*n_scales, T)
        image = torch.log1p(image.clamp(min=0))
        image = (image - image.mean()) / (image.std() + 1e-6)

        image = F.interpolate(
            image.unsqueeze(0).unsqueeze(0),
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (1, H, W)

        return image.expand(3, -1, -1).contiguous()  # (3, H, W)

    def _make_label(self, row: dict) -> torch.Tensor:
        votes = np.array([row[col] for col in VOTE_COLS], dtype=np.float32)
        votes += self.cfg.label_smoothing
        return torch.from_numpy(votes / votes.sum())
