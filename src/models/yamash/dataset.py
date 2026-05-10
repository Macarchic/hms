import os
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset

from src.models.yamash.augment import xy_masking
from src.models.yamash.config import YamashConfig

FS = 200
VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]

NEEDED_COLS = list(dict.fromkeys(c for pair in BIPOLAR_PAIRS for c in pair))


def _bandpass(signals: np.ndarray, lowcut: float, highcut: float, order: int) -> np.ndarray:
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=FS, output="sos")
    return sosfilt(sos, signals, axis=1).astype(np.float32)


class HMSDataset(Dataset):
    """
    One item = one unique eeg_id.
    Training:   random sub_id per call
    Validation: always first sub_id

    Builds a 512×512 image by stacking 3 time crops of the bandpass-filtered
    raw EEG (10 s / 25 s / 50 s), each resized to the same height then
    concatenated vertically — yamash's "raw signal as image" approach.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: str,
        config: YamashConfig,
        training: bool = True,
    ):
        self.eeg_dir = eeg_dir
        self.cfg = config
        self.training = training

        grouped = df.sort_values("eeg_sub_id").groupby("eeg_id")
        self.groups = {eid: grp.to_dict("records") for eid, grp in grouped}
        self.eeg_ids = list(self.groups.keys())

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
        chunk = eeg.iloc[start:start + self.cfg.window_samples]

        signals = np.stack(
            [chunk[a].values - chunk[b].values for a, b in BIPOLAR_PAIRS],
            axis=0,
        ).astype(np.float32)  # (18, window_samples)

        if signals.shape[1] < self.cfg.window_samples:
            pad = self.cfg.window_samples - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad)))

        signals = np.nan_to_num(signals, nan=0.0, posinf=1024.0, neginf=-1024.0)
        signals = np.clip(signals, -1024.0, 1024.0) / 32.0
        return _bandpass(signals, self.cfg.bandpass_low, self.cfg.bandpass_high, self.cfg.bandpass_order)

    def _build_image(self, signals: np.ndarray) -> torch.Tensor:
        """Stack 3 time crops (10 s / 25 s / 50 s) as horizontal strips → (3, H, W)."""
        n = self.cfg.window_samples  # 10000
        half = n // 2
        crops = [
            signals[:, half - 1000:half + 1000],  # center 2000 samples (10 s)
            signals[:, half - 2500:half + 2500],  # center 5000 samples (25 s)
            signals,                               # full 10000 samples (50 s)
        ]

        sz = self.cfg.image_size
        strip_h = sz // 3  # 170; last strip gets sz - 2*strip_h = 172 to reach 512

        strips = []
        for i, crop in enumerate(crops):
            h = sz - 2 * strip_h if i == 2 else strip_h
            t = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)  # (1,1,18,N)
            strip = F.interpolate(t, size=(h, sz), mode="bilinear", align_corners=False)
            strips.append(strip.squeeze(0).squeeze(0))  # (h, sz)

        image = torch.cat(strips, dim=0)  # (sz, sz)
        image = (image - image.mean()) / (image.std() + 1e-6)

        if self.training:
            image = image.unsqueeze(0)      # (1, sz, sz) for xy_masking
            image = xy_masking(image.expand(3, -1, -1).clone())
        else:
            image = image.unsqueeze(0).expand(3, -1, -1)

        return image.contiguous()  # (3, sz, sz)

    def _make_label(self, row: dict) -> torch.Tensor:
        votes = np.array([row[col] for col in VOTE_COLS], dtype=np.float32)
        votes += self.cfg.label_smoothing
        return torch.from_numpy(votes / votes.sum())
