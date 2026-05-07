"""
Runs inference with both trained models and averages their predictions.

Usage:
    python src/predict.py

Outputs: submission.csv
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HMSSpectrogramDataset, HMSEEGDataset, VOTE_COLS
from spec_model import SpectrogramModel
from eeg_model import EEGModel
from train import HMSModule
from utils import get_device, DATA_DIR, CKPT_DIR

LABEL_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']


def load_model(module_class, model, ckpt_path, device):
    lit = module_class.load_from_checkpoint(ckpt_path, model=model, map_location=device)
    lit.eval()
    return lit.to(device)


def predict_probs(lit_model, loader, device) -> np.ndarray:
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = lit_model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main():
    device = get_device()
    print(f'device: {device}')

    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test_df['eeg_id'] = test_df['eeg_id'].astype(int)
    for col in VOTE_COLS:
        test_df[col] = 0
    if 'eeg_label_offset_seconds' not in test_df.columns:
        test_df['eeg_label_offset_seconds'] = 0.0
    if 'spectrogram_label_offset_seconds' not in test_df.columns:
        test_df['spectrogram_label_offset_seconds'] = 0.0

    # ── spectrogram predictions ──────────────────────────────────────────────
    spec_ckpt = os.path.join(CKPT_DIR, 'spec_best.ckpt')
    spec_probs = None
    if os.path.exists(spec_ckpt):
        lit = load_model(HMSModule, SpectrogramModel(pretrained=False), spec_ckpt, device)
        ds  = HMSSpectrogramDataset(test_df, os.path.join(DATA_DIR, 'test_spectrograms'), augment=False)
        spec_probs = predict_probs(lit, DataLoader(ds, batch_size=8, shuffle=False), device)
        print(f'spec predictions: {spec_probs.shape}')
    else:
        print('spec checkpoint not found, skipping')

    # ── eeg predictions ──────────────────────────────────────────────────────
    eeg_ckpt = os.path.join(CKPT_DIR, 'eeg_best.ckpt')
    eeg_probs = None
    if os.path.exists(eeg_ckpt):
        lit = load_model(HMSModule, EEGModel(), eeg_ckpt, device)
        ds  = HMSEEGDataset(test_df, os.path.join(DATA_DIR, 'test_eegs'), augment=False)
        eeg_probs = predict_probs(lit, DataLoader(ds, batch_size=8, shuffle=False), device)
        print(f'eeg predictions: {eeg_probs.shape}')
    else:
        print('eeg checkpoint not found, skipping')

    # ── ensemble ─────────────────────────────────────────────────────────────
    available = [p for p in [spec_probs, eeg_probs] if p is not None]
    if not available:
        raise RuntimeError('no checkpoints found — train at least one model first')

    final_probs = np.mean(available, axis=0)

    # ── write submission ──────────────────────────────────────────────────────
    submission = pd.DataFrame(final_probs, columns=LABEL_COLS)
    submission.insert(0, 'eeg_id', test_df['eeg_id'].values)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'submission.csv')
    submission.to_csv(out_path, index=False)
    print(f'\nsaved: {out_path}')
    print(submission)


if __name__ == '__main__':
    main()
