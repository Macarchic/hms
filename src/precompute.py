"""
Pre-computes STFT spectrograms for all EEG windows and saves to disk.
Run once before training to eliminate repeated computation per epoch.

Usage:
    python src/precompute.py                        # всі дані
    python src/precompute.py --limit 1000           # перші N для перевірки
    python src/precompute.py --split test           # тільки тест

Output: data/eeg_cache/{eeg_id}_{eeg_sub_id}.npy  shape=(16,13,198) float16
        ~8.6 GB total for full train set
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (
    EEG_ALL_COLS, EEG_FS, EEG_WINDOW_SAMPLES,
    _CHAINS, _BUTTER_SOS, _FREQ_MASK, EEG_N_FFT, EEG_HOP, EEG_TIME_STEPS,
)
from scipy.signal import sosfilt, stft as scipy_stft
from utils import DATA_DIR, CACHE_DIR


def process_window(data: np.ndarray, col_idx: dict, offset_sec: float) -> np.ndarray:
    """Full pipeline: raw signal → (16, 13, 198) float16 spectrogram."""
    start = int(offset_sec * EEG_FS)
    end = start + EEG_WINDOW_SAMPLES
    file_len = len(data)
    if end > file_len:
        end = file_len
        start = max(0, end - EEG_WINDOW_SAMPLES)
    segment = data[start:end]

    if len(segment) < EEG_WINDOW_SAMPLES:
        pad = np.zeros((EEG_WINDOW_SAMPLES - len(segment), segment.shape[1]), dtype=np.float32)
        segment = np.concatenate([segment, pad], axis=0)

    # brain leads
    leads = []
    for chain in _CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            leads.append(segment[:, col_idx[a]] - segment[:, col_idx[b]])
    eeg = np.stack(leads, axis=0)   # (16, 10000)

    # NaN fill → clip → bandpass → z-score
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

    # STFT
    specs = []
    for ch in range(eeg.shape[0]):
        _, _, Zxx = scipy_stft(eeg[ch], fs=EEG_FS, nperseg=EEG_N_FFT,
                               noverlap=EEG_N_FFT - EEG_HOP,
                               boundary=None, padded=False)
        mag = np.abs(Zxx)[_FREQ_MASK, :EEG_TIME_STEPS]
        specs.append(mag)
    spec = np.stack(specs, axis=0)   # (16, 13, 198)
    spec = np.log1p(spec)

    return spec.astype(np.float16)


def cache_path(eeg_id: int, sub_id: int) -> str:
    return os.path.join(CACHE_DIR, f'{eeg_id}_{sub_id}.npy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=None, help='process only first N windows')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    rows = []
    if args.split in ('train', 'all'):
        df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        df = df.drop_duplicates(subset=['eeg_id', 'eeg_sub_id']).reset_index(drop=True)
        rows.append(df[['eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds']])

    if args.split in ('test', 'all'):
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        test['eeg_sub_id'] = 0
        test['eeg_label_offset_seconds'] = 0.0
        rows.append(test[['eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds']])

    all_rows = pd.concat(rows, ignore_index=True)
    if args.limit:
        all_rows = all_rows.iloc[:args.limit]

    # пропускаємо вже існуючі
    todo = all_rows[~all_rows.apply(
        lambda r: os.path.exists(cache_path(int(r['eeg_id']), int(r['eeg_sub_id']))), axis=1
    )]
    print(f'Всього вікон: {len(all_rows)}  |  пропущено (є в кеші): {len(all_rows) - len(todo)}  |  до обробки: {len(todo)}')

    if todo.empty:
        print('Кеш вже повний.')
        return

    # групуємо по eeg_id щоб читати parquet один раз на файл
    todo_by_file = todo.groupby('eeg_id')
    n_files = len(todo_by_file)
    done = 0
    skipped = 0

    for file_idx, (eeg_id, group) in enumerate(todo_by_file):
        parquet_path = os.path.join(DATA_DIR, 'train_eegs', f'{eeg_id}.parquet')
        if not os.path.exists(parquet_path):
            parquet_path = os.path.join(DATA_DIR, 'test_eegs', f'{eeg_id}.parquet')
        if not os.path.exists(parquet_path):
            skipped += len(group)
            continue

        raw = pd.read_parquet(parquet_path)
        data = raw[EEG_ALL_COLS].values.astype(np.float32)
        col_idx = {c: i for i, c in enumerate(EEG_ALL_COLS)}

        for _, row in group.iterrows():
            sub_id = int(row['eeg_sub_id'])
            offset = float(row['eeg_label_offset_seconds'])
            out_path = cache_path(eeg_id, sub_id)

            spec = process_window(data, col_idx, offset)
            np.save(out_path, spec)
            done += 1

        pct = (file_idx + 1) / n_files * 100
        cached_gb = done * 16 * 13 * 198 * 2 / 1e9
        print(f'\r  [{file_idx+1:5d}/{n_files}] {pct:5.1f}%  збережено: {done:6d}  розмір: {cached_gb:.2f} GB',
              end='', flush=True)

    print(f'\n\nГотово. Збережено: {done}  |  пропущено (файл не знайдено): {skipped}')
    total_gb = done * 16 * 13 * 198 * 2 / 1e9
    print(f'Розмір кешу: {total_gb:.2f} GB  →  {CACHE_DIR}')


if __name__ == '__main__':
    main()
