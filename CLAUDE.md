# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle [HMS — Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) competition solution. The task is multi-class soft-label classification of EEG brain activity into 6 categories: seizure, lpd, gpd, lrda, grda, other.

## Commands

```bash
# Train spectrogram model (EfficientNet-B0)
python src/train.py --modality spec

# Train EEG model (1D CNN)
python src/train.py --modality eeg

# Common flags
python src/train.py --modality spec --epochs 30 --batch_size 32 --lr 1e-3 --num_workers 4

# Quick sanity check (1 epoch, small batch)
python src/train.py --modality spec --epochs 1 --batch_size 4

# Run ensemble inference → generates submission.csv
python src/predict.py
```

No requirements.txt or Makefile. Dependencies live in `.venv/`; activate with `source .venv/bin/activate`. Key packages: PyTorch, timm (EfficientNet), pandas, pyarrow, kagglehub.

## Architecture

Dual-modality ensemble: two independent models trained separately, predictions averaged at inference time.

### Data pipeline (`src/dataset.py`)

| Constant | Value |
|---|---|
| `VOTE_COLS` | 6 label columns (seizure_vote … other_vote) |
| `SPEC_GROUPS` | 4 EEG chain groups: LL, RL, LP, RP |
| `EEG_CHANNELS` | 20 standard scalp electrodes + EKG |
| `EEG_FS` | 200 Hz |
| `EEG_WINDOW_SEC` | 10 s → 2000 samples |

**HMSSpectrogramDataset**: reads `train_spectrograms/{id}.parquet` → tensor `(4, 100, 300)`. Training augmentations: horizontal flip (p=0.5), frequency masking (p=0.5).

**HMSEEGDataset**: reads `train_eegs/{id}.parquet`, extracts 10-s window at `eeg_label_offset_seconds` → tensor `(20, 2000)`. Training augmentations: Gaussian noise (σ=0.01), per-channel random sign flip.

Both datasets normalise vote columns to a probability distribution used as soft targets.

### Models

**SpectrogramModel** (`src/spec_model.py`): EfficientNet-B0 (`timm`, pretrained) with `in_chans=4`, global avg pool, Dropout(0.2) → Linear(1280→6).

**EEGModel** (`src/eeg_model.py`): 5-block 1D CNN, filter progression 64→128→256→256→512, kernels [7,7,5,5,3], stride 2 each. Adaptive avg pool → Dropout(0.2) → Linear(512→6).

### Training (`src/train.py`)

- Loss: KL-divergence (`F.kl_div(log_softmax(logits), soft_targets)`)
- Optimiser: AdamW
- Scheduler: cosine annealing
- Split: `GroupShuffleSplit` on `patient_id` (80/20), preventing leakage
- Device priority: MPS → CUDA → CPU
- Checkpoints: `checkpoints/{spec_best,eeg_best}.pt` (best val loss)

### Inference (`src/predict.py`)

Loads both checkpoints (skips missing ones), averages softmax outputs, writes `submission.csv` in Kaggle format.

## Data layout

```
data/
  train.csv                  # 106,800 labelled segments
  test.csv                   # 1 test segment
  sample_submission.csv
  train_eegs/                # 17,300 parquet files
  train_spectrograms/        # 11,138 parquet files
  test_eegs/                 # 1 parquet file
  test_spectrograms/         # 1 parquet file
```

`train.csv` key columns: `eeg_id`, `spectrogram_id`, `patient_id`, `eeg_label_offset_seconds`, `spectrogram_label_offset_seconds`, plus the 6 vote columns. Deduplication in `train.py` keeps one row per `(spectrogram_id, spectrogram_sub_id)` or `(eeg_id, eeg_sub_id)` depending on modality.
