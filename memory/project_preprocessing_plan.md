---
name: EEG preprocessing plan
description: Agreed preprocessing steps for HMS EEG model — what to implement next
type: project
---

Agreed minimal preprocessing that should give noticeable improvement over baseline:

1. Рівень 1 — per-channel z-score normalization
2. Рівень 2 — brain leads (16 bipolar derivations from 4 chains: LL, RL, LP, RP)
3. Рівень 3 — bandpass filter 0.5–20 Hz (scipy butter, order=5)
4. Take 50 seconds (10 000 samples) starting from eeg_label_offset_seconds (same as eeg-raw-model 2nd place)
5. Labels — do NOT touch, keep per-row soft labels as-is

**Why:** Current val_acc=7.5% because raw signal without filtering/normalization is noise to the model.
**How to apply:** Implement in src/dataset.py HMSEEGDataset.__getitem__
