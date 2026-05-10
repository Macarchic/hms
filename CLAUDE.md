# HMS – Harmful Brain Activity Classification

## Змагання
Kaggle: [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
Задача: 6-класова класифікація патерну активності мозку з ЕЕГ / спектрограм.

**6 класів (expert_consensus / vote columns):**
| Клас | Повна назва |
|------|-------------|
| Seizure | Епілептичний напад |
| LPD | Lateralized Periodic Discharges |
| GPD | Generalized Periodic Discharges |
| LRDA | Lateralized Rhythmic Delta Activity |
| GRDA | Generalized Rhythmic Delta Activity |
| Other | Інше |

**Метрика:** KL-divergence між передбаченим розподілом і усередненими голосами лікарів.

---

## Структура даних

```
data/
  train.csv               # 106 800 анотацій-вікон
  test.csv
  sample_submission.csv
  train_eegs/             # 17 300 parquet-файлів (сирий ЕЕГ)
  train_spectrograms/     # 11 138 parquet-файлів (спектрограми Kaggle)
  test_eegs/
  test_spectrograms/
```

### train.csv — ключові поля
| Поле | Опис |
|------|------|
| eeg_id | ID сирого ЕЕГ-файлу |
| eeg_sub_id | Порядковий номер вікна в межах цього файлу |
| eeg_label_offset_seconds | Початок 50-сек вікна всередині ЕЕГ-файлу |
| spectrogram_id | ID спектрограми |
| spectrogram_label_offset_seconds | Початок 10-хв вікна всередині спектрограми |
| patient_id | Пацієнт |
| expert_consensus | Мажоритарна мітка |
| seizure_vote … other_vote | Голоси 6 лікарів (сума ≤ 6 на рядок) |

### ЕЕГ parquet (train_eegs/*.parquet)
- **20 стовпців:** 19 каналів (10–20 система: Fp1 F3 C3 P3 F7 T3 T5 O1 Fz Cz Pz Fp2 F4 C4 P4 F8 T4 T6 O2) + EKG
- **Частота дискретизації:** 200 Гц → 1 рядок = 5 мс
- **Мінімальна довжина запису:** 10 000 рядків = 50 сек (файли бувають довшими)
- **Одиниці:** мікровольти (µV), float32
- **Анотоване вікно:** 50 сек (10 000 рядків), що береться з offset = `eeg_label_offset_seconds`

### Спектрограма parquet (train_spectrograms/*.parquet)
- **401 стовпець:** `time` (секунди) + 400 частотних бінів (4 ланцюги × 100 бінів)
- **4 ланцюги (bipolar montage):** LL (Left Lateral), RL (Right Lateral), LP (Left Parasagittal), RP (Right Parasagittal)
- **Частотний діапазон:** 0.59 – 19.92 Гц, крок ~0.20 Гц (100 бінів)
- **Часовий крок:** 2 сек → 1 рядок = 2 сек
- **Мінімальна довжина:** 300 рядків = 600 сек = 10 хв (файли бувають довшими)
- **Анотоване вікно:** 10 хв, offset = `spectrogram_label_offset_seconds`
- **Одиниці:** потужність (µV²/Гц), float32

### Зв'язок ЕЕГ ↔ Спектрограма
Спектрограма **похідна** від ЕЕГ: Kaggle попередньо порахував Short-Time Fourier Transform (STFT) по 4 bipolar-ланцюгах і зберіг потужності. Один 2-секундний рядок спектрограми = FFT одного 2-секундного вікна ЕЕГ (400 рядків при 200 Гц).

---

## Поточний пайплайн (src/)

| Файл | Призначення |
|------|-------------|
| `src/utils.py` | Шляхи (DATA_DIR, CACHE_DIR, …), `get_device()` |
| `src/logger.py` | Централізований логер (імпортувати звідси, не inline) |
| `src/eeg_model.py` | Архітектура моделі для ЕЕГ |
| `src/callbacks.py` | Callbacks для тренування |

### Кроки запуску
```bash
source .venv/bin/activate

# 1. Передобчислення кешу (STFT → parquet)
python src/precompute.py --limit 100   # тест на 100
python src/precompute.py               # повний (~8.6 GB, ~30 хв)

# 2. Тренування
python src/train.py --modality eeg  --epochs 30 --batch_size 16
python src/train.py --modality spec --epochs 30 --batch_size 16

# Санітарна перевірка
python src/train.py --modality eeg --epochs 1 --batch_size 4 --train_size 8 --val_size 4

# 3. Predict / submission
python src/predict.py
```
Чекпоінти: `checkpoints/eeg_best.ckpt`, `checkpoints/spec_best.ckpt`
Early stopping: 7 епох без покращення val_loss.

---

## Правила кодування
- Логер: завжди `from src.logger import get_logger`, не inline logging
- CLI params (--batch_size, --train_size, etc.): тільки `int`, не float/fraction
- Форматування: без вирівнювальних пробілів (Flake8 E221)
- Без зайвих коментарів (лише non-obvious WHY)

---

## Середовище
- Python venv: `.venv/`
- Пристрій: MPS (Apple Silicon) → CUDA → CPU (auto in `get_device()`)
- Залежності: `requirements.txt` (torch 2.11, timm 1.0, pandas 3.0, pyarrow 24)
