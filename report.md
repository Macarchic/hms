# HMS — Аналіз препроцесингу: рішення фіналістів

Проведено аналіз рішень фіналістів змагання HMS. Нижче зведено як саме вони підходили до препроцесингу даних, які трансформи обрали і чому.

---

## Чому не використовували готові спектрограми Kaggle

Всі автори топ-рішень працювали від сирого ЕЕГ, а не від преобчислених спектрограм Kaggle. Три причини:

1. **20 Гц зрізання** — готові спектрограми мають верхню межу 19.92 Гц. Seizure може мати компоненти до 25–40 Гц, ці дані повністю губляться.
2. **2-секундний часовий крок** — занадто грубо для швидких подій. LPD/seizure spike тривають десятки мілісекунд, а один рядок спектрограми = 2 секунди.
3. **Усереднення по ланцюгу** — колонка `LL_x` = середнє потужностей чотирьох bipolar пар (Fp1–F7, F7–T3, T3–T5, T5–O1). Губиться просторова різниця між парами — а саме вона може відрізняти lateralized від generalized патернів.

Робота від сирого ЕЕГ дає повний контроль: довільний частотний діапазон, довільна часова роздільна здатність, окремий сигнал для кожної bipolar пари.

---

## Методи трансформації

### yamash — найпростіший підхід (CV ~0.24)

Сирий відфільтрований ЕЕГ **без частотного трансформу взагалі** — часовий ряд виставляється горизонтально як піксельні рядки зображення.

**Пайплайн:**
```
Raw EEG
  → longitudinal bipolar montage (18 signals)
  → bandpass filter (різний діапазон для різних моделей)
  → 3 часових кропи: 18×2000, 18×5000, 18×10000
  → кожен resize до однакової висоти
  → concatenate вертикально
  → 512×512 зображення
  → 2D CNN (convnext_atto × 4, inception_next_tiny × 1)
```

На inference використовує entmax замість softmax — дає більш гострий розподіл, що ефективно для KL-divergence метрики.

---

### suguuuuu — CWT scalogram (CV ~0.2399 one model)

**Чому CWT, а не STFT:**

STFT має фіксований розмір вікна → жорсткий tradeoff:
- Широке вікно → хороша частотна роздільна здатність, погана часова
- Вузьке вікно → навпаки

CWT (Continuous Wavelet Transform) має **адаптивне вікно** — воно є самим вейвлетом, масштабованим під кожну частоту:
- Низькі частоти (delta 0.5–4 Гц) → вейвлет розтягується → широке "вікно" (~2 сек) → точна частота
- Високі частоти (seizure 15–40 Гц) → вейвлет стискається → вузьке "вікно" (~25 мс) → точний момент

ЕЕГ — нестаціонарний сигнал: патерни з'являються і зникають за секунди, змінюють частоту. CWT для цього ідеальний.

**Пайплайн:**
```
Raw EEG (50 сек, 10000 семплів)
  → 18 bipolar пар (без EKG)
  → clip(-1024, 1024) / 32
  → crop центральні 50 сек (10000 frames) — краще ніж 25 або 10 сек
  → CWT (Morlet wavelet)
  → stack вертикально
  → resize → 512×512
  → MaxViT_base
```

**CWT параметри:**
```
wavelet_width = 7       ← кількість циклів вейвлета Morlet (форма, не розмір вікна)
fs = 200                ← частота дискретизації
lower_freq = 0.5 Гц
upper_freq = 40 Гц      ← важливо: 0.5–40 Гц краще ніж 0.5–20 Гц
n_scales = 40           ← 40 частотних рядків у log-scale
border_crop = 1         ← обрізання крайових артефактів
stride = 16             ← крок по часовій осі
```

**Нормалізація міток:** summing labels with eeg_id (якщо один eeg_id має кілька анотацій — суммуємо голоси).

**Звідки 18 × 40 × 625:**
```
Input: 18 bipolar pairs × 10,000 samples

n_scales = 40     → 40 частотних рядків (вісь Y кожного scalogram)
stride = 16       → часова вісь: 10000 / 16 = 625

Output per channel: 40 × 625
Output total:      18 × 40 × 625
                    │    │    └── часові кадри
                    │    └─────── частотні скейли
                    └──────────── bipolar пари
```

`border_crop=1` — видаляє крайові артефакти CWT (cone of influence): на краях сигналу вейвлет виходить за межі і заповнює нулями, значення ненадійні. При stride=16 і 10000 семплах це не змінює розмір (10000/16 = 625 рівно).

**Як resize 18 × 40 × 625 → 512 × 512:**
```
18 scalograms (кожен 40 × 625)
  → stack vertically (vstack)
  → 720 × 625 зображення    (18 × 40 = 720 рядків)
  → bilinear interpolation
  → 512 × 512
```

Коефіцієнти стискання: висота 720→512 (×0.711), ширина 625→512 (×0.819). Це **зжимання, не кроп**. Покриття (весь часовий діапазон і весь частотний) зберігається. Губиться лише роздільна здатність — дрібні деталі менші за ~2 пікселі розмиваються, але ЕЕГ-патерни займають секунди і цілі частотні смуги (десятки пікселів), тому втрата несуттєва. Кроп був би гіршим — він викинув би реальні частоти або реальні часові відрізки.

512×512 — стандартний розмір для timm-моделей (MaxViT, EfficientNet тощо).

---

### kfuji — CWT з Paul wavelet

Підхід ідентичний suguuuuu, але замість Morlet використав **Paul wavelet** — щоб збільшити різноманітність ансамблю. Реалізація: PyTorchWavelets.

```
Raw EEG (без EKG)
  → 18 bipolar пар
  → CWT (Paul wavelet, параметри як у suguuuuu)
  → stack вертикально
  → resize → 512×512
  → 2D CNN (timm)
```

---

### Muku — два незалежні підходи (CV ~0.2229)

Найскладніший пайплайн: дві гілки, кожна дає окремий 2D image для timm-моделі, виходи об'єднуються через FC layer.

**Вхід:** 16-channel anterior-posterior montages з raw EEG, 50 сек / 10000 семплів.

#### Гілка 1: 1D temporal convolution

Ідея: замість ручного вибору трансформу — нехай мережа **сама навчиться** знаходити ритмічні патерни. Натхнення: EEGNet paper, G2Net top-1 solution.

```
Центральні 10 сек (2000 семплів) із 16 каналів
  → Conv1D(kernel_size=200, out_channels=16 або 14)
  → BatchNorm → SiLU → AvgPool
  → 16 feature maps (256 або 224 точок кожна)
  → stack by montage channels або by feature maps → 2D image
  → timm model
```

`kernel_size = 200 = fs` — ядро рівно 1 секунді сигналу. Мережа вчиться знаходити конкретні ритмічні патерни (1–3 Гц для GRDA/LRDA, 1+ Гц спайки для LPD/GPD тощо) без явного FFT.

#### Гілка 2: Superlets (замість STFT)

STFT відкинув через: "loss of resolution in time or frequency." Перейшов на **Superlets** (суперпозиція вейвлетів з адаптивним порядком) — дають вищу роздільну здатність одночасно і в часі, і в частоті.

```
16 каналів (50 сек)
  → Superlet Transform
     min_freq=0.5, max_freq=20.0
     base_cycle=1, min_order=1, max_order=16
  → extract central 10–40 сек
  → resize → 256×256 або 224×224
  → timm model
```

Реалізація: github.com/irhum/superlets

#### Тренування (Muku)

- **2-stage:** Stage 1 — total_votes > 1 (широке охоплення), Stage 2 — total_votes > 9 (якісні мітки)
- **Label smoothing:** додати 0.02 до кожного vote перед нормалізацією → регуляризація для семплів з малою кількістю голосів
- **Augmentation:** ±5 сек random time-shift; random bandpass filter (для wave-моделей); XYMasking (для spec-моделей)
- **Bandpass filter highcut 30–40 Гц:** high-frequency noise корелює з класом Other → важливо зберегти

---

## Порівняльна таблиця

| Автор | Вхід | Трансформ | Вихідний розмір |
|-------|------|-----------|-----------------|
| yamash | raw EEG, 18 bipolar | без трансформу (raw signal as image) | 512×512 |
| suguuuuu | raw EEG, 18 bipolar | CWT Morlet (0.5–40 Гц, 40 scales) | 18×40×625 → 512×512 |
| kfuji | raw EEG, 18 bipolar | CWT Paul wavelet | → 512×512 |
| Muku | raw EEG, 16ch AP | 1D temporal conv + Superlets | 256×256 / 224×224 |

Kaggle-спектрограми як основний вхід — **ніхто з топів не використовував**.

---

## Ключові висновки для нашого пайплайну (препроцесинг)

1. **CWT (0.5–40 Гц) краще ніж Kaggle spectrograms** — вища роздільна здатність, ширший діапазон, окремі пари
2. **50 сек (10000 семплів) краще ніж 10 або 25 сек** — підтверджено suguuuuu
3. **0.5–40 Гц краще ніж 0.5–20 Гц** — підтверджено suguuuuu (важливо для Seizure)
4. **18 bipolar пар зберігати окремо** — не усереднювати по ланцюгу як Kaggle
5. **Clip + normalize** перед CWT: `x.clip(-1024, 1024) / 32`
6. **Label smoothing +0.02** + 2-stage training — ефективна стратегія для шумних міток
7. **Resize через bilinear interpolation** (не кроп) — зберігає все покриття даних

---

# Архітектура моделей

## Загальне: з нуля чи fine-tune?

**Всі використовували pretrained моделі з ImageNet** — виключно fine-tuning, ніхто не тренував from scratch. Конкретні checkpoint'и:
- suguuuuu / kfuji: `maxvit_base_tf_512.in21k_ft_in1k` — претренована на ImageNet-21k, потім fine-tuned на ImageNet-1k, потім fine-tuned на ЕЕГ
- Muku: swinv2, caformer, gcvit — всі з timm, ImageNet pretrained
- yamash: convnext_atto, inception_next_tiny — теж timm, ImageNet

Чому ImageNet-pretrained працює на ЕЕГ scalogram: scalogram — це зображення, і низькорівневі фічі CNN (детектори країв, текстур, локальних патернів) корисні і для спектрограм.

---

## yamash

**Моделі (5 штук у фінальному submission):**
```
4 × convnext_atto  — різні seed + різний діапазон bandpass filter
                     CV: 0.2452 / 0.2385 / 0.2457 / 0.2351
1 × inception_next_tiny
                     CV: 0.2309
```

**Ансамбль:** Non-negative Linear Regression (не просте усереднення).
Кожна цільова змінна (6 класів) передбачається окремою NNLS-регресією з виходів всіх моделей. Перейшов від simple average до NNLS коли побачив, що кореляція між CV і public LB зберігається навіть при overfitting — більш "впевнені" моделі отримували більшу вагу.

**Softmax → entmax на inference:**
- Softmax завжди дає ненульові ймовірності, але реальні мітки мають багато нулів (лікарі часто однозначні)
- entmax з alpha ≈ 1.03 — trохи "crisp-іший" розподіл, значення більш різкі без явних нулів
- Покращення: +0.004 на public і private LB
- Training: softmax; Inference: entmax — важлива різниця

---

## suguuuuu

**Модель 1** (4 seed ensemble, one model CV: 0.2399):
```
CWT 18×40×625 → stack → 512×512 → MaxViT_base
```

**Модель 2** (4 seed ensemble, one model CV: 0.2381) — dual input:
```
Гілка A: CWT → 512×512 → MaxViT_base ──┐
                                         concat → FC
Гілка B: raw EEG (yamash method) →      ─┘
         stack 18×10000 → 512×512 → Convnext_atto
```
Модель 2 бачить одночасно CWT scalogram і сирий ЕЕГ як зображення.

**Тренування:**
```
2-stage: Stage1 = 5 epochs (всі дані)
         Stage2 = 15 epochs (total_votes ≥ 10)
Optimizer:    Adan
Scheduler:    CosineAnnealingLR
LR:           1e-3
Augmentation: XYMasking, Mixup
```

**Що не спрацювало (явно зазначено):** STFT, Kaggle spec, CQT, train with plotted image, various montage configurations.

---

## kfuji

**Модель:** `maxvit_base_tf_512.in21k_ft_in1k`
Претренована ImageNet-21k → fine-tune ImageNet-1k → fine-tune ЕЕГ. Вхід 512×512.

**Sampling стратегія:** на кожну епоху для кожного `eeg_id` випадково вибирається один `eeg_sub_id` — різні вікна одного запису розподілені між епохами, а не батчами (зменшує data leakage між train/val).

**Тренування:**
```
Stage1: 5 epochs, всі дані
Stage2: 15 epochs, тільки total_votes ≥ 10
LR:           1e-3
Optimizer:    Adan
Scheduler:    CosineAnnealingLR
Augmentation: XYMasking, Mixup
```

**Три варіації** (merged у фінальний submission):
```
2000 frames (10 сек), Paul(m=4)   CV: 0.2475
5000 frames (25 сек), Paul(m=4)   CV: 0.2309
5000 frames (25 сек), Paul(m=16)  CV: 0.2311
```
`m` — порядок Paul wavelet: менший `m` → краща часова точність, гірша частотна; більший — навпаки.

**Що не спрацювало:** DOG (Derivative of Gaussian) замість Paul; довші вікна 50 сек (добре окремо, але не давали приросту в ансамблі).

---

## Muku (best CV: 0.2229)

**Архітектура — multi-branch fusion:**
```
                   ┌─→ timm model ──────────────┐
[Superlet image] ──┤                              │
                   └─→ GRU (optional) ───────────┤
                                                  ├→ FC → prediction
[1D conv maps] ──→ stack left channels  → timm ──┤
               └─→ stack right channels → timm ──┘
```

**Backbone (всі ImageNet pretrained через timm):**
```
Best:   swinv2_tiny_window16   CV: 0.2229
Інші (внесли у ансамбль):
        swinv2_tiny_window8
        caformer_s18
        gcvit_xtiny / gcvit_xxtiny
        convnextv2_atto
        maxvit_pico
        inception_next_tiny
        poolformerv2_s12
        nextvit_small
```
Висновок Muku: **маленькі моделі спрацювали краще за великі** — менший overfit на EEG специфіку, KL divergence чутлива до extreme predictions тому diverse ensemble важливий.

**Superlet resize (детально):**
```
Raw output: (32, 10000) per channel   ← freqs = linspace(0.5, 20.0, 32)
Збереження: resize to (32, 1000) .npy ← для економії диску

Under training:
  1. Crop center: (32, 256/512/768)    ← не max 1000, щоб 5сек time-shift міг зсунути
  2. Resize each channel: (16, 256)    ← для 16ch montage; (64, 256) для 4ch avg
  3. Stack: 16 channels × 16px = 256  → final (256, 256)
```
Чому resize до 256: swinv2 має ліміт 256px або 224px — прийнятний tradeoff за доступ до кращого backbone.

**Loss і тренування:**
```
Loss:      KLDivLoss + aux loss на кожному виході гілки
Optimizer: Adan
Scheduler: CosineAnnealingLR
LR:        Stage1 = 1e-3 / Stage2 = 1e-4
Stage1:    total_votes > 1
Stage2:    total_votes > 9
Label smoothing: +0.02 до кожного vote перед нормалізацією
```

---

## Зведена таблиця

| | yamash | suguuuuu | kfuji | Muku |
|--|--|--|--|--|
| **Backbone** | convnext_atto, inception_next_tiny | MaxViT_base | maxvit_base_tf_512 | swinv2_tiny_window16 та ін. |
| **Pretrained** | ImageNet | ImageNet-21k→1k | ImageNet-21k→1k | ImageNet |
| **Input size** | 512×512 | 512×512 | 512×512 | 256×256 / 224×224 |
| **Optimizer** | — | Adan | Adan | Adan |
| **Scheduler** | — | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| **Loss** | — | KLDivLoss + aux | KLDivLoss | KLDivLoss + aux |
| **LR** | — | 1e-3 → 1e-4 | 1e-3 | 1e-3 → 1e-4 |
| **2-stage** | ні | >1 → >9 votes | all → ≥10 votes | >1 → >9 votes |
| **Augmentation** | — | XYMasking, Mixup | XYMasking, Mixup | XYMasking + Mixup + time-shift + bandpass |
| **Ensemble** | NNLS × 5 моделей | 4 seed + dual-input model | 3 варіації Paul wavelet | multi-branch + diverse backbones |
| **Output** | entmax (inference) | softmax | softmax | softmax |

---

## Ключові висновки для нашого пайплайну (моделі)

1. **Adan + CosineAnnealingLR** — де-факто стандарт у цьому змаганні (всі хто вказав optimizer)
2. **KLDivLoss** — відповідає метриці змагання; aux loss на проміжних виходах гілок допомагає
3. **2-stage training** — Stage 1 широкі дані, Stage 2 тільки якісні (≥10 votes) — краще ніж один прохід
4. **Дрібні timm моделі** краще за великі — менший overfit, краще в ансамблі
5. **Seed ensemble** (та сама архітектура × 4 seed) — базова стратегія стабільності CV
6. **Fine-tune від ImageNet** — всі без виключення, training from scratch ніхто не використовував
7. **entmax замість softmax на inference** — дає +0.004 LB для розподілів з нулями (yamash)
