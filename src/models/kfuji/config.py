from dataclasses import dataclass


@dataclass
class KfujiConfig:
    # CWT (Paul wavelet)
    lower_freq: float = 0.5
    upper_freq: float = 20.0
    n_scales: int = 40
    cwt_stride: int = 16
    cwt_border_crop: int = 1
    cwt_m: int = 4  # Paul wavelet order; kfuji used m=4 and m=16 as variations

    # Data
    window_samples: int = 10000  # 50 sec × 200 Hz; kfuji also tried 2000 and 5000
    image_size: int = 512
    label_smoothing: float = 0.02

    # Model
    backbone: str = "convnext_small.fb_in1k"  # "maxvit_base_tf_512.in21k_ft_in1k"
    num_classes: int = 6
    pretrained: bool = True

    # Training (used in model.py configure_optimizers)
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 8
    num_workers: int = 4
