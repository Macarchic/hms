from dataclasses import dataclass


@dataclass
class YamashConfig:
    # Signal processing
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    bandpass_order: int = 5

    # Data
    window_samples: int = 10000  # 50 sec × 200 Hz
    image_size: int = 512
    label_smoothing: float = 0.02

    # Model
    backbone: str = "convnext_atto.d2_in1k"
    num_classes: int = 6
    pretrained: bool = True

    # Inference: entmax sharpens predictions vs. softmax
    entmax_alpha: float = 1.03

    # Training
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 4
