import os
import torch

DATA_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
CKPT_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
LOGS_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'cache', 'eeg'))


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
