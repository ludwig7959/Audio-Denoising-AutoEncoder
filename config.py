import os
from distutils.util import strtobool
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

N_FFT = int(os.getenv('N_FFT', 2046))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 512))
DEVICE = torch.device(os.getenv('DEVICE'))

INPUT_AUDIO_PATH = Path(os.getenv('INPUT_PATH', 'input'))
TARGET_AUDIO_PATH = Path(os.getenv('TARGET_PATH', 'target'))
NORMALIZATION = bool(strtobool(os.getenv('NORMALIZATION', 'True')))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))

MODEL_TYPE = os.getenv('MODEL_TYPE').lower()
EPOCHS = int(os.getenv('EPOCHS', 30))
SAVE_EACH_EPOCH = bool(strtobool(os.getenv('SAVE_EACH_EPOCH', 'True')))

EARLY_STOPPING = bool(strtobool(os.getenv('EARLY_STOPPING', 'True')))
EARLY_STOPPING_LOSS = os.getenv('EARLY_STOPPING_LOSS', 'loss')
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', 5))