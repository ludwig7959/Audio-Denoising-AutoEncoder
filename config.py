import os
from distutils.util import strtobool
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

N_FFT = int(os.getenv('N_FFT', 2046))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 512))
DEVICE = torch.device(os.getenv('DEVICE'))
MODEL_TYPE = os.getenv('MODEL_TYPE').lower()

INPUT_AUDIO_PATH = Path(os.getenv('INPUT_PATH', 'input'))
TARGET_AUDIO_PATH = Path(os.getenv('TARGET_PATH', 'target'))
NORMALIZATION = bool(strtobool(os.getenv('NORMALIZATION', 'True')))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))

EPOCHS = int(os.getenv('EPOCHS', 30))
SAVE_EACH_EPOCH = bool(strtobool(os.getenv('SAVE_EACH_EPOCH', 'True')))

EARLY_STOPPING = bool(strtobool(os.getenv('EARLY_STOPPING', 'True')))
EARLY_STOPPING_LOSS = os.getenv('EARLY_STOPPING_LOSS', 'loss')
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', 5))

MODEL_PATH=os.getenv('MODEL_PATH')
NOISY_PATH=os.getenv('NOISY_PATH')
OUTPUT_PATH=os.getenv('OUTPUT_PATH')