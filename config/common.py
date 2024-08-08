import os

import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device(os.getenv('DEVICE'))
N_FFT = int(os.getenv('N_FFT', 2046))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 512))
SAMPLING_RATE = int(os.getenv('SAMPLING_RATE', 16000))
MODEL_TYPE = os.getenv('MODEL_TYPE').lower()