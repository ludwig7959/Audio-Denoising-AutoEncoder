import torch

import config.common

HAMMING_WINDOW = torch.hamming_window(window_length=config.common.N_FFT)