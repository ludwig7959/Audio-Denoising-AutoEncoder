import math

import numpy as np
import torch
from torch import nn


def slice_waveform(waveform, length):
    waveform = waveform.numpy()
    num_chunks = math.ceil(waveform.shape[1] / length)
    chunks = []

    for i in range(num_chunks):
        start = i * length
        end = min((i + 1) * length, waveform.shape[1])
        chunk = waveform[:, start:end]
        if chunk.shape[1] < length:
            chunk = np.pad(chunk, ((0, 0), (0, length - chunk.shape[1])), mode='constant', constant_values=0)

        chunks.append(torch.from_numpy(chunk))

    return chunks


def complex_mse_loss(y_true, y_pred):
    real_loss = nn.MSELoss()(y_true.real, y_pred.real)
    imag_loss = nn.MSELoss()(y_true.imag, y_pred.imag)
    return real_loss + imag_loss


def max_normalize(tensor, max_val):
    return tensor / max_val


def max_denormalize(tensor, max_val):
    return tensor * max_val
