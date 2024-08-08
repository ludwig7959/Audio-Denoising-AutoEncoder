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


def min_max_normalize(self, tensor, min_val, max_val):
    abs_tensor = torch.abs(tensor)

    normalized_tensor = (abs_tensor - min_val) / (max_val - min_val)

    scaled_tensor = normalized_tensor * 2 - 1

    phase = torch.angle(tensor)
    scaled_tensor = scaled_tensor * torch.exp(1j * phase)

    return scaled_tensor


def min_max_denormalize(tensor, min_val, max_val):
    abs_tensor = torch.abs(tensor)

    normalized_tensor = (abs_tensor + 1) / 2

    denormalized_tensor = normalized_tensor * (max_val - min_val) + min_val

    phase = torch.angle(tensor)
    denormalized_tensor = denormalized_tensor * torch.exp(1j * phase)

    return denormalized_tensor
