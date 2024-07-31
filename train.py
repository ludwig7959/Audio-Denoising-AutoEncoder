import gc
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from torch import optim, nn
from torchvision.transforms import transforms

from model import DCUnet

load_dotenv()

N_FFT = int(os.getenv('N_FFT', 2046))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 512))
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

def _slice(waveform, length):
    waveform = waveform.numpy()
    num_chunks = math.ceil(waveform.shape[1] / length)
    chunks = []

    for i in range(num_chunks):
        start = i * length
        end = min((i + 1) * length, waveform.shape[1])
        chunk = waveform[:, start:end]
        if chunk.shape[1] < length:
            chunk = np.pad(chunk, ((0, 0), (0, length - chunk.shape[1])), mode='constant')

        chunks.append(torch.from_numpy(chunk))

    return chunks


def _stft(waveform):
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT), return_complex=True, center=True)
    stft_resized = transforms.Resize((1024, 1024))(stft.squeeze())

    return stft_resized


noise_audio_path = Path(input('노이즈가 포함된 경로를 입력하세요: '))
noise_files = sorted(list(noise_audio_path.rglob('*.wav')))

noisier_data = []
noisy_data = []
noisier_abs = []
noisy_abs = []
for noise_file in noise_files:
    waveform, sr = torchaudio.load(noise_file)
    noisier = waveform + waveform

    y_length = int(N_FFT / 2 + 1)
    signal_length = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)
    noisier_chunks = _slice(noisier, signal_length)
    noisy_chunks = _slice(waveform, signal_length)

    for i in range(len(noisier_chunks)):
        tensor_image = _stft(noisier_chunks[i])
        tensor_image = tensor_image.unsqueeze(0)
        noisier_data.append(tensor_image)
        noisier_abs.append(tensor_image.abs())

    for i in range(len(noisy_chunks)):
        tensor_image = _stft(noisy_chunks[i])
        tensor_image = tensor_image.unsqueeze(0)
        noisy_data.append(tensor_image)
        noisy_abs.append(tensor_image.abs())

input_stacked = torch.stack(noisier_abs)
target_stacked = torch.stack(noisy_abs)

normalize_min = torch.min(input_stacked.min(), target_stacked.min())
normalize_max = torch.max(input_stacked.max(), target_stacked.max())


def min_max_normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)


def create_batches(noisier_data, noisy_data, batch_size):
    for i in range(0, len(noisier_data), batch_size):
        features_batch = noisier_data[i:i + batch_size]
        labels_batch = noisy_data[i:i + batch_size]
        yield features_batch, labels_batch


batch_size = int(os.getenv('BATCH_SIZE', 8))
batches = list(create_batches(noisier_data, noisy_data, batch_size))


def l2_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


gc.collect()
torch.cuda.empty_cache()

model = DCUnet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
os.makedirs("models", exist_ok=True)

epochs = 100
for epoch in range(epochs):
    model.train()
    for features_batch, labels_batch in batches:
        input = min_max_normalize(torch.stack(features_batch).to(DEVICE), normalize_min, normalize_max)
        label = min_max_normalize(torch.stack(labels_batch).to(DEVICE), normalize_min, normalize_max)
        output = model(input)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'normalize_min': normalize_min,
        'normalize_max': normalize_max
    }, 'models/model_' + str(epoch+1) + '.pth')

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    gc.collect()
    torch.cuda.empty_cache()
