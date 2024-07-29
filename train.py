import gc
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torch import optim

from model import DCUnet

load_dotenv()

N_FFT = int(os.getenv('N_FFT', 2048))
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


def _spectrogram(waveform):
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT), return_complex=True)
    spectrogram = torch.abs(stft)
    spectrogram_2d = spectrogram[0]
    spectrogram_2d_np = spectrogram_2d.numpy()
    cmap = plt.get_cmap('viridis')
    spectrogram_rgb = cmap(spectrogram_2d_np / np.max(spectrogram_2d_np))[:, :, :3]
    spectrogram_rgb = cv2.resize(spectrogram_rgb, (1024, 1024))

    return spectrogram_rgb


noise_audio_path = Path(input('노이즈가 포함된 경로를 입력하세요: '))
noise_files = sorted(list(noise_audio_path.rglob('*.wav')))

noisier_data = []
noisy_data = []
for noise_file in noise_files:
    waveform, sr = torchaudio.load(noise_file)
    noisier = waveform + waveform

    y_length = int(N_FFT / 2 + 1)
    signal_length = int(y_length * HOP_LENGTH - N_FFT + HOP_LENGTH)
    noisier_chunks = _slice(noisier, signal_length)
    noisy_chunks = _slice(waveform, signal_length)

    for i in range(len(noisier_chunks)):
        tensor_image = torch.from_numpy(_spectrogram(noisier_chunks[i])).to(torch.float32)
        tensor_image = tensor_image.permute(2, 0, 1)
        noisier_chunks[i] = tensor_image

    for i in range(len(noisy_chunks)):
        tensor_image = torch.from_numpy(_spectrogram(noisy_chunks[i])).to(torch.float32)
        tensor_image = tensor_image.permute(2, 0, 1)
        noisy_chunks[i] = tensor_image

    noisier_data.append(noisier_chunks)
    noisy_data.append(noisy_chunks)


def create_batches(noisier_data, noisy_data, batch_size):
    for i in range(0, len(noisier_data), batch_size):
        features_batch = noisier_data[i:i + batch_size]
        labels_batch = noisy_data[i:i + batch_size]
        yield features_batch, labels_batch


batch_size = 2
batches = list(create_batches(noisier_data, noisy_data, batch_size))


def l2_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


gc.collect()
torch.cuda.empty_cache()

model = DCUnet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
os.makedirs("Weights", exist_ok=True)

epochs = 100
for epoch in range(epochs):
    model.train()
    for features_batch, labels_batch in batches:
        model.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True, device=DEVICE)
        for i in range(len(features_batch)):
            for j in range(len(features_batch[i])):
                input = features_batch[i][j].unsqueeze(0).to(DEVICE)
                label = labels_batch[i][j].unsqueeze(0).to(DEVICE)
                output = model(input)
                loss += l2_loss(output, label)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'Weights/model_'+str(epoch+1)+'.pth')
    torch.save(optimizer.state_dict(), 'Weights/opt_'+str(epoch+1)+'.pth')

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    gc.collect()
    torch.cuda.empty_cache()
