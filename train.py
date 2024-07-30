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


def _spectrogram(waveform):
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT), return_complex=True, center=True)
    spectrogram = torch.abs(stft)
    spectrogram_2d = spectrogram[0]
    spectrogram_2d_np = spectrogram_2d.numpy()
    spectrogram_resized = cv2.resize(spectrogram_2d_np, (1024, 1024))

    return spectrogram_resized


noise_audio_path = Path(input('노이즈가 포함된 경로를 입력하세요: '))
noise_files = sorted(list(noise_audio_path.rglob('*.wav')))

noisier_data = []
noisy_data = []
for noise_file in noise_files:
    waveform, sr = torchaudio.load(noise_file)
    noisier = waveform + waveform

    y_length = int(N_FFT / 2 + 1)
    signal_length = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)
    noisier_chunks = _slice(noisier, signal_length)
    noisy_chunks = _slice(waveform, signal_length)

    for i in range(len(noisier_chunks)):
        tensor_image = torch.from_numpy(_spectrogram(noisier_chunks[i])).to(torch.float32)
        tensor_image = tensor_image.unsqueeze(0)
        noisier_data.append(tensor_image)

    for i in range(len(noisy_chunks)):
        tensor_image = torch.from_numpy(_spectrogram(noisy_chunks[i])).to(torch.float32)
        tensor_image = tensor_image.unsqueeze(0)
        noisy_data.append(tensor_image)


def create_batches(noisier_data, noisy_data, batch_size):
    for i in range(0, len(noisier_data), batch_size):
        features_batch = noisier_data[i:i + batch_size]
        labels_batch = noisy_data[i:i + batch_size]
        yield features_batch, labels_batch


batch_size = os.getenv('BATCH_SIZE', 8)
batches = list(create_batches(noisier_data, noisy_data, batch_size))


def l2_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


gc.collect()
torch.cuda.empty_cache()

model = DCUnet().to(DEVICE)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters())
os.makedirs("Weights", exist_ok=True)

epochs = 100
for epoch in range(epochs):
    model.train()
    for features_batch, labels_batch in batches:
        input = torch.stack(features_batch).to(DEVICE)
        label = torch.stack(labels_batch).to(DEVICE)
        output = model(input)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'Weights/model_'+str(epoch+1)+'.pth')
    torch.save(optimizer.state_dict(), 'Weights/opt_'+str(epoch+1)+'.pth')

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    gc.collect()
    torch.cuda.empty_cache()
