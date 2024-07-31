import gc
import math
import os

import cv2
import numpy as np
import torch
import torchaudio
import soundfile as sf
from dotenv import load_dotenv
from matplotlib import pyplot as plt

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
            chunk = np.pad(chunk, ((0, 0), (0, length - chunk.shape[1])), mode='constant', constant_values=0)

        chunks.append(torch.from_numpy(chunk))

    return chunks


def preprocess_audio(file):
    waveform, sr = torchaudio.load(file)
    num_samples = waveform.shape[1]

    y_length = int(N_FFT / 2 + 1)
    signal_length = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)

    spectrograms = []
    phases = []
    shape = (0, 0)

    sliced = _slice(waveform, signal_length)
    for i in range(len(sliced)):
        stft = torch.stft(sliced[i].to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE), return_complex=True)
        spectrogram = torch.abs(stft)
        phase = torch.angle(stft)
        spectrogram_np = spectrogram.cpu().numpy()
        shape = spectrogram_np.shape
        spectrogram = torch.from_numpy(cv2.resize(np.transpose(spectrogram_np, (1, 2, 0)), (1024, 1024))).to(torch.float32)
        spectrogram = spectrogram.squeeze().unsqueeze(0)
        spectrograms.append(spectrogram)
        phases.append(phase)

    return spectrograms, phases, sr, shape


def min_max_normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)


def min_max_denormalize(tensor, min_val, max_val):
    return tensor * (max_val - min_val) + min_val

gc.collect()
torch.cuda.empty_cache()

model_weights_path = "models/model_100.pth"
dcunet = DCUnet().to(DEVICE)
loaded_model = torch.load(model_weights_path, weights_only=False)
dcunet.load_state_dict(loaded_model['model_state_dict'])
normalize_min = loaded_model['normalize_min']
normalize_max = loaded_model['normalize_max']

input_path = input('노이즈를 제거할 오디오 파일들의 경로를 입력하세요: ')
output_path = input('출력물 경로를 입력하세요: ')


for file in os.listdir(input_path):
    if not file.endswith(".wav"):
        continue

    spectrograms, phases, sr, shape = preprocess_audio(os.path.join(input_path, file))
    waveforms = []
    for i in range(len(spectrograms)):
        spectrogram = min_max_normalize(spectrograms[i].to(DEVICE), normalize_min, normalize_max)
        phase = phases[i]

        denoised_spectrogram = dcunet.forward(spectrogram.unsqueeze(0)).detach().cpu().numpy().squeeze()
        resized = cv2.resize(denoised_spectrogram, (shape[1], shape[0]))

        complex_spectrogram = min_max_denormalize(torch.tensor(resized, dtype=torch.float32).to(DEVICE) * torch.exp(1j * phase).squeeze(), normalize_min, normalize_max)

        waveform = torch.istft(complex_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE))
        waveform_numpy = waveform.detach().cpu().numpy()

        if waveform_numpy.ndim == 2:
            waveform_numpy = waveform_numpy[0]

        waveforms.append(waveform_numpy)
    final = np.concatenate(waveforms, axis=-1)
    sf.write(os.path.join(output_path, file), final, samplerate=sr)

    gc.collect()
    torch.cuda.empty_cache()
