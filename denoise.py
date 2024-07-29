import gc
import os

import cv2
import numpy as np
import torch
import torchaudio
import soundfile as sf
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from skimage.color import rgb2gray

from model import DCUnet

load_dotenv()

N_FFT = int(os.getenv('N_FFT', 3072))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 768))

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

def preprocess_audio(file):
    waveform, sr = torchaudio.load(file)

    waveform = waveform.cpu().numpy()
    num_samples = waveform.shape[1]

    y_length = int(N_FFT / 2 + 1)
    signal_length = int(y_length * HOP_LENGTH - N_FFT + HOP_LENGTH)

    spectrograms = []
    phases = []
    shape = (0, 0)
    for start in range(0, num_samples, signal_length):
        end = min(start + signal_length, num_samples)
        segment = torch.tensor(waveform[:, start:end], dtype=torch.float32, device=DEVICE)

        stft = torch.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE), return_complex=True)
        spectrogram = torch.abs(stft)
        phase = torch.angle(stft)
        spectrogram_2d = spectrogram[0]
        spectrogram_2d_np = spectrogram_2d.cpu().numpy()
        shape = spectrogram_2d_np.shape
        cmap = plt.get_cmap('viridis')
        spectrogram_rgb = cmap(spectrogram_2d_np / np.max(spectrogram_2d_np))[:, :, :3]
        spectrogram_rgb = cv2.resize(spectrogram_rgb, (1024, 1024))
        spectrograms.append(spectrogram_rgb)
        phases.append(phase)

    return spectrograms, phases, sr, shape

gc.collect()
torch.cuda.empty_cache()

model_weights_path = "Weights/model_100.pth"
dcunet = DCUnet().to(DEVICE)
weights = torch.load(model_weights_path, map_location=torch.device(DEVICE), weights_only=True)
dcunet.load_state_dict(weights)

input_path = input('노이즈를 제거할 오디오 파일들의 경로를 입력하세요: ')
output_path = input('출력물 경로를 입력하세요: ')


for file in os.listdir(input_path):
    if not file.endswith(".wav"):
        continue

    spectrograms, phases, sr, shape = preprocess_audio(os.path.join(input_path, file))
    waveforms = []
    for i in range(len(spectrograms)):
        spectrogram = spectrograms[i]
        phase = phases[i]

        vmin = np.min(spectrogram)
        vmax = np.max(spectrogram)

        tensor_image = torch.from_numpy(spectrogram).to(torch.float32)
        tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        denoised_spectrogram = dcunet.forward(tensor_image).detach().cpu().numpy().squeeze()
        denoised_spectrogram = np.transpose(denoised_spectrogram, (1, 2, 0))
        resized = cv2.resize(denoised_spectrogram, (shape[1], shape[0]))
        gray_image = rgb2gray(resized)
        norm = Normalize(vmin=vmin, vmax=vmax)
        spectrogram_normalized = norm(gray_image)
        spectrogram_original = spectrogram_normalized * (vmax - vmin) + vmin

        print(spectrogram_original.shape)
        print(phase.shape)
        complex_spectrogram = torch.tensor(spectrogram_original, dtype=torch.float32).to(DEVICE) * torch.exp(1j * phase).squeeze()
        complex_spectrogram_tensor = torch.tensor(complex_spectrogram, dtype=torch.cfloat)

        waveform = torch.istft(complex_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE))
        waveform_numpy = waveform.detach().cpu().numpy()

        if waveform_numpy.ndim == 2:
            waveform_numpy = waveform_numpy[0]

        waveforms.append(waveform_numpy)
    final = np.concatenate(waveforms, axis=-1)
    sf.write(os.path.join(output_path, file), final, samplerate=sr)

    gc.collect()
    torch.cuda.empty_cache()
