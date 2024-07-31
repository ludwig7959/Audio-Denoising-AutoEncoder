import gc
import math
import os

import numpy as np
import torch
import torchaudio
import soundfile as sf
from dotenv import load_dotenv
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
            chunk = np.pad(chunk, ((0, 0), (0, length - chunk.shape[1])), mode='constant', constant_values=0)

        chunks.append(torch.from_numpy(chunk))

    return chunks


def preprocess_audio(file):
    waveform, sr = torchaudio.load(file)
    audio_length = waveform.size(1)

    y_length = int(N_FFT / 2 + 1)
    slice_length = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)

    stfts = []
    shape = (0, 0)

    sliced = _slice(waveform, slice_length)
    for i in range(len(sliced)):
        stft = torch.stft(sliced[i].to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE), return_complex=True)
        shape = stft.squeeze().shape
        stft_resized = transforms.Resize((1024, 1024))(stft.squeeze())
        stfts.append(stft_resized)

    return stfts, sr, shape, audio_length


def min_max_normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)


def min_max_denormalize(tensor, min_val, max_val):
    return tensor * (max_val - min_val) + min_val


gc.collect()
torch.cuda.empty_cache()

model_weights_path = input('Enter the path of the model: ')
dcunet = DCUnet().to(DEVICE)
loaded_model = torch.load(model_weights_path, map_location=DEVICE, weights_only=False)
dcunet.load_state_dict(loaded_model['model_state_dict'])
normalize_min = loaded_model['normalize_min']
normalize_max = loaded_model['normalize_max']

input_path = input('Enter the path of the directory that contains audio files to denoise: ')
output_path = input('Enter the output path: ')

gc.collect()
torch.cuda.empty_cache()

for file in os.listdir(input_path):
    if not file.endswith(".wav"):
        continue

    stfts, sr, shape, audio_length = preprocess_audio(os.path.join(input_path, file))
    waveforms = []
    for i in range(len(stfts)):
        stft = min_max_normalize(stfts[i].to(DEVICE), normalize_min, normalize_max)
        denoised_spectrogram = dcunet.forward(stft.unsqueeze(0).unsqueeze(0)).squeeze()
        resized = transforms.Resize(shape)(denoised_spectrogram)
        denormalized = min_max_denormalize(resized.to(DEVICE), normalize_min, normalize_max)

        waveform = torch.istft(denormalized, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE))
        waveform_numpy = waveform.detach().cpu().numpy()

        if waveform_numpy.ndim == 2:
            waveform_numpy = waveform_numpy[0]

        waveforms.append(waveform_numpy)
    final = np.concatenate(waveforms, axis=-1)[:audio_length]
    sf.write(os.path.join(output_path, file), final, samplerate=sr)

    gc.collect()
    torch.cuda.empty_cache()
