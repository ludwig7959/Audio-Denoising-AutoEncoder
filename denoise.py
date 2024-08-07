import gc
import os

import numpy as np
import torch
import torchaudio
import soundfile as sf
from dotenv import load_dotenv
from torchvision.transforms import transforms

import config
from function import slice_waveform, min_max_normalize, min_max_denormalize
from model import DCUnet, DAAE


def preprocess_audio(file):
    waveform, sr = torchaudio.load(file)
    audio_length = waveform.size(1)

    y_length = int(config.N_FFT / 2 + 1)
    slice_length = int(y_length * config.HOP_LENGTH - config.HOP_LENGTH + 2)

    stfts = []
    shape = (0, 0)

    sliced = slice_waveform(waveform, slice_length)
    for i in range(len(sliced)):
        stft = torch.stft(sliced[i].to(config.DEVICE), n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, window=torch.hamming_window(window_length=config.N_FFT).to(config.DEVICE), return_complex=True)
        shape = stft.squeeze().shape
        stft_resized = transforms.Resize((1024, 1024))(stft.squeeze())
        stfts.append(stft_resized)

    return stfts, sr, shape, audio_length


gc.collect()
torch.cuda.empty_cache()

if config.MODEL_TYPE == 'dcunet':
    model = DCUnet().to(config.DEVICE)
elif config.MODEL_TYPE == 'daae':
    model = DAAE().to(config.DEVICE)
else:
    print('Invalid model type')
    print('Available model types: DCUNet, DAAE')
    exit(0)
loaded_weights = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
model.load_state_dict(loaded_weights['model_state_dict'])
normalize_min = loaded_weights['normalize_min']
normalize_max = loaded_weights['normalize_max']

os.makedirs(config.OUTPUT_PATH, exist_ok=True)

gc.collect()
torch.cuda.empty_cache()

for file in os.listdir(config.NOISY_PATH):
    if not file.endswith(".wav"):
        continue

    stfts, sr, shape, audio_length = preprocess_audio(os.path.join(config.NOISY_PATH, file))
    waveforms = []
    for i in range(len(stfts)):
        stft = min_max_normalize(stfts[i].to(config.DEVICE), normalize_min, normalize_max)
        denoised_spectrogram = model(stft.unsqueeze(0)).squeeze()
        resized = transforms.Resize(shape)(denoised_spectrogram)
        denormalized = min_max_denormalize(resized.to(config.DEVICE), normalize_min, normalize_max)

        waveform = torch.istft(denormalized, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, window=torch.hamming_window(window_length=config.N_FFT).to(config.DEVICE))
        waveform_numpy = waveform.detach().cpu().numpy()

        if waveform_numpy.ndim == 2:
            waveform_numpy = waveform_numpy[0]

        waveforms.append(waveform_numpy)
    final = np.concatenate(waveforms, axis=-1)[:audio_length]
    sf.write(os.path.join(config.OUTPUT_PATH, file), final, samplerate=sr)

    gc.collect()
    torch.cuda.empty_cache()
