import gc

import numpy as np
import soundfile as sf
import torchaudio
from torchvision.transforms import transforms

from config.common import *
from config.denoise import *
from function import slice_waveform, min_max_normalize, min_max_denormalize
from model import DCUnet, DAAE
from preprocess import preprocess


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    if MODEL_TYPE == 'dcunet':
        model = DCUnet().to(DEVICE)
    elif MODEL_TYPE == 'daae':
        model = DAAE().to(DEVICE)
    else:
        print('Invalid model type')
        print('Available model types: DCUNet, DAAE')
        exit(0)

    loaded_weights = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(loaded_weights['model_state_dict'])
    normalize_min = loaded_weights['normalize_min']
    normalize_max = loaded_weights['normalize_max']

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    for file in os.listdir(NOISY_PATH):
        if not file.endswith(".wav"):
            continue

        audio, sr = torchaudio.load(os.path.join(NOISY_PATH, file))
        preprocessed = preprocess(audio)
        audio_length = preprocessed.size(1)

        y_length = int(N_FFT / 2 + 1)
        slice_length = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)

        stfts = []
        shape = (0, 0)
        sliced = slice_waveform(preprocessed, slice_length)
        for i in range(len(sliced)):
            stft = torch.stft(sliced[i].to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE), return_complex=True)
            shape = stft.squeeze().shape
            stft_resized = transforms.Resize((1024, 1024))(stft.squeeze())
            stfts.append(stft_resized)

        waveforms = []
        for i in range(len(stfts)):
            stft = min_max_normalize(stfts[i].to(DEVICE), normalize_min, normalize_max)
            denoised_spectrogram = model(stft.unsqueeze(0)).squeeze()
            resized = transforms.Resize(shape)(denoised_spectrogram)
            denormalized = min_max_denormalize(resized.to(DEVICE), normalize_min, normalize_max)

            waveform = torch.istft(denormalized, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE))
            waveform_numpy = waveform.detach().cpu().numpy()

            if waveform_numpy.ndim == 2:
                waveform_numpy = waveform_numpy[0]

            waveforms.append(waveform_numpy)

        final = np.concatenate(waveforms, axis=-1)[:audio_length]
        sf.write(os.path.join(OUTPUT_PATH, file), final, samplerate=sr)

        gc.collect()
        torch.cuda.empty_cache()
