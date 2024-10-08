import gc

import librosa
import numpy as np
import soundfile as sf
from torchvision.transforms import transforms

from config.common import *
from config.denoise import *
from function import slice_waveform, max_normalize, max_denormalize
from model import DCUnet, DAAE
from preprocess import common_preprocess


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
    normalize_max = loaded_weights['normalize_max']

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()

    for file in os.listdir(NOISY_PATH):
        if not file.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(NOISY_PATH, file), sr=None)
        preprocessed = torch.tensor(common_preprocess(audio, sr)).unsqueeze(0)
        audio_length = preprocessed.size(1)

        slice_length = int(TIME_DOMAIN_SIZE * HOP_LENGTH - HOP_LENGTH + 2)

        stfts = []
        shape = (0, 0)
        sliced = slice_waveform(preprocessed, slice_length)
        for i in range(len(sliced)):
            stft = torch.stft(sliced[i].to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE), return_complex=True)
            shape = stft.squeeze().shape
            stft_resized = transforms.Resize((1024, TIME_DOMAIN_SIZE))(stft.squeeze()).unsqueeze(0)
            stfts.append(stft_resized)

        waveforms = []
        for i in range(len(stfts)):
            stft = max_normalize(stfts[i].to(DEVICE), normalize_max)
            denoised_spectrogram = model(stft.unsqueeze(0)).squeeze()
            resized = transforms.Resize(shape)(denoised_spectrogram)
            denormalized = max_denormalize(resized.to(DEVICE), normalize_max)

            waveform = torch.istft(denormalized, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT).to(DEVICE))
            waveform_numpy = waveform.detach().cpu().numpy()

            if waveform_numpy.ndim == 2:
                waveform_numpy = waveform_numpy[0]

            waveforms.append(waveform_numpy)

        final = np.concatenate(waveforms, axis=-1)[:audio_length]
        sf.write(os.path.join(OUTPUT_PATH, file), final, samplerate=sr)

        gc.collect()
        torch.cuda.empty_cache()
