import librosa
import numpy as np

import soundfile as sf

from config.common import *
from config.preprocess import *


def preprocess(audio, sr):
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    audio_normalized = rms_normalize(audio_resampled, TARGET_RMS)

    return audio_normalized


def rms_normalize(y, target_rms):
    rms = np.sqrt(np.mean(y ** 2))
    scaling_factor = target_rms / rms
    normalized_y = y * scaling_factor

    return normalized_y


if __name__ == '__main__':

    if not os.path.isdir(PREPROCESSING_INPUT_PATH):
        print(f'Directory {PREPROCESSING_INPUT_PATH} doesn''t exist.')
        exit(0)

    os.makedirs(PREPROCESSING_OUTPUT_PATH, exist_ok=True)

    for audio_file_name in os.listdir(PREPROCESSING_INPUT_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(PREPROCESSING_INPUT_PATH, audio_file_name), sr=None, mono=True)
        sf.write(os.path.join(PREPROCESSING_OUTPUT_PATH, audio_file_name), preprocess(audio, sr), samplerate=SAMPLING_RATE)
