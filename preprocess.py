import librosa
import numpy as np

import soundfile as sf

from config.common import *
from config.preprocess import *


def common_preprocess(audio, sr):
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    audio_normalized = rms_normalize(audio_resampled, TARGET_RMS)

    return audio_normalized


def rms_normalize(y, target_rms):
    rms = np.sqrt(np.mean(y ** 2))
    scaling_factor = target_rms / rms
    normalized_y = y * scaling_factor

    return normalized_y


if __name__ == '__main__':

    if not os.path.isdir(PREPROCESSING_CLEAN_PATH):
        print(f'Directory {PREPROCESSING_CLEAN_PATH} doesn''t exist.')
        exit(0)

    if not os.path.isdir(PREPROCESSING_NOISE_PATH):
        print(f'Directory {PREPROCESSING_NOISE_PATH} doesn''t exist.')
        exit(0)

    os.makedirs(PREPROCESSING_INPUT_PATH, exist_ok=True)
    os.makedirs(PREPROCESSING_TARGET_PATH, exist_ok=True)

    target_samples = int(TIME_DOMAIN_SIZE * HOP_LENGTH - HOP_LENGTH + 2)

    clean_audios = []
    for audio_file_name in os.listdir(PREPROCESSING_CLEAN_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(PREPROCESSING_CLEAN_PATH, audio_file_name), sr=None, mono=True)
        preprocessed = common_preprocess(audio, sr)

        end = min(target_samples, preprocessed.shape[0])
        cut_waveform = preprocessed[:end]
        if cut_waveform.shape[0] < target_samples:
            cut_waveform = np.pad(cut_waveform, (0, target_samples - cut_waveform.shape[0]), mode='constant', constant_values=0)
        clean_audios.append(cut_waveform)

    print('Number of clean audios: ', len(clean_audios))

    noise_audios = []
    for audio_file_name in os.listdir(PREPROCESSING_NOISE_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, sr = librosa.load(os.path.join(PREPROCESSING_NOISE_PATH, audio_file_name), sr=None, mono=True)
        preprocessed = common_preprocess(audio, sr)
        audio_length = len(preprocessed)

        n = int(audio_length / target_samples)
        pad_length = target_samples * (n + 1) - audio_length
        current_length = 0
        audio_to_pad = np.array([], dtype=np.float32)
        while current_length < pad_length:
            clip_length = min(len(audio), pad_length - current_length)
            audio_to_pad = np.concatenate((audio_to_pad, audio[:clip_length]))
            current_length += clip_length
        padded_audio = np.concatenate((audio, audio_to_pad))

        for j in range(n + 1):
            start = target_samples * j
            noise_audios.append(padded_audio[start:start + target_samples])

    print('Number of noise audios: ', len(noise_audios))

    index = 0
    if len(clean_audios) > len(noise_audios):
        for i in range(len(clean_audios)):
            synthesized = clean_audios[i] + noise_audios[i % len(noise_audios)]
            sf.write(os.path.join(PREPROCESSING_INPUT_PATH, f'audio-{index + 1}.wav'), synthesized,
                     samplerate=SAMPLING_RATE)
            sf.write(os.path.join(PREPROCESSING_TARGET_PATH, f'audio-{index + 1}.wav'), clean_audios[i],
                     samplerate=SAMPLING_RATE)

            index += 1
    else:
        for i in range(len(noise_audios)):
            synthesized = clean_audios[i % len(clean_audios)] + noise_audios[i]
            sf.write(os.path.join(PREPROCESSING_INPUT_PATH, f'audio-{index + 1}.wav'), synthesized,
                     samplerate=SAMPLING_RATE)
            sf.write(os.path.join(PREPROCESSING_TARGET_PATH, f'audio-{index + 1}.wav'),
                     clean_audios[i % len(clean_audios)], samplerate=SAMPLING_RATE)

            index += 1
