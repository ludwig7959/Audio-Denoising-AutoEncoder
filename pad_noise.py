import librosa
import numpy as np
import soundfile as sf

from config.common import *
from config.pad_noise import *

if __name__ == '__main__':
    if not os.path.isdir(PADDING_INPUT_PATH):
        print('Input path doesn''t exist')
        exit(0)

    os.makedirs(PADDING_OUTPUT_PATH, exist_ok=True)

    y_length = int(N_FFT / 2 + 1)
    target_samples = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)

    index = 0
    for audio_file_name in os.listdir(PADDING_INPUT_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, _ = librosa.load(os.path.join(PADDING_INPUT_PATH, audio_file_name), sr=SAMPLING_RATE, mono=True)
        audio_length = len(audio)

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

            sf.write(os.path.join(PADDING_OUTPUT_PATH, f'padded-{index + 1}.wav'),
                     padded_audio[start:start + target_samples], samplerate=SAMPLING_RATE)

            index += 1
