import random

import librosa
import numpy as np
import soundfile as sf

from config.concatenate import *
from config.common import *

if __name__ == '__main__':
    if not os.path.isdir(CONCATENATING_INPUT_PATH):
        print('Input path doesn''t exist')
        exit(0)

    os.makedirs(CONCATENATING_OUTPUT_PATH, exist_ok=True)

    audio_clips = []
    for audio_file_name in os.listdir(CONCATENATING_INPUT_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, _ = librosa.load(os.path.join(CONCATENATING_INPUT_PATH, audio_file_name), sr=SAMPLING_RATE, mono=True)
        audio_clips.append(audio)

    y_length = int(N_FFT / 2 + 1)
    target_samples = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)

    for i in range(CONCATENATING_COUNT):
        current_samples = 0
        output_audio = np.array([], dtype=np.float32)

        while current_samples < target_samples:
            random_clip = random.choice(audio_clips)
            clip_length = min(len(random_clip), target_samples - current_samples)
            output_audio = np.concatenate((output_audio, random_clip[:clip_length]))
            current_samples += clip_length

        sf.write(os.path.join(CONCATENATING_OUTPUT_PATH, f'concatenated-{i + 1}.wav'), output_audio,
                 samplerate=SAMPLING_RATE)
