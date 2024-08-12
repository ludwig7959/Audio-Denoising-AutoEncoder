import random

import librosa
import soundfile as sf

from config.common import *
from config.synthesize import *

if __name__ == '__main__':
    if not os.path.isdir(SYNTHESIZING_CLEAN_PATH):
        print('Clean audio path doesn''t exist')
        exit(0)

    if not os.path.isdir(SYNTHESIZING_NOISE_PATH):
        print('Noise audio path doesn''t exist')
        exit(0)

    os.makedirs(SYNTHESIZING_OUTPUT_INPUT, exist_ok=True)
    os.makedirs(SYNTHESIZING_OUTPUT_TARGET, exist_ok=True)

    clean_audios = []
    for audio_file_name in os.listdir(SYNTHESIZING_CLEAN_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, _ = librosa.load(os.path.join(SYNTHESIZING_CLEAN_PATH, audio_file_name), sr=SAMPLING_RATE, mono=True)
        clean_audios.append(audio)

    noise_audios = []
    for audio_file_name in os.listdir(SYNTHESIZING_NOISE_PATH):
        if not audio_file_name.endswith(".wav"):
            continue

        audio, _ = librosa.load(os.path.join(SYNTHESIZING_NOISE_PATH, audio_file_name), sr=SAMPLING_RATE, mono=True)
        noise_audios.append(audio)

    index = 0
    for i in range(len(clean_audios)):
        random.shuffle(noise_audios)
        for j in range(NOISES_PER_CLEAN):
            synthesized = clean_audios[i] + noise_audios[j]
            sf.write(os.path.join(SYNTHESIZING_OUTPUT_INPUT, f'audio-{index + 1}.wav'), synthesized,
                     samplerate=SAMPLING_RATE)
            sf.write(os.path.join(SYNTHESIZING_OUTPUT_TARGET, f'audio-{index + 1}.wav'), clean_audios[i],
                     samplerate=SAMPLING_RATE)

            index += 1
