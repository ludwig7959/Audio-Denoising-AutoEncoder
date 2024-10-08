import os.path
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

import config.common
from function import max_normalize


class DenoiserDataset(Dataset):
    def __init__(self, input_path, target_path, n_fft=64, hop_length=16, window=torch.hamming_window(64)):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

        self.signal_length = int(config.common.TIME_DOMAIN_SIZE * self.hop_length - self.hop_length + 2)

        self.inputs = []
        self.targets = []
        for file in sorted(list(Path(input_path).rglob('*.wav'))):
            target_file = os.path.join(target_path, file.name)
            if not os.path.isfile(target_file):
                print(f'Skipping {file} because there is no matching target.')
                continue

            input_waveform, _ = torchaudio.load(file)
            target_waveform, _ = torchaudio.load(target_file)

            input_waveform = self._cut(input_waveform)
            target_waveform = self._cut(target_waveform)

            input_stft = torch.stft(input_waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                                    return_complex=True, center=True).squeeze()
            input_stft = transforms.Resize((1024, config.common.TIME_DOMAIN_SIZE))(input_stft).unsqueeze(0)

            target_stft = torch.stft(target_waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                                     return_complex=True, center=True).squeeze()
            target_stft = transforms.Resize((1024, config.common.TIME_DOMAIN_SIZE))(target_stft).unsqueeze(0)

            self.inputs.append(input_stft)
            self.targets.append(target_stft)

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

        self.len_ = len(self.inputs)

        self.max = torch.max(torch.abs(self.inputs).max(), torch.abs(self.targets).max())

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def _cut(self, waveform):
        end = min(self.signal_length, waveform.shape[1])
        cut_waveform = waveform[:, 0:end]
        if cut_waveform.shape[1] < self.signal_length:
            cut_waveform = np.pad(cut_waveform, ((0, 0), (0, self.signal_length - cut_waveform.shape[1])), mode='constant', constant_values=0)

        return cut_waveform

    def normalize(self, max_val):
        self.inputs = max_normalize(self.inputs, max_val)
        self.targets = max_normalize(self.targets, max_val)

    def get_max(self):
        return self.max.item()
