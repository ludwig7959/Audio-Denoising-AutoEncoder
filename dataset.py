import os.path
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms

from function import min_max_normalize


class DenoiserDataset(Dataset):
    def __init__(self, input_path, target_path, n_fft=64, hop_length=16, window=torch.hamming_window(64)):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

        y_length = int(self.n_fft / 2 + 1)
        self.signal_length = int(y_length * self.hop_length - self.hop_length + 2)

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
            input_stft = transforms.Resize((1024, 1024))(input_stft.squeeze())

            target_stft = torch.stft(target_waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                                     return_complex=True, center=True).squeeze()
            target_stft = transforms.Resize((1024, 1024))(target_stft.squeeze())

            self.inputs.append(input_stft)
            self.targets.append(target_stft)

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

        self.len_ = len(self.inputs)

        self.min = torch.min(torch.abs(self.inputs).min(), torch.abs(self.targets).min())
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

    def normalize(self, min_val, max_val):
        self.inputs = min_max_normalize(self.inputs, min_val, max_val)
        self.targets = min_max_normalize(self.targets, min_val, max_val)

    def get_min_max(self):
        return self.min.item(), self.max.item()
