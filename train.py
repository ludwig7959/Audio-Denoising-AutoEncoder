import gc
import os
from pathlib import Path

import torch
import torchaudio
from dotenv import load_dotenv
from torch import optim
from torchvision.transforms import transforms

from function import slice_waveform
from model import DCUnet, IDAAE

load_dotenv()

N_FFT = int(os.getenv('N_FFT', 2046))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 512))
DEVICE = torch.device(os.getenv('DEVICE'))


def _stft(waveform):
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hamming_window(window_length=N_FFT),
                      return_complex=True, center=True)
    stft_resized = transforms.Resize((1024, 1024))(stft.squeeze())

    return stft_resized


input_audio_path = Path(os.getenv('INPUT_PATH', 'input'))
if not input_audio_path.is_dir():
    print('Input path doesn''t exist')
    exit(0)
target_audio_path = Path(os.getenv('TARGET_PATH', 'target'))
if not target_audio_path.is_dir():
    print('Target path doesn''t exist')
    exit(0)

input_files = sorted(list(input_audio_path.rglob('*.wav')))
input_data = []
target_data = []
input_abs = []
target_abs = []
for input_file in input_files:
    target_file = target_audio_path / input_file.name
    if not target_file.is_file():
        print(f'Skipping {input_file} because there is no matching target.')
        continue

    input_waveform, input_sr = torchaudio.load(input_file)
    target_waveform, target_sr = torchaudio.load(target_file)

    # Change stereo audio to monoaudio
    if input_waveform.size(0) > 1:
        input_waveform = torch.mean(input_waveform, dim=0, keepdim=True)
    if target_waveform.size(0) > 1:
        target_waveform = torch.mean(target_waveform, dim=0, keepdim=True)

    y_length = int(N_FFT / 2 + 1)
    signal_length = int(y_length * HOP_LENGTH - HOP_LENGTH + 2)
    input_chunks = slice_waveform(input_waveform, signal_length)
    target_chunks = slice_waveform(target_waveform, signal_length)

    for i in range(len(input_chunks)):
        tensor_image = _stft(input_chunks[i])
        tensor_image = tensor_image.unsqueeze(0)
        input_data.append(tensor_image)
        input_abs.append(tensor_image.abs())

    for i in range(len(target_chunks)):
        tensor_image = _stft(target_chunks[i])
        tensor_image = tensor_image.unsqueeze(0)
        target_data.append(tensor_image)
        target_abs.append(tensor_image.abs())

input_stacked = torch.stack(input_abs)
target_stacked = torch.stack(target_abs)

normalize_min = torch.min(input_stacked.min(), target_stacked.min())
normalize_max = torch.max(input_stacked.max(), target_stacked.max())


def min_max_normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)


def create_batches(input_data, target_data, batch_size):
    for i in range(0, len(input_data), batch_size):
        features_batch = min_max_normalize(torch.stack(input_data[i:i + batch_size]), normalize_min, normalize_max)
        labels_batch = min_max_normalize(torch.stack(target_data[i:i + batch_size]), normalize_min, normalize_max)
        yield features_batch, labels_batch


batch_size = int(os.getenv('BATCH_SIZE', 4))
batches = list(create_batches(input_data, target_data, batch_size))

gc.collect()
torch.cuda.empty_cache()

model_type = os.getenv('MODEL_TYPE').lower()
if model_type == 'dcunet':
    model = DCUnet().to(DEVICE)
elif model_type == 'idaae':
    model = IDAAE().to(DEVICE)
else:
    print('Invalid model type')
    print('Available model types: DCUNet, iDAAE')
    exit(0)

optimizer = optim.Adam(model.parameters())
os.makedirs("models", exist_ok=True)

epochs = int(os.getenv('EPOCHS', 30))
for epoch in range(epochs):
    epoch_loss = model.train_epoch(batches)

    model.save(epoch + 1, normalize_min, normalize_max)

    print(f'Epoch {epoch + 1}', end=", ")
    losses = []
    for name, value in epoch_loss:
        losses.append(f'{name}: value')
    print(", ".join(losses))

    gc.collect()
    torch.cuda.empty_cache()
