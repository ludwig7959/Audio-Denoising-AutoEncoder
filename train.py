import gc
import os
from datetime import datetime

import torch
import torchaudio
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

import config
from function import slice_waveform, min_max_normalize
from model import DCUnet, DAAE

def _stft(waveform):
    stft = torch.stft(waveform, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, window=torch.hamming_window(window_length=config.N_FFT),
                      return_complex=True, center=True)
    stft_resized = transforms.Resize((1024, 1024))(stft.squeeze())

    return stft_resized

if not config.INPUT_AUDIO_PATH.is_dir():
    print('Input path doesn''t exist')
    exit(0)
if not config.TARGET_AUDIO_PATH.is_dir():
    print('Target path doesn''t exist')
    exit(0)


input_files = sorted(list(config.INPUT_AUDIO_PATH.rglob('*.wav')))
input_data = []
target_data = []
input_abs = []
target_abs = []
for input_file in input_files:
    target_file = config.TARGET_AUDIO_PATH / input_file.name
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

    y_length = int(config.N_FFT / 2 + 1)
    signal_length = int(y_length * config.HOP_LENGTH - config.HOP_LENGTH + 2)
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

normalize_min = torch.min(input_stacked.min(), target_stacked.min()) if config.NORMALIZATION else 0.0
normalize_max = torch.max(input_stacked.max(), target_stacked.max()) if config.NORMALIZATION else 1.0


def create_batches(input_data, target_data, batch_size):
    for i in range(0, len(input_data), batch_size):
        features_batch = min_max_normalize(torch.stack(input_data[i:i + batch_size]), normalize_min, normalize_max)
        labels_batch = min_max_normalize(torch.stack(target_data[i:i + batch_size]), normalize_min, normalize_max)
        yield features_batch, labels_batch


batches = list(create_batches(input_data, target_data, config.BATCH_SIZE))

gc.collect()
torch.cuda.empty_cache()

if config.MODEL_TYPE == 'dcunet':
    model = DCUnet().to(config.DEVICE)
elif config.MODEL_TYPE == 'daae':
    model = DAAE().to(config.DEVICE)
else:
    print('Invalid model type')
    print('Available model types: DCUNet, DAAE')
    exit(0)

class EarlyStopping:
    def __init__(self, loss_type='loss', patience=5, min_delta=0):
        self.loss_type = loss_type
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, losses):
        val_loss = losses[self.loss_type]
        if self.best is None:
            self.best = val_loss
        elif val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


os.makedirs("models", exist_ok=True)
os.makedirs("summary", exist_ok=True)

optimizer = optim.Adam(model.parameters())
early_stopping = EarlyStopping(config.EARLY_STOPPING_LOSS, config.EARLY_STOPPING_PATIENCE)
summary_writer = SummaryWriter(log_dir=f'summary/summary_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

for epoch in range(config.EPOCHS):
    epoch_loss = model.train_epoch(batches)

    print(f'Epoch {epoch + 1}', end=", ")
    losses = []
    for name, value in epoch_loss.items():
        summary_writer.add_scalar(f'Loss/{name}', value, epoch)
        losses.append(f'{name}: {value}')
    print(", ".join(losses))

    # Saving each epoch when set to true
    if config.SAVE_EACH_EPOCH:
        model.save(str(epoch + 1), normalize_min, normalize_max)

    # Early stopping
    early_stopping(epoch_loss)
    if config.EARLY_STOPPING and early_stopping.early_stop:
        print('Early Stopping...')
        model.save('early_stopped', normalize_min, normalize_max)
        summary_writer.close()
        exit(0)

    gc.collect()
    torch.cuda.empty_cache()

model.save('conclusion', normalize_min, normalize_max)
summary_writer.close()
