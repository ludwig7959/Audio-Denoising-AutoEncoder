import gc
from datetime import datetime

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.common import *
from config.train import *
from dataset import DenoiserDataset
from model import DCUnet, DAAE
from window import HAMMING_WINDOW

if __name__ == '__main__':
    if not os.path.isdir(TRAINING_INPUT_PATH):
        print('Input path doesn''t exist')
        exit(0)
    if not os.path.isdir(TRAINING_TARGET_PATH):
        print('Target path doesn''t exist')
        exit(0)

    train_dataset = DenoiserDataset(TRAINING_INPUT_PATH, TRAINING_TARGET_PATH, n_fft=N_FFT, hop_length=HOP_LENGTH, window=HAMMING_WINDOW)
    if NORMALIZATION:
        normalize_max = train_dataset.get_max()
    else:
        normalize_max = 1.0
    train_dataset.normalize(normalize_max)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validation_dataloader = None
    if VALIDATION:
        validation_dataset = DenoiserDataset(VALIDATION_INPUT_PATH, VALIDATION_TARGET_PATH, n_fft=N_FFT, hop_length=HOP_LENGTH, window=HAMMING_WINDOW)
        validation_dataset.normalize(normalize_max)
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

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
    early_stopping = EarlyStopping(EARLY_STOPPING_LOSS, EARLY_STOPPING_PATIENCE)
    summary_writer = SummaryWriter(log_dir=f'summary/summary_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    for epoch in range(EPOCHS):
        epoch_loss = model.train_epoch(train_dataloader, validation_dataloader)

        print(f'Epoch {epoch + 1}', end=", ")
        losses = []
        for name, value in epoch_loss.items():
            summary_writer.add_scalar(f'Loss/{name}', value, epoch)
            losses.append(f'{name}: {value}')
        print(", ".join(losses))

        # Saving each epoch when set to true
        if SAVE_EACH_EPOCH:
            model.save(str(epoch + 1), normalize_max)

        # Early stopping
        early_stopping(epoch_loss)
        if EARLY_STOPPING and early_stopping.early_stop:
            print('Early Stopping...')
            model.save('early_stopped', normalize_max)
            summary_writer.close()
            exit(0)

        gc.collect()
        torch.cuda.empty_cache()

    model.save('conclusion', normalize_max)
    summary_writer.close()
