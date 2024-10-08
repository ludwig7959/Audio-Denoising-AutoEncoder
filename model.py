import torch
import torch.nn as nn
from torch import optim

import layer
from function import complex_mse_loss
from config.common import DEVICE


class DCUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = layer.ComplexConv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.batch1 = layer.ComplexBatchNorm2d(num_features=16)
        self.activation1 = layer.ComplexLeakyReLU()

        self.conv2 = layer.ComplexConv2d(in_channels=16, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.batch2 = layer.ComplexBatchNorm2d(num_features=32)
        self.activation2 = layer.ComplexLeakyReLU()

        self.conv3 = layer.ComplexConv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.batch3 = layer.ComplexBatchNorm2d(num_features=64)
        self.activation3 = layer.ComplexLeakyReLU()

        self.conv4 = layer.ComplexConv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2)
        self.batch4 = layer.ComplexBatchNorm2d(num_features=128)
        self.activation4 = layer.ComplexLeakyReLU()

        self.conv5 = layer.ComplexConv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2)
        self.batch5 = layer.ComplexBatchNorm2d(256)
        self.activation5 = layer.ComplexLeakyReLU()

        self.conv6 = layer.ComplexConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.batch6 = layer.ComplexBatchNorm2d(256)
        self.activation6 = layer.ComplexLeakyReLU()

        self.conv7 = layer.ComplexConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, padding=1, stride=2)
        self.batch7 = layer.ComplexBatchNorm2d(128)
        self.activation7 = layer.ComplexLeakyReLU()

        self.conv8 = layer.ComplexConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batch8 = layer.ComplexBatchNorm2d(64)
        self.activation8 = layer.ComplexLeakyReLU()

        self.conv9 = layer.ComplexConvTranspose2d(in_channels=192, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.batch9 = layer.ComplexBatchNorm2d(64)
        self.activation9 = layer.ComplexLeakyReLU()

        self.conv10 = layer.ComplexConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.batch10 = layer.ComplexBatchNorm2d(32)
        self.activation10 = layer.ComplexLeakyReLU()

        self.conv11 = layer.ComplexConvTranspose2d(in_channels=96, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.batch11 = layer.ComplexBatchNorm2d(32)
        self.activation11 = layer.ComplexLeakyReLU()

        self.conv12 = layer.ComplexConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.batch12 = layer.ComplexBatchNorm2d(16)
        self.activation12 = layer.ComplexLeakyReLU()

        self.conv13 = layer.ComplexConvTranspose2d(in_channels=48, out_channels=16, kernel_size=4, padding=1, stride=2)
        self.batch13 = layer.ComplexBatchNorm2d(16)
        self.activation13 = layer.ComplexLeakyReLU()

        self.conv14 = layer.ComplexConvTranspose2d(in_channels=16, out_channels=2, kernel_size=3, padding=1, stride=1)
        self.batch14 = layer.ComplexBatchNorm2d(2)
        self.activation14 = layer.ComplexLeakyReLU()

        self.conv15 = layer.ComplexConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=1)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x1 = self.activation1(self.batch1(self.conv1(x)))
        skip_connection4 = self.activation2(self.batch2(self.conv2(x1)))
        skip_connection3 = self.activation3(self.batch3(self.conv3(skip_connection4)))
        skip_connection2 = self.activation4(self.batch4(self.conv4(skip_connection3)))
        skip_connection1 = self.activation5(self.batch5(self.conv5(skip_connection2)))
        x5 = skip_connection1
        x6 = self.activation6(self.batch6(self.conv6(x5)))

        x7 = torch.cat((skip_connection1, x6), dim=1)
        x7 = self.activation7(self.batch7(self.conv7(x7)))

        x8 = self.activation8(self.batch8(self.conv8(x7)))

        x9 = torch.cat((skip_connection2, x8), dim=1)
        x9 = self.activation9(self.batch9(self.conv9(x9)))

        x10 = self.activation10(self.batch10(self.conv10(x9)))

        x11 = torch.cat((skip_connection3, x10), dim=1)
        x11 = self.activation11(self.batch11(self.conv11(x11)))

        x12 = self.activation12(self.batch12(self.conv12(x11)))

        x13 = torch.cat((skip_connection4, x12), dim=1)
        x13 = self.activation13(self.batch13(self.conv13(x13)))

        x14 = self.activation14(self.batch14(self.conv14(x13)))

        x15 = self.conv15(x14)

        mask_phase = x15 / (torch.abs(x15) + 1e-8)
        mask_magnitude = torch.tanh(torch.abs(x15))
        mask = mask_phase * mask_magnitude

        mask_real = mask.real
        mask_imag = mask.imag

        output_real = mask_real * x.real - mask_imag * x.imag
        output_imag = mask_real * x.imag + mask_imag * x.real

        output = torch.complex(output_real, output_imag)

        return output

    def train_epoch(self, train_loader, validation_loader):
        losses = {'loss': 0.0, 'val_loss': 0.0}
        epoch_loss = 0.0
        num_batches = len(train_loader)
        for input_batch, target_batch in train_loader:
            self.train()
            
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            output = self(input_batch)
            loss = complex_mse_loss(output, target_batch)
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        losses['loss'] = epoch_loss / num_batches

        if validation_loader is not None:
            val_loss = 0.0
            num_batches = len(validation_loader)
            for input_batch, target_batch in train_loader:
                self.eval()

                input_batch = input_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)

                with torch.no_grad():
                    output = self(input_batch)
                    loss = complex_mse_loss(output, target_batch)
                    val_loss += loss.item()

            losses['val_loss'] = val_loss / num_batches

        return losses

    def save(self, name, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalize_max': max
        }, f'models/dcunet_{name}.pth')


class DAAE(nn.Module):

    # sigma: Corruption level
    def __init__(self, sigma=0.5):
        super().__init__()

        self.sigma = sigma

        self.autoencoder = DCUnet()
        self.discriminator = self.Discriminator()

        self.optimizer_autoencoder = optim.RMSprop(self.autoencoder.parameters())
        self.optimizer_discriminator = optim.RMSprop(self.discriminator.parameters())

    def train_epoch(self, train_loader, validation_loader):
        loss = {'generator': 0.0, 'reconstruction': 0.0, 'discriminator': 0.0}
        epoch_loss_generator = 0.0
        epoch_loss_reconstruction = 0.0
        epoch_loss_discriminator = 0.0

        b = 0
        for features_batch, labels_batch in train_loader:
            self.autoencoder.train()
            self.discriminator.train()

            corrupted = features_batch.to(DEVICE)
            clean = labels_batch.to(DEVICE)
            z_fake = self.autoencoder(corrupted)

            self.optimizer_discriminator.zero_grad()
            discriminated_real = self.discriminator(clean)
            discriminated_fake = self.discriminator(z_fake.detach())

            real_labels = torch.ones_like(discriminated_real)
            fake_labels = torch.zeros_like(discriminated_fake)

            discriminator_loss = 0.5 * nn.BCELoss()(discriminated_real, real_labels) + nn.BCELoss()(discriminated_fake, fake_labels)
            discriminator_loss.backward(retain_graph=True)
            self.optimizer_discriminator.step()

            self.optimizer_autoencoder.zero_grad()

            reconstruction_loss = complex_mse_loss(clean, z_fake)
            g_discriminated_fake = self.discriminator(z_fake)
            generator_loss = -torch.mean(torch.log(g_discriminated_fake))
            autoencoder_loss = (0.995 * reconstruction_loss) + (0.005 * generator_loss)
            autoencoder_loss.backward()

            self.optimizer_autoencoder.step()

            epoch_loss_generator += generator_loss.item()
            epoch_loss_reconstruction += reconstruction_loss.item()
            epoch_loss_discriminator += discriminator_loss.item()
            b += 1

        loss['generator'] = epoch_loss_generator / b
        loss['reconstruction'] = epoch_loss_reconstruction
        loss['discriminator'] = epoch_loss_discriminator / b

        return loss

    def forward(self, x):
        return self.autoencoder(x)

    def corrupt(self, x):
        noise_real = self.sigma * torch.randn(x.real.size())
        noise_imag = self.sigma * torch.randn(x.imag.size())

        return torch.complex(x.real + noise_real.to(x.device), x.imag + noise_imag.to(x.device))

    def save(self, name, min, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_autoencoder_state_dict': self.optimizer_autoencoder.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'normalize_max': max
        }, f'models/daae_{name}.pth')

    class AutoEncoder2d(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = layer.ComplexConv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
            self.batch1 = layer.ComplexBatchNorm2d(num_features=16)
            self.activation1 = layer.ComplexLeakyReLU()

            self.conv2 = layer.ComplexConv2d(in_channels=16, out_channels=32, kernel_size=4, padding=1, stride=2)
            self.batch2 = layer.ComplexBatchNorm2d(num_features=32)
            self.activation2 = layer.ComplexLeakyReLU()

            self.conv3 = layer.ComplexConv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2)
            self.batch3 = layer.ComplexBatchNorm2d(num_features=64)
            self.activation3 = layer.ComplexLeakyReLU()

            self.conv4 = layer.ComplexConv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2)
            self.batch4 = layer.ComplexBatchNorm2d(num_features=128)
            self.activation4 = layer.ComplexLeakyReLU()

            self.conv5 = layer.ComplexConv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2)
            self.batch5 = layer.ComplexBatchNorm2d(256)
            self.activation5 = layer.ComplexLeakyReLU()

            self.conv7 = layer.ComplexConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, padding=1, stride=2)
            self.batch7 = layer.ComplexBatchNorm2d(128)
            self.activation7 = layer.ComplexLeakyReLU()

            self.conv8 = layer.ComplexConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
            self.batch8 = layer.ComplexBatchNorm2d(64)
            self.activation8 = layer.ComplexLeakyReLU()

            self.conv9 = layer.ComplexConvTranspose2d(in_channels=192, out_channels=32, kernel_size=4, padding=1, stride=2)
            self.batch9 = layer.ComplexBatchNorm2d(64)
            self.activation9 = layer.ComplexLeakyReLU()

            self.conv10 = layer.ComplexConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, padding=1, stride=2)
            self.batch10 = layer.ComplexBatchNorm2d(32)
            self.activation10 = layer.ComplexLeakyReLU()

        def encode(self, x):
            x = self.activation1(self.batch1(self.conv1(x)))
            x = self.activation2(self.batch2(self.conv2(x)))
            x = self.activation3(self.batch3(self.conv3(x)))
            z = self.activation4(self.batch4(self.conv4(x)))

            return z

        def decode(self, z, x):
            z = self.activation5(self.batch5(self.conv5(z)))
            z = self.activation6(self.batch6(self.conv6(z)))
            z = self.activation7(self.batch7(self.conv7(z)))
            z = self.conv8(z)

            mask_phase = z / (torch.abs(z) + 1e-8)
            mask_magnitude = torch.sigmoid(torch.abs(x))
            mask = mask_phase * mask_magnitude

            mask_real = mask.real
            mask_imag = mask.imag

            output_real = mask_real * x.real - mask_imag * x.imag
            output_imag = mask_real * x.imag + mask_imag * x.real

            output = torch.complex(output_real, output_imag)

            return output

        def forward(self, x):
            z = self.encode(x)
            z = self.decode(z, x)
            return z

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = layer.ComplexConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2)
            self.activation1 = layer.ComplexLeakyReLU()
            self.conv2 = layer.ComplexConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
            self.activation2 = layer.ComplexLeakyReLU()
            self.conv3 = layer.ComplexConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2)
            self.activation3 = layer.ComplexLeakyReLU()

            self.real_layer = nn.Conv2d(32, 1, kernel_size=8, stride=1, padding=0)
            self.sigmoid = nn.Sigmoid()

        def discriminate(self, x):
            x = self.activation1(self.conv1(x))
            x = self.activation2(self.conv2(x))
            x = self.activation3(self.conv3(x))

            magnitude = torch.sqrt(x.real**2 + x.imag**2)
            output = self.real_layer(magnitude)
            output = self.sigmoid(output)

            return output

        def forward(self, z):
            return self.discriminate(z)

