import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import binary_cross_entropy

import layer
from function import complex_mse_loss


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
        output_real = torch.squeeze(output_real, 1)
        output_imag = mask_real * x.imag + mask_imag * x.real
        output_imag = torch.squeeze(output_imag, 1)

        output = torch.complex(output_real, output_imag)

        return output

    def train_epoch(self, batches):
        loss = {'loss': 0.0}
        epoch_loss = 0.0
        for features_batch, labels_batch in batches:
            self.train()

            input = features_batch
            label = labels_batch
            output = self(input)
            loss = complex_mse_loss(output, label)
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss['loss'] = epoch_loss

    def save(self, name, min, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalize_min': min,
            'normalize_max': max
        }, f'models/dcunet_{name}.pth')


class DAAE(nn.Module):

    # sigma: Corruption level
    def __init__(self, sigma=0.1):
        super().__init__()

        self.sigma = sigma

        self.autoencoder = DCUnet()
        self.discriminator = self.Discriminator()

        self.optimizer_autoencoder = optim.RMSprop(self.autoencoder.parameters())
        self.optimizer_discriminator = optim.RMSprop(self.discriminator.parameters())

    def train_epoch(self, batches):
        loss = {'generator': 0.0, 'reconstruction': 0.0, 'discriminator': 0.0}
        epoch_loss_generator = 0.0
        epoch_loss_reconstruction = 0.0
        epoch_loss_discriminator = 0.0

        for features_batch, labels_batch in batches:
            self.autoencoder.train()
            self.discriminator.train()

            input = features_batch.to(next(self.parameters()).device)
            corrupted = self.corrupt(input)

            z_fake = self.autoencoder(corrupted)

            self.optimizer_discriminator.zero_grad()
            discriminated_real = self.discriminator(input)
            discriminated_fake = self.discriminator(z_fake.detach())
            discriminator_loss = 0.005 * torch.mean(
                binary_cross_entropy(discriminated_real, torch.ones_like(discriminated_real, device=next(self.parameters()).device)) +
                binary_cross_entropy(discriminated_fake, torch.zeros_like(discriminated_fake)))
            discriminator_loss.backward(retain_graph=True)
            self.optimizer_discriminator.step()

            self.optimizer_autoencoder.zero_grad()

            reconstruction_loss = complex_mse_loss(input, z_fake)
            g_discriminated_fake = self.discriminator(z_fake)
            generator_loss = torch.mean(binary_cross_entropy(g_discriminated_fake, torch.ones_like(g_discriminated_fake)))
            autoencoder_loss = (0.995 * reconstruction_loss) + (0.005 * generator_loss)
            autoencoder_loss.backward()

            self.optimizer_autoencoder.step()

            epoch_loss_generator += generator_loss.item()
            epoch_loss_reconstruction += reconstruction_loss.item()
            epoch_loss_discriminator += discriminator_loss.item()

        loss['generator'] = epoch_loss_generator
        loss['reconstruction'] = epoch_loss_reconstruction
        loss['discriminator'] = epoch_loss_discriminator

        return loss

    def corrupt(self, x):
        noise_real = self.sigma * torch.randn(x.real.size())
        noise_imag = self.sigma * torch.randn(x.imag.size())

        return torch.complex(x.real + noise_real.to(x.device), x.imag + noise_imag.to(x.device))

    def save(self, name, min, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_autoencoder_state_dict': self.optimizer_autoencoder.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'normalize_min': min,
            'normalize_max': max
        }, f'models/daae_{name}.pth')

    class AutoEncoder2d(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = layer.ComplexConv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=2)
            self.batch1 = layer.ComplexBatchNorm2d(num_features=16)
            self.activation1 = layer.ComplexLeakyReLU()
            self.conv2 = layer.ComplexConv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
            self.batch2 = layer.ComplexBatchNorm2d(num_features=32)
            self.activation2 = layer.ComplexLeakyReLU()
            self.conv3 = layer.ComplexConv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)
            self.batch3 = layer.ComplexBatchNorm2d(num_features=64)
            self.activation3 = layer.ComplexLeakyReLU()
            self.conv4 = layer.ComplexConv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
            self.batch4 = layer.ComplexBatchNorm2d(num_features=128)
            self.activation4 = layer.ComplexLeakyReLU()

            self.conv5 = layer.ComplexConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.batch5 = layer.ComplexBatchNorm2d(num_features=64)
            self.activation5 = layer.ComplexLeakyReLU()
            self.conv6 = layer.ComplexConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.batch6 = layer.ComplexBatchNorm2d(num_features=32)
            self.activation6 = layer.ComplexLeakyReLU()
            self.conv7 = layer.ComplexConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.batch7 = layer.ComplexBatchNorm2d(num_features=16)
            self.activation7 = layer.ComplexLeakyReLU()
            self.conv8 = layer.ComplexConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=2, output_padding=1)

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

            self.conv1 = layer.ComplexConv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=2)
            self.activation1 = layer.ComplexLeakyReLU()
            self.conv2 = layer.ComplexConv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
            self.activation2 = layer.ComplexLeakyReLU()
            self.conv3 = layer.ComplexConv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)
            self.activation3 = layer.ComplexLeakyReLU()
            self.conv4 = layer.ComplexConv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=2)

            self.linear5 = layer.ComplexLinear(in_features=64 * 64 * 64, out_features=1024)
            self.activation5 = layer.ComplexLeakyReLU()
            self.linear6 = layer.ComplexLinear(in_features=1024, out_features=256)
            self.activation6 = layer.ComplexLeakyReLU()

            self.linear7 = nn.Linear(in_features=512, out_features=128)
            self.activation7 = nn.LeakyReLU()
            self.linear8 = nn.Linear(in_features=128, out_features=16)
            self.activation8 = nn.LeakyReLU()
            self.linear9 = nn.Linear(in_features=16, out_features=1)
            self.activation9 = nn.Sigmoid()

        def discriminate(self, x):
            x = self.activation1(self.conv1(x))
            x = self.activation2(self.conv2(x))
            x = self.activation3(self.conv3(x))
            x = self.conv4(x)

            x = self.activation5(self.linear5(torch.complex(x.real.view(x.size(0), -1), x.imag.view(x.size(0), -1))))
            x = self.activation6(self.linear6(x))

            x = self.activation7(self.linear7(torch.cat((torch.abs(x), torch.angle(x)), dim=1)))
            x = self.activation8(self.linear8(x))
            x = self.activation9(self.linear9(x))

            return x

        def forward(self, z):
            return self.discriminate(z)

