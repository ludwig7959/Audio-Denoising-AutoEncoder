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
        loss = {'Loss': []}
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

        loss['Loss'].append(epoch_loss)

    def save(self, epoch, min, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalize_min': min,
            'normalize_max': max
        }, 'models/dcunet_' + str(epoch + 1) + '.pth')


class IDAAE(nn.Module):

    # sigma: Corruption level
    # M: Corruption repeat count
    def __init__(self, sigma = 0.1, M = 20):
        super().__init__()

        self.sigma = sigma
        self.M = M

        self.autoencoder = self.AutoEncoder2d()
        self.discriminator = self.Discriminator()
        self.linear_svm = LinearSVM(0.5)

        self.optimizer_autoencoder = optim.RMSprop(self.autoencoder.parameters())
        self.optimizer_discriminator = optim.RMSprop(self.discriminator.parameters())

    def train_epoch(self, batches):
        loss = {'Encoder Loss': [], 'Rec. Loss': [], 'Dis. Loss': []}
        epoch_loss_encoder = 0.0
        epoch_loss_reconstruction = 0.0
        epoch_loss_discriminator = 0.0

        torch.autograd.set_detect_anomaly(True)
        for features_batch, labels_batch in batches:
            self.autoencoder.train()
            self.discriminator.train()

            input = features_batch.to(next(self.parameters()).device)
            label = labels_batch.to(next(self.parameters()).device)
            corrupted = self.corrupt(input)

            z_fake = self.autoencoder.encode(corrupted)
            z_real = self.sample_z(z_fake.size(0))

            self.optimizer_discriminator.zero_grad()
            output_real = self.discriminator(z_real)
            output_fake = self.discriminator(z_fake.detach())
            discriminator_loss = 0.5 * torch.mean(
                binary_cross_entropy(output_real.real, torch.ones_like(output_real.real, device=next(self.parameters()).device)) +
                binary_cross_entropy(output_real.imag, torch.ones_like(output_real.imag, device=next(self.parameters()).device)) +
                binary_cross_entropy(output_fake.real, torch.zeros_like(output_fake.real)) +
                binary_cross_entropy(output_fake.imag, torch.zeros_like(output_fake.imag)))
            discriminator_loss.backward()
            self.optimizer_discriminator.step()

            self.optimizer_autoencoder.zero_grad()
            z_fake_approx = 0
            for _ in range(self.M):
                x_tilde = self.corrupt(input)
                z_tilde = self.autoencoder.encode(x_tilde)
                z_fake_approx += z_tilde
            z_fake_approx /= self.M
            output_fake = self.discriminator(z_fake_approx)
            encoder_loss = torch.mean(
                binary_cross_entropy(output_fake.real, torch.ones_like(output_fake.real, device=next(self.parameters()).device)) +
                binary_cross_entropy(output_fake.imag, torch.ones_like(output_fake.imag, device=next(self.parameters()).device)))

            reconstructed = self.autoencoder.decode(z_fake, corrupted)
            reconstruction_loss = torch.mean(complex_mse_loss(reconstructed, input))
            autoencoder_loss = encoder_loss + reconstruction_loss
            autoencoder_loss.backward()
            self.optimizer_autoencoder.step()

            epoch_loss_encoder += encoder_loss.item()
            epoch_loss_reconstruction += reconstruction_loss.item()
            epoch_loss_discriminator += discriminator_loss.item()

        torch.autograd.set_detect_anomaly(False)

        loss['Encoder Loss'].append(epoch_loss_encoder)
        loss['Rec. Loss'].append(epoch_loss_reconstruction)
        loss['Dis. Loss'].append(epoch_loss_discriminator)

        return loss

    def sample_z(self, num_samples = 25):
        z_real = torch.randn(num_samples, 100)
        z_imag = torch.randn(num_samples, 100)

        return torch.complex(z_real, z_imag).to(next(self.parameters()).device)

    def corrupt(self, x):
        noise_real = self.sigma * torch.rand(x.real.size())
        noise_imag = self.sigma * torch.rand(x.imag.size())

        return torch.complex(x.real + noise_real.to(x.device), x.imag + noise_imag.to(x.device))

    def save(self, epoch, min, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_autoencoder_state_dict': self.optimizer_autoencoder.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'normalize_min': min,
            'normalize_max': max
        }, 'models/idaae_' + str(epoch + 1) + '.pth')

    class AutoEncoder2d(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = layer.ComplexConv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=2)
            self.activation1 = layer.ComplexReLU()
            self.conv2 = layer.ComplexConv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
            self.activation2 = layer.ComplexReLU()
            self.conv3 = layer.ComplexConv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2)
            self.activation3 = layer.ComplexReLU()
            self.conv4 = layer.ComplexConv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
            self.activation4 = layer.ComplexReLU()
            self.linear5 = layer.ComplexLinear(128 * 64 * 64, 100)

            self.linear6 = layer.ComplexLinear(100, 128 * 64 * 64)
            self.activation6 = layer.ComplexReLU()
            self.conv7 = layer.ComplexConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.activation7 = layer.ComplexReLU()
            self.conv8 = layer.ComplexConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.activation8 = layer.ComplexReLU()
            self.conv9 = layer.ComplexConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.activation9 = layer.ComplexReLU()
            self.conv10 = layer.ComplexConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=2, output_padding=1)
            self.activation10 = layer.ComplexSigmoid()

        def encode(self, x):
            x = self.activation1(self.conv1(x))
            x = self.activation2(self.conv2(x))
            x = self.activation3(self.conv3(x))
            x = self.activation4(self.conv4(x))
            x = self.linear5(torch.complex(x.real.view(x.size(0), -1), x.imag.view(x.size(0), -1)))

            return x

        def decode(self, x, original_x):
            x = self.activation6(self.linear6(x))
            x = self.activation7(self.conv7(torch.complex(x.real.view(x.size(0), -1, 64, 64), x.imag.view(x.size(0), -1, 64, 64))))
            x = self.activation8(self.conv8(x))
            x = self.activation8(self.conv9(x))
            x = self.activation10(self.conv10(x))

            mask_phase = x / (torch.abs(x) + 1e-8)
            mask_magnitude = torch.sigmoid(torch.abs(x))
            mask = mask_phase * mask_magnitude

            mask_real = mask.real
            mask_imag = mask.imag

            output_real = mask_real * original_x.real - mask_imag * original_x.imag
            output_imag = mask_real * original_x.imag + mask_imag * original_x.real

            output = torch.complex(output_real, output_imag)

            return output

        def forward(self, x):
            z = self.encode(x)
            z = self.decode(z, x)
            return z

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear1 = layer.ComplexLinear(100, 1000)
            self.activation1 = layer.ComplexReLU()
            self.linear2 = layer.ComplexLinear(1000, 1000)
            self.activation2 = layer.ComplexReLU()
            self.linear3 = layer.ComplexLinear(1000, 1)
            self.activation3 = layer.ComplexSigmoid()

        def discriminate(self, z):
            z = self.activation1(self.linear1(z))
            z = self.activation2(self.linear2(z))
            z = self.activation3(self.linear3(z))

            return z

        def forward(self, z):
            return self.discriminate(z)


class LinearSVM(nn.Module):

    def __init__(self, l2_penalty):
        super().__init__()

        self.l2_penalty = l2_penalty
        self.decision_function = layer.ComplexLinear(100, 1)

    def forward(self, x):
        h = self.decision_function(x)
        return h

    def loss(self, output, y):
        loss = torch.mean(torch.clamp(1 - output * y, min=0))
        loss += self.l2_penalty * torch.mean(self.decision_function.real_linear.weight ** 2)
        loss += self.l2_penalty * torch.mean(self.decision_function.im_linear.weight ** 2)
        return loss

    def binary_class_score(self, output, target, thresh=0.5):
        pred_label = torch.gt(output, thresh)
        class_score_test = torch.eq(pred_label, target.type_as(pred_label))
        return  class_score_test.float().sum()/target.size(0)
