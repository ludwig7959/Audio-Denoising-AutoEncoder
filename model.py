import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import binary_cross_entropy

import layer
from function import complex_mse_loss


class DCUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.optimizer = optim.Adam(self.parameters())

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
            epoch_loss += loss
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
    def __init__(self, sigma=0.1, M=20):
        super().__init__()

        self.sigma = sigma
        self.M = M

        self.optimizer = optim.Adam(self.parameters())
        self.optimizer_discriminator = optim.Adam(self.parameters())

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

        self.discriminator = Discriminator(prior=self.norm_prior)
        self.linear_svm = LinearSVM(0.5)

    def norm_prior(self, noSamples=25):
        z = torch.rand(noSamples, 100)
        return z

    def encode(self, x):
        x1 = self.activation1(self.conv1(x))
        x2 = self.activation2(self.conv2(x1))
        x3 = self.activation3(self.conv3(x2))
        x4 = self.activation4(self.conv4(x3))
        x5 = self.linear5(torch.complex(x4.real.view(x4.size(0), -1), x4.imag.view(x4.size(0), -1)))

        return x5

    def corrupt(self, x):
        noise_real = self.sigma * torch.rand(x.real.size())
        noise_imag = self.sigma * torch.rand(x.imag.size())

        return torch.complex(x.real + noise_real, x.imag + noise_imag)

    def sample_z(self, noSamples=25):
        z_real = self.norm_prior(noSamples=noSamples)
        z_imag = self.norm_prior(noSamples=noSamples)

        return torch.complex(z_real, z_imag)

    def decode(self, x):
        x6 = self.activation6(self.linear6(x))
        x7 = self.activation7(self.conv7(torch.complex(x6.real.view(x6.size(0), -1, 64, 64), x6.imag.view(x6.size(0), -1, 64, 64))))
        x8 = self.activation8(self.conv8(x7))
        x9 = self.activation8(self.conv9(x8))
        x10 = self.activation10(self.conv10(x9))

        mask_phase = x10 / (torch.abs(x10) + 1e-8)
        mask_magnitude = torch.sigmoid(torch.abs(x10))
        mask = mask_phase * mask_magnitude

        mask_real = mask.real
        mask_imag = mask.imag

        output_real = mask_real * x.real - mask_imag * x.imag
        output_real = torch.squeeze(output_real, 1)
        output_imag = mask_real * x.imag + mask_imag * x.real
        output_imag = torch.squeeze(output_imag, 1)

        output = torch.complex(output_real, output_imag)

        return output

    def forward(self, x):
        x_corr = torch.zeros(x.size(), dtype=torch.complex64)
        for m in range(self.M):
            x_corr += self.corrupt(x)
        x_corr /= self.M
        z = self.encode(x_corr)
        return z, self.decode(z)

    def rec_loss(self, rec_x, x, loss='BCE'):
        if loss == 'BCE':
            return torch.mean(binary_cross_entropy(rec_x.real, x.real, size_average=True) + binary_cross_entropy(rec_x.imag, x.imag, size_average=True))
        elif loss == 'MSE':
            return torch.mean(complex_mse_loss(rec_x, x))
        else:
            print('unknown loss: '+loss)

    def train_epoch(self, batches):
        loss = {'Encoder Loss': [], 'Rec. Loss': [], 'Dis. Loss': []}
        epoch_loss_encoder = 0.0
        epoch_loss_reconstruction = 0.0
        epoch_loss_discriminator = 0.0
        for features_batch, labels_batch in batches:
            self.train()
            self.discriminator.train()

            input = features_batch
            label = labels_batch

            z_fake, x_reconstructed = self.forward(input)

            reconstruction_loss = self.rec_loss(x_reconstructed, input, loss='MSE')  #loss='BCE' or 'MSE'
            encoder_loss = self.discriminator.gen_loss(z_fake)
            discriminator_loss = self.discriminator.dis_loss(z_fake)

            autoencoder_loss = reconstruction_loss + 1.0 * encoder_loss

            self.optimizer_discriminator.zero_grad()
            discriminator_loss.backward()
            self.optimizer_discriminator.step()

            self.optimizer.zero_grad()
            autoencoder_loss.backward()
            self.optimizer.step()

            epoch_loss_encoder += encoder_loss
            epoch_loss_reconstruction += reconstruction_loss
            epoch_loss_discriminator += discriminator_loss

        loss['Encoder Loss'].append(epoch_loss_encoder)
        loss['Rec. Loss'].append(epoch_loss_reconstruction)
        loss['Dis. Loss'].append(epoch_loss_discriminator)

        return loss

    def save(self, epoch, min, max):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'normalize_min': min,
            'normalize_max': max
        }, 'models/idaae_' + str(epoch + 1) + '.pth')


class Discriminator(nn.Module):
    def __init__(self, prior):
        super().__init__()

        self.prior = prior

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

    def dis_loss(self, z):
        z_real = torch.complex(self.prior(z.size(0)), self.prior(z.size(0)))
        p_real = self.discriminate(z_real)

        z_fake = z.detach()
        p_fake = self.discriminate(z_fake)

        ones = torch.ones(p_real.size())
        zeros = torch.zeros(p_real.size())

        return 0.5 * torch.mean(binary_cross_entropy(p_real.real, ones) + binary_cross_entropy(p_real.imag, ones) + binary_cross_entropy(p_fake.real, zeros) + binary_cross_entropy(p_fake.imag, zeros))

    def gen_loss(self, z):
        p_fake = self.discriminate(z)
        ones = torch.ones(p_fake.size())
        return torch.mean(binary_cross_entropy(p_fake.real, ones) + binary_cross_entropy(p_fake.imag, ones))


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
