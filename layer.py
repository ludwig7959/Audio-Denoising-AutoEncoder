import torch
from torch import nn


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.real_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.im_linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        output_real = self.real_linear(x.real)
        output_imag = self.im_linear(x.imag)

        return torch.complex(output_real, output_imag)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)

        self.im_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x.real
        x_im = x.imag

        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.complex(c_real, c_im)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x.real
        x_im = x.imag

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.complex(ct_real, ct_im)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x_real = x.real
        x_im = x.imag

        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)

        output = torch.complex(n_real, n_im)
        return output


class ComplexSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return torch.complex(self.sigmoid(x.real), self.sigmoid(x.imag))


class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

    def forward(self, x):

        return torch.complex(self.relu(x.real), self.relu(x.imag))


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):

        return torch.complex(self.leaky_relu(x.real), self.leaky_relu(x.imag))


class ComplexTanh(nn.Module):
    def __init__(self):
        super().__init__()

        self.tanh = nn.Tanh()

    def forward(self, x):

        return torch.complex(self.tanh(x.real), self.tanh(x.imag))
