import torch
import torch.nn as nn


class DCUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=16)
        self.activation1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=16)
        self.activation2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=32)
        self.activation3 = nn.LeakyReLU()
        self.down3 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=32)
        self.activation4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(num_features=64)
        self.activation5 = nn.LeakyReLU()
        self.down5 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(num_features=64)
        self.activation6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch7 = nn.BatchNorm2d(num_features=128)
        self.activation7 = nn.LeakyReLU()
        self.down7 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batch8 = nn.BatchNorm2d(num_features=128)
        self.activation8 = nn.LeakyReLU()

        self.conv9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch9 = nn.BatchNorm2d(256)
        self.activation9 = nn.LeakyReLU()
        self.drop9 = nn.Dropout(0.5)
        self.down9 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch10 = nn.BatchNorm2d(256)
        self.activation10 = nn.LeakyReLU()

        self.conv11 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch11 = nn.BatchNorm2d(256)
        self.activation11 = nn.LeakyReLU()
        self.drop11 = nn.Dropout(0.5)

        self.up12 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv12 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.batch12 = nn.BatchNorm2d(128)
        self.activation12 = nn.LeakyReLU()

        self.conv13 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batch13 = nn.BatchNorm2d(128)
        self.activation13 = nn.LeakyReLU()

        self.conv14 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.batch14 = nn.BatchNorm2d(64)
        self.activation14 = nn.LeakyReLU()

        self.up15 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv15 = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.batch15 = nn.BatchNorm2d(64)
        self.activation15 = nn.LeakyReLU()

        self.conv16 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batch16 = nn.BatchNorm2d(64)
        self.activation16 = nn.LeakyReLU()

        self.conv17 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.batch17 = nn.BatchNorm2d(32)
        self.activation17 = nn.LeakyReLU()

        self.up18 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv18 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)
        self.batch18 = nn.BatchNorm2d(32)
        self.activation18 = nn.LeakyReLU()

        self.conv19 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batch19 = nn.BatchNorm2d(32)
        self.activation19 = nn.LeakyReLU()

        self.conv20 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.batch20 = nn.BatchNorm2d(16)
        self.activation20 = nn.LeakyReLU()

        self.up21 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv21 = nn.ConvTranspose2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
        self.batch21 = nn.BatchNorm2d(16)
        self.activation21 = nn.LeakyReLU()

        self.conv22 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batch22 = nn.BatchNorm2d(16)
        self.activation22 = nn.LeakyReLU()

        self.conv23 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.batch23 = nn.BatchNorm2d(3)
        self.activation23 = nn.LeakyReLU()

        self.conv24 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.batch24 = nn.BatchNorm2d(3)
        self.activation24 = nn.Tanh()

    def forward(self, x):
        x = self.activation1(self.batch1(self.conv1(x)))
        x = self.activation2(self.batch2(self.conv2(x)))
        skip_connection4 = self.activation3(self.batch3(self.conv3(x)))
        x = self.down3(skip_connection4)
        x = self.activation4(self.batch4(self.conv4(x)))
        skip_connection3 = self.activation5(self.batch5(self.conv5(x)))
        x = self.down5(skip_connection3)
        x = self.activation6(self.batch6(self.conv6(x)))
        skip_connection2 = self.activation7(self.batch7(self.conv7(x)))
        x = self.down7(skip_connection2)
        x = self.activation8(self.batch8(self.conv8(x)))
        x = self.activation9(self.batch9(self.conv9(x)))
        skip_connection1 = self.drop9(x)
        x = self.down9(skip_connection1)
        x = self.activation10(self.batch10(self.conv10(x)))
        x = self.activation11(self.batch11(self.conv11(x)))
        x = self.drop11(x)
        x = self.up12(x)
        x = self.activation12(self.batch12(self.conv12(torch.cat((skip_connection1, x), dim=1))))
        x = self.activation13(self.batch13(self.conv13(x)))
        x = self.activation14(self.batch14(self.conv14(x)))
        x = self.up15(x)
        x = self.activation15(self.batch15(self.conv15(torch.cat((skip_connection2, x), dim=1))))
        x = self.activation16(self.batch16(self.conv16(x)))
        x = self.activation17(self.batch17(self.conv17(x)))
        x = self.up18(x)
        x = self.activation18(self.batch18(self.conv18(torch.cat((skip_connection3, x), dim=1))))
        x = self.activation19(self.batch19(self.conv19(x)))
        x = self.activation20(self.batch20(self.conv20(x)))
        x = self.up21(x)
        x = self.activation21(self.batch21(self.conv21(torch.cat((skip_connection4, x), dim=1))))
        x = self.activation22(self.batch22(self.conv22(x)))
        x = self.activation23(self.batch23(self.conv23(x)))
        x = self.activation24(self.batch24(self.conv24(x)))

        return x
