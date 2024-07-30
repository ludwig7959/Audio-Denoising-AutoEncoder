import torch
import torch.nn as nn
import torchsummary


class DCUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=16)
        self.activation1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.activation2 = nn.LeakyReLU()
        self.down2 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=64)
        self.activation3 = nn.LeakyReLU()
        self.down3 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=128)
        self.activation4 = nn.LeakyReLU()
        self.down4 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(256)
        self.activation5 = nn.LeakyReLU()
        self.drop5 = nn.Dropout(0.5)
        self.down5 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv6 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(256)
        self.activation6 = nn.LeakyReLU()
        self.drop6 = nn.Dropout(0.5)

        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.batch7 = nn.BatchNorm2d(128)
        self.activation7 = nn.LeakyReLU()

        self.conv8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.batch8 = nn.BatchNorm2d(64)
        self.activation8 = nn.LeakyReLU()

        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv9 = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.batch9 = nn.BatchNorm2d(64)
        self.activation9 = nn.LeakyReLU()

        self.conv10 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.batch10 = nn.BatchNorm2d(32)
        self.activation10 = nn.LeakyReLU()

        self.up11 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv11 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)
        self.batch11 = nn.BatchNorm2d(32)
        self.activation11 = nn.LeakyReLU()

        self.conv12 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.batch12 = nn.BatchNorm2d(16)
        self.activation12 = nn.LeakyReLU()

        self.up13 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv13 = nn.ConvTranspose2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
        self.batch13 = nn.BatchNorm2d(16)
        self.activation13 = nn.LeakyReLU()

        self.conv14 = nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)
        self.batch14 = nn.BatchNorm2d(2)
        self.activation14 = nn.LeakyReLU()

        self.conv15 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.batch15 = nn.BatchNorm2d(1)
        self.activation15 = nn.Tanh()

    def forward(self, x):
        x1 = self.activation1(self.batch1(self.conv1(x)))
        skip_connection4 = self.activation2(self.batch2(self.conv2(x1)))
        x2 = self.down2(skip_connection4)
        skip_connection3 = self.activation3(self.batch3(self.conv3(x2)))
        x3 = self.down3(skip_connection3)
        skip_connection2 = self.activation4(self.batch4(self.conv4(x3)))
        x4 = self.down4(skip_connection2)
        skip_connection1 = self.activation5(self.batch5(self.conv5(x4)))
        x5 = self.drop5(self.down5(skip_connection1))
        x6 = self.drop6(self.activation6(self.batch6(self.conv6(x5))))

        x7 = self.up7(x6)
        x7 = torch.cat((skip_connection1, x7), dim=1)
        x7 = self.activation7(self.batch7(self.conv7(x7)))

        x8 = self.activation8(self.batch8(self.conv8(x7)))

        x9 = self.up9(x8)
        x9 = torch.cat((skip_connection2, x9), dim=1)
        x9 = self.activation9(self.batch9(self.conv9(x9)))

        x10 = self.activation10(self.batch10(self.conv10(x9)))

        x11 = self.up11(x10)
        x11 = torch.cat((skip_connection3, x11), dim=1)
        x11 = self.activation11(self.batch11(self.conv11(x11)))

        x12 = self.activation12(self.batch12(self.conv12(x11)))

        x13 = self.up13(x12)
        x13 = torch.cat((skip_connection4, x13), dim=1)
        x13 = self.activation13(self.batch13(self.conv13(x13)))

        x14 = self.activation14(self.batch14(self.conv14(x13)))

        x15 = self.activation15(self.batch15(self.conv15(x14)))

        return x15
