from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# =========================================================================================
# Enc-Dec model utility functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
# =========================================================================================

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernelsize, name, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            OrderedDict([(
                name + "conv", nn.Conv2d(in_channels, mid_channels, kernel_size=kernelsize, padding=1)),
                (name + "bn", nn.BatchNorm2d(mid_channels)),
                (name + "relu", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernelsize, name, mid_channels=None):
        super().__init__()
        pad = (kernelsize - 1) // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            OrderedDict([(
                name + "conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=kernelsize, padding=pad)),
                (name + "bn1", nn.BatchNorm2d(mid_channels)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "conv2", nn.Conv2d(mid_channels, out_channels, kernel_size=kernelsize, padding=pad)),
                (name + "bn2", nn.BatchNorm2d(out_channels)),
                (name + "relu2", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, name):
        super().__init__()
        pad = (kernel_size - 1) // 2
        mid_channels = out_channels
        self.maxpool_conv = nn.Sequential(
            OrderedDict([
                (name + "maxpool", nn.MaxPool2d(2)),
                (name + "conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad)),
                (name + "bn1", nn.BatchNorm2d(mid_channels)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "conv2", nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad)),
                (name + "bn2", nn.BatchNorm2d(out_channels)),
                (name + "relu2", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, name, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, name)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, 3, name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpSR(nn.Module):
    """Upscaling with Super Resolution """

    def __init__(self, in_channels, out_channels, kernel_size, name):
        super().__init__()
        self.upsample_conv = nn.Sequential(
            OrderedDict([
                (name + "conv", nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False)),
                (name + "bn", nn.BatchNorm2d(out_channels)),
                (name + "pixsh", nn.PixelShuffle(upscale_factor=2)),
                (name + "relu", nn.PReLU()), ])
        )

    def forward(self, x):
        return self.upsample_conv(x)

class UpMetaSR(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, hout, wout, kernel_size, name, bilinear=True):
        super().__init__()
        self.out_h = hout
        self.out_w = wout
        pad = (kernel_size - 1) // 2
        self.conv0 = nn.Sequential(
            OrderedDict([
                (name + "conv0", nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=pad)), ])
        )
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, name)
        else:
            self.conv = DoubleConv(in_channels, out_channels, 3, name)

    def forward(self, x1, x2):
        print(x1.size())
        print(x2.size())
        x1 = self.conv0(x1)
        x1 = F.interpolate(x1, [self.out_h, self.out_w], mode='bilinear')
        x2 = F.interpolate(x2, [self.out_h, self.out_w], mode='bilinear')
        print(x1.size())
        print(x2.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        print(diffX)
        print(diffX)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """convolution"""

    def __init__(self, in_channels, out_channels, name):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            OrderedDict([(
                name + "conv", nn.Conv2d(in_channels, out_channels, kernel_size=1)), ])
        )
        self.conv1 = nn.Sequential(
            OrderedDict([(
                name + "conv1", nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)), ])
        )

    def forward(self, x):
        try:
            return self.conv(x)
        except:
            return self.conv1(x)


class FullConn(nn.Module):
    """convolution"""

    def __init__(self, in_channels, out_channels, name):
        super(FullConn, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                (name + "fclayer", nn.Linear(in_channels, out_channels)),
                (name + "relu", nn.ReLU(inplace=True)), ])
        )

    def forward(self, x):
        return self.fc(x)


class FullConnOnly(nn.Module):
    """convolution"""

    def __init__(self, in_channels, out_channels, name):
        super(FullConnOnly, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                (name + "fclayer", nn.Linear(in_channels, out_channels)), ])
        )

    def forward(self, x):
        return self.fc(x)


