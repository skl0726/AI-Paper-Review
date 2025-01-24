import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class EncoderBlock(nn.Module):
    """After double conv, downscaling with maxpool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.doubleconv = ConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.doubleconv(x)
        x = self.maxpool(skip)

        return skip, x


class DecoderBlock(nn.Module):
    """After upscaling/deconv, double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.doubleconv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.doubleconv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # CenterCrop by the feature map size of the up-convolution
        x = torch.cat((transforms.CenterCrop((x.shape[2], x.shape[3]))(skip), x), dim=1)

        """
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        """

        x = self.doubleconv(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.encoder1 = EncoderBlock(n_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bridge
        factor = 2 if bilinear else 1
        self.bridge = ConvBlock(512, 1024 // factor)

        # Decoder
        self.decoder1 = DecoderBlock(1024, 512 // factor, bilinear)
        self.decoder2 = DecoderBlock(512, 256 // factor, bilinear)
        self.decoder3 = DecoderBlock(256, 128 // factor, bilinear)
        self.decoder4 = DecoderBlock(128, 64, bilinear)

        # outputs
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        x = self.bridge(x)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.output(x)

        return x