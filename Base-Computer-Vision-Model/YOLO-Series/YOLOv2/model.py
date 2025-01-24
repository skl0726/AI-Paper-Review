import torch
import torch.nn as nn
#from torchvision.models import vgg16_bn

"""
tuple: (filter size, kernel size, stride, padding)
"M": maxpooling with stride 2x2 and kernel 2x2
"""

middle_architecture_config = [
    (32, 3, 1, 1),
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    (64, 1, 1, 0),
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
]

extra_architecture_config = [
    "M",
    (1024, 3, 1, 1),
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

final_architecture_config = [
    (1024, 3, 3, 1),
    (125, 1, 1, 0), # anchor 5, class 20 (5 * (5 + 20))
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class YOLOv2_VGG_16(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        #self.num_anchors = 5
        #self.num_classes = num_classes

        self.darknet_middle = self._make_layers(middle_architecture_config, in_channels)
        #self.darkent_middle = nn.Sequential(*list(vgg.features.children())[:-1])
        self.darknet_extra = self._make_layers(extra_architecture_config, 512)

        self.skip_module = CNNBlock(512, 64, kernel_size=1, stride=1, padding=0)

        self.final = self._make_layers(final_architecture_config, 1280)
        """
        self.final = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), kernel_size=1, stride=1, padding=0)
        )
        """

        self._init_conv2d()

    def _make_layers(self, architecture, in_channels):
        layers = []
        in_channels = in_channels

        for param in architecture:
            if type(param) == tuple:
                layers.append(CNNBlock(in_channels, param[0], kernel_size=param[1], stride=param[2], padding=param[3]))
                in_channels = param[0]
            elif type(param) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def _init_conv2d(self):
        for c in self.darknet_middle.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.darknet_extra.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.skip_module.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

        for c in self.final.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, mean=0.0, std=0.01)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        output_size = x.size(-1)
        output_size /= 32
        output_size = int(output_size)

        x = self.darknet_middle(x)
        skip_x = self.skip_module(x) # torch.Size([B, 512, 26, 26]) -> torch.Size([B, 64, 26, 26])

        # YOLOv2 reorg layer
        skip_x = skip_x.view(-1, 64, output_size, 2, output_size, 2).contigous()    # torch.Size([B, 64, 13, 2, 13, 2])
        skip_x = skip_x.permute(0, 3, 5, 1, 2, 4).contigous()                       # torch.Size([B, 64, 2, 2, 13, 13])
        skip_x = skip_x.view(-1, 256, output_size, output_size)                     # 256 x 13 x 13 torch.Size([B, 256, 13, 13])

        x = self.darknet_extra(x)           # torch.Size([B, 1024, 13, 13])
        x = torch.cat([x, skip_x], dim=1)   # torch.Size([B, 1280, 13, 13])
        x = self.final(x)                   # torch.Size([B, 125, 13, 13])

        return x
