import torch
import torch.nn as nn


"""
tuple: (kernel size, filter size, stride, padding)
"M": maxpooling with stride 2x2 and kernel 2x2
list: structured by tuples and lastly int with number of repeats
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for param in architecture:
            if type(param) == tuple:
                layers.append(CNNBlock(in_channels, param[1], kernel_size=param[0], stride=param[2], padding=param[3]))
                in_channels = param[1]
            elif type(param) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(param) == list:
                param1 = param[0]
                param2 = param[1]
                num_repeat = param[2]
                for _ in range(num_repeat):
                    layers.append(in_channels, param1[1], kernel_size=param1[0], stride=param1[2], padding=param1[3])
                    layers.append(param1[1], param2[1], kernel_size=param2[0], stride=param2[2], padding=param2[3])
                    in_channels = param2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(5*B+C)),
        )
