import torch
import torch.nn as nn
from .bottleneck import Bottleneck


class ResNet(nn.Module):
    expansion = 4

    def __init__(self, layers, channels):
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=self.in_channels,
                kernel_size=7, padding=3, stride=2,
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(mid_channels=channels[0], blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(mid_channels=channels[1], blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(mid_channels=channels[2], blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(mid_channels=channels[3], blocks=layers[3], stride=2)

    def _make_layer(self, mid_channels, blocks, stride):
        out_channels = mid_channels * self.expansion

        layers = []
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels, out_channels=out_channels,
                    kernel_size=1, stride=1, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(Bottleneck(
            in_channels=self.in_channels,
            mid_channels=mid_channels,
            stride=stride,
            downsample=downsample,
        ))

        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                stride=1,
            ))

        return nn.Sequential(*layers)    

    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] 输入图像

        Returns:
            c2: [B, 256,  H/4,  W/4]
            c3: [B, 512,  H/8,  W/8]
            c4: [B, 1024, H/16, W/16]
            c5: [B, 2048, H/32, W/32]
        """
        feature = self.stem(images)          # [B, 64, H/4, W/4]

        c2 = self.layer1(feature)           # [B, 256,  H/4,  W/4]
        c3 = self.layer2(c2)                # [B, 512,  H/8,  W/8]
        c4 = self.layer3(c3)                # [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)                # [B, 2048, H/32, W/32]

        return c2, c3, c4, c5

        