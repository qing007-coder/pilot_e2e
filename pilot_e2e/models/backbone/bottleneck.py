import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()

        self.out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels,
            kernel_size=1, stride=1,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=mid_channels,
            kernel_size=3, padding=1, stride=stride,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(
            in_channels=mid_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1
        )
        self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):

        residual = x

        out = self.conv1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3

        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out