import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Downsample if shape mismatch
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, x_channels=3, layers=[128, 256, 512, 1024]):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(x_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_channels)
        self.relu  = nn.ReLU(inplace=True)

        # Build residual blocks
        blocks = []
        for out_channels in layers:
            blocks.append(ResNetCell(self.in_channels, out_channels, stride=2))
            self.in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # flatten spatial dims

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            out = block(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

if __name__ == '__main__':
    x = torch.rand(2, 3, 256, 256)
    resnet = ResNet()    
    out = resnet(x)

