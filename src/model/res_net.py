import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, make_block, layer_sizes):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(make_block, 64, layer_sizes[0], stride=1)
        self.layer1 = self._make_layer(make_block, 128, layer_sizes[1], stride=2)
        self.layer2 = self._make_layer(make_block, 256, layer_sizes[2], stride=2)
        self.layer3 = self._make_layer(make_block, 512, layer_sizes[3], stride=2)

    def _make_layer(self, make_block, channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(channels//2, channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels),
            )
        layers = []
        layers.append(make_block(
            channels//2 if (stride == 2) else channels,
            channels,
            stride, downsample
        ))
        for i in range(1, blocks):
            layers.append(make_block(channels, channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)

        return x