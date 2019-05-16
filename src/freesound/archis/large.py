import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cut_channels=0, skip=True, ksize=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cut_channels = cut_channels
        self.skip = skip

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, 1, ksize // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, ksize, 1, ksize // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, ipt):
        if self.cut_channels:
            ipt = ipt[:, :-self.cut_channels]
        x = self.conv1(ipt)
        x = self.conv2(x)
        if self.skip:
            x = torch.cat([x, ipt], 1)
        x = F.avg_pool2d(x, 2)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, ksize=7),
            ConvBlock(in_channels=64 + 3, out_channels=128),
            ConvBlock(in_channels=128 + 64 + 3, out_channels=256),
            ConvBlock(in_channels=256 + 128 + 64 + 3, out_channels=384),
            ConvBlock(in_channels=384 + 256 + 128 + 64 + 3, out_channels=384),
            ConvBlock(in_channels=384 + 384 + 256 + 128 + 64 + 3, out_channels=512),
        )

        self.last_conv = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(1731, 1024, 2, stride=2, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.last_conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x
