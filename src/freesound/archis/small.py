import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cut_channels=0, skip=True, use_custom_init=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cut_channels = cut_channels
        self.skip = skip
        self.use_custom_init = use_custom_init

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if self.use_custom_init:
            self._init_weights()

    def _init_weights(self):
        if self.use_custom_init:
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
    def __init__(self, num_classes=80, in_channels=3, width_multiplier=1, use_custom_init=True):
        super().__init__()
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier
        self.use_custom_init = use_custom_init

        self.conv = nn.Sequential(
            ConvBlock(in_channels=self.in_channels,
                      out_channels=int(64 * width_multiplier),
                      use_custom_init=self.use_custom_init),

            ConvBlock(in_channels=(int(64 * width_multiplier) +
                                   self.in_channels),
                      out_channels=int(128 * width_multiplier),
                      use_custom_init=self.use_custom_init),

            ConvBlock(in_channels=(int(128 * width_multiplier) +
                                   int(64 * width_multiplier) +
                                   self.in_channels),
                      out_channels=int(256 * width_multiplier),
                      use_custom_init=self.use_custom_init),

            ConvBlock(in_channels=(int(256 * width_multiplier) +
                                   int(128 * width_multiplier) +
                                   int(64 * width_multiplier)),
                      out_channels=int(768 * width_multiplier),
                      cut_channels=self.in_channels,
                      skip=False,
                      use_custom_init=self.use_custom_init)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(int(768 * width_multiplier), 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        s1 = x.shape[1]
        if s1 == 1 and s1 != self.in_channels:
            x = x.repeat(1, self.in_channels, 1, 1)
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x
