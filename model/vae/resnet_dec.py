import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class BottleneckDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, upscale=False):
        super().__init__()

        # PixelShuffle up-sample
        if upscale:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.upsample = None

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if self.upsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        if self.upsample:
            x = self.upsample(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)

        return x


class ResnetDec(nn.Module):
    def __init__(self, latent_dim, output_size=(3, 375, 1224)):
        super().__init__()
        self.output_channels, self.output_height, self.output_width = output_size
        self.latent_dim = latent_dim

        self.init_planes = 2048  # 512 for ResNet18, ResNet34; 2048 for ResNet50, ResNet101, ResNet152
        self.feature_size = (self.output_height // 32, self.output_width // 32)  # H/32, W/32

        self.fc = nn.Linear(latent_dim, self.init_planes * self.feature_size[0] * self.feature_size[1])

        # self.pre_up_scaler2 = self._make_layer(2048, 2048, num_blocks=1, upscale=True)
        # self.pre_up_scaler1 = self._make_layer(2048, 2048, num_blocks=1, upscale=True)

        self.layer4 = self._make_layer(2048, 1024, num_blocks=3, upscale=True)  # Layer4
        self.layer3 = self._make_layer(1024, 512, num_blocks=36, upscale=True)  # Layer3
        self.layer2 = self._make_layer(512, 256, num_blocks=8, upscale=True)    # Layer2
        self.layer1 = self._make_layer(256, 64, num_blocks=3, upscale=True)     # Layer1

        self.conv_final = nn.Conv2d(64, self.output_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, num_blocks, upscale):
        layers = [BottleneckDecoder(in_channels, out_channels // 2, out_channels, upscale=upscale)]
        for _ in range(1, num_blocks):
            layers.append(BottleneckDecoder(out_channels, out_channels // 2, out_channels))
        return nn.Sequential(*layers)

    def forward(self, z):

        x = self.fc(z)  # (B, 2048) → (B, 2048 * H/32 * W/32)
        x = F.relu(x, inplace=True)
        
        # reshape from (B, 2048 * H/32 * W/32) to (B, 2048, H/32, W/32)
        x = x.view(-1, self.init_planes, self.feature_size[0], self.feature_size[1])  # reshape

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.conv_final(x)
        x = self.sigmoid(x)

        # interpolate to the target size
        x = F.interpolate(x, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)

        return x
