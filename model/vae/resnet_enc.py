import torch
import torchvision
from torch import nn
from enum import Enum


class EncoderStr(Enum):
    ResNet18 = 18
    ResNet34 = 34
    ResNet50 = 50
    ResNet101 = 101
    ResNet152 = 152


class ResnetEnc(nn.Module):
    def __init__(self, latent_dim, mod_str: EncoderStr = EncoderStr.ResNet152):
        super(ResnetEnc, self).__init__()
        self.latent_dim = latent_dim
        if not isinstance(mod_str, EncoderStr):
            raise ValueError('depth should be an instance of EncoderDepth Enum to follow ResNet\'s design')
        if mod_str == EncoderStr.ResNet18:
            self.resnet = torchvision.models.resnet18()
            output_feature = 512
        elif mod_str == EncoderStr.ResNet34:
            self.resnet = torchvision.models.resnet34()
            output_feature = 512
        elif mod_str == EncoderStr.ResNet50:
            self.resnet = torchvision.models.resnet50()
            output_feature = 2048
        elif mod_str == EncoderStr.ResNet101:
            self.resnet = torchvision.models.resnet101()
            output_feature = 2048
        elif mod_str == EncoderStr.ResNet152:
            self.resnet = torchvision.models.resnet152()
            output_feature = 2048
        self.resnet.fc = nn.Identity()
        # output feature: 512 for ResNet18, ResNet34; 2048 for ResNet50, ResNet101, ResNet152
        self.fc_mean = nn.Linear(output_feature, latent_dim)
        self.fc_logvar = nn.Linear(output_feature, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        resnet_output = self.resnet(x)
        mean = self.fc_mean(resnet_output)
        logvar = self.fc_logvar(resnet_output)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
