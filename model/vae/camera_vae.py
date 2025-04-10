import torch
import torchvision
from torch import nn

from model.vae.resnet_enc import ResnetEnc
from model.vae.resnet_dec import ResnetDec


class CameraVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CameraVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = ResnetEnc(latent_dim)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.decoder = ResnetDec(latent_dim)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        img = self.decoder(z)
        return img, mean, logvar

    @staticmethod
    def loss_function(recon_x, x, mean, logvar):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)
        return BCE + KLD
