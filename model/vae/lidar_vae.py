import torch
from torch import nn

from model.vae.pointnet_plus_enc import PointNetPlusEnc
from model.vae.foldingnet_dec import FoldingNetDec

from pytorch3d.loss import chamfer_distance

class LidarVAE(nn.Module):
    def __init__(self, latent_dim, point_dim, num_of_sample=115600, batch_parallel=False):
        super(LidarVAE, self).__init__()
        self.num_of_sample = num_of_sample
        self.point_dim = point_dim
        self.latent_dim = latent_dim
        self.batch_parallel = batch_parallel

        # param layers_config: (S, Q, mlp_layers) * layer_num
        #                           - S: sampled num
        #                           - Q: queried num
        #                           - R: query radius
        #                           - mlp_layers: list of node number in each layer
        self.encoder_layer_cfg = [
            (4096, 64, 1.0, [64, 128, 128]),
            (1024, 64, 2.5, [128, 128, 256]),
            (256, 32, 5.0, [256, 256, 512]),
            (64, 16, 10.0, [512, 512, 1024])
        ]
        self.encoder_layer_cfg_large = [
            (4096, 128, 1.0, [64, 128, 128, 256]),
            (2048, 128, 2.5, [256, 256, 256, 512]),
            (1024, 64, 5.0, [512, 512, 512, 1024]),
            (256, 64, 10.0, [1024, 1024, 1024, 2048]),
            (64, 32, 20.0, [2048, 2048, 2048, 4096])
        ]

        self.decoder_layer_cfg = [
            ([1024, 1024, 512]),
            ([1024, 512, 256]),
            ([512, 256, 4]),
        ]
        self.decoder_layer_cfg_large = [
            ([4096, 4096, 2048]),
            ([2048, 2048, 1024]),
            ([1024, 1024, 512]),
            ([512, 512, 256]),
            ([256, 256, 4])
        ]

        self.encoder = PointNetPlusEnc(latent_dim, point_dim, self.encoder_layer_cfg, batch_parallel)
        self.decoder = FoldingNetDec(latent_dim, self.decoder_layer_cfg, self.num_of_sample)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        poins = self.decoder(z)
        return poins, mean, logvar

    def loss_function(self, recon_points, points, mean, logvar):

        # the shape of points and recon_points is [batch, num_points, 3]
        recon_points_t = recon_points[:, :3, :].permute(0, 2, 1)    # [batch, 3, num_points] -> [batch, num_points, 3]
        points_t = points[:, :3, :].permute(0, 2, 1)
        # use libs, required shape: [batch, num_points, 3]
        reconstructed_loss, _ = chamfer_distance(recon_points_t, points_t)

        KLD = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp()) / points.size(0)

        return reconstructed_loss + KLD

