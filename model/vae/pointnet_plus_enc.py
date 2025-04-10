import torch
from torch import nn


class PointNetPlusEnc(nn.Module):

    def __init__(self, latent_dim, feature_dim, layer_cfg, batch_parallel=False):
        super(PointNetPlusEnc, self).__init__()
        self.latent_dim = latent_dim
        self.layer_cfg = layer_cfg
        self.feature_dim = feature_dim
        self.batch_parallel = batch_parallel

        # mlp used as pointnet, which is actually conv2d
        self.mlps = nn.ModuleList()
        # fps and ball query
        self.fps_ops = []
        self.ball_query_ops = []

        input_dim = feature_dim
        for fps_samples, ball_query_num, query_radius, mlp_layers in layer_cfg:
            self.fps_ops.append(self.farthest_point_sampling)
            self.ball_query_ops.append(self.ball_query)
            mlp_layers = [input_dim] + mlp_layers
            mlp = []
            for i in range(len(mlp_layers) - 1):
                mlp.append(nn.Conv2d(mlp_layers[i], mlp_layers[i + 1], 1))
                mlp.append(nn.BatchNorm2d(mlp_layers[i + 1]))
                mlp.append(nn.ReLU(True))
            self.mlps.append(nn.Sequential(*mlp))
            input_dim = mlp_layers[-1]

        self.global_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)


    '''
    if batch_parallel disabled, batch: [tensor(F, N1), tensor(F, N2), ...]
    if batch_parallel enabled, batch: tensor(batch_size, F, N)
    '''
    def farthest_point_sampling(self, batch, num_samples):

        if self.batch_parallel:
            B, F, N = batch.shape
            S = num_samples
            assert S <= N, "num_samples should be less than num_points"

            batch_coord = batch[:, :3, :]

            sampled_idx = torch.zeros(B, S, dtype=torch.long, device=batch.device)
            distance = torch.ones(B, N, device=batch.device) * float('inf')

            farthest_idx = torch.randint(0, N, (B,), device=batch.device)

            for i in range(S):
                sampled_idx[:, i] = farthest_idx
                farthest_point = batch_coord.gather(dim=2, index=farthest_idx.view(B, 1, 1).expand(B, 3, 1))  # [B, 3, 1]
                dist = torch.cdist(batch_coord.transpose(1, 2), farthest_point.transpose(1, 2), p=2).squeeze(-1)  # [B, N]
                distance = torch.minimum(distance, dist)
                farthest_idx = torch.argmax(distance, dim=1)

            sampled_results = batch.gather(dim=2, index=sampled_idx.unsqueeze(1).expand(-1, F, -1))  # [B, F, S]

        else:
            sampled_results = None

        return sampled_results

    def ball_query(self, batch, radius, query_num, sampled_center):
        """
        :param batch: [B, F, N]
                      B=batch, F=feature(3 or 4), N=num_points
        :param radius:
        :param query_num:
        :param sampled_center:  [B, F, S], S=sample_num
        :return: query_results, [B, F, S, Q], Q=query_num
        """

        if self.batch_parallel:
            B, F, N = batch.shape               # [B, F, N]
            _, _, S = sampled_center.shape      # [B, F, S]

            dist = torch.cdist(
                batch[:, :3, :].permute(0, 2, 1),  # => [B, N, 3]
                sampled_center[:, :3, :].permute(0, 2, 1),  # => [B, S, 3]
                p=2
            )  # dist => [B, N, S]

            dist = dist.permute(0, 2, 1)  # => [B, S, N]

            idx = torch.topk(dist, k=query_num, dim=2, largest=False, sorted=False).indices  # [B, S, Q]
            mask = (dist < radius)

            has_valid = mask.any(dim=2, keepdim=True)  # [B, S, 1]
            first_valid_idx = torch.argmax(mask.to(torch.int), dim=2)  # [B, S]
            first_valid_idx = torch.where(
                has_valid.squeeze(-1),
                first_valid_idx,
                torch.zeros_like(first_valid_idx)
            )  # [B, S]
            first_valid_idx = first_valid_idx.unsqueeze(-1).expand(-1, -1, query_num)  # [B, S, Q]

            masked_idx = mask.gather(dim=2, index=idx)  # [B, S, Q] (True/False)

            idx = torch.where(masked_idx, idx, first_valid_idx)  # [B, S, Q]

            batch_expanded = batch.unsqueeze(2).expand(-1, -1, S, -1)  # => [B, F, S, N]
            idx_for_gather = idx.unsqueeze(1).expand(-1, F, -1, -1)  # => [B, F, S, Q]
            query_results = batch_expanded.gather(dim=3, index=idx_for_gather)


            query_results[:, :3, :, :] -= sampled_center[:, :3, :].unsqueeze(-1)  # => [B, 3, S, 1] 广播到 [B, 3, S, Q]

            return query_results

        else:
            return None

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        for i, (fps, ball_query, mlp) in enumerate(zip(self.fps_ops, self.ball_query_ops, self.mlps)):
            S, Q, R, _ = self.layer_cfg[i]
            sampled_points = fps(x, S)                          # [B, S, F]
            grouped_points = ball_query(x, R, Q, sampled_points)# [B, F, S, Q]
            grouped_feature = mlp(grouped_points)               # [B, C, S, Q]
            x = torch.max(grouped_feature, dim=-1)[0]           # [B, F', S]

        x = x.mean(dim=-1)
        x = self.global_mlp(x)

        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)

        return z, mu, log_var
