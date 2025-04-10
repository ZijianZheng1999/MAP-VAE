import torch
from torch import nn


class MLP(nn.Module):
    """
    Input shape: (B, N, in_dim)
    Output shape: (B, N, out_dim)
    """
    def __init__(self, in_dim, layer_dims, with_bn=True):
        super(MLP, self).__init__()

        layers = []
        prev_dim = in_dim

        for i, dim in enumerate(layer_dims):
            linear_layer = nn.Linear(prev_dim, dim)
            layers.append(linear_layer)
            if i < len(layer_dims) - 1:
                if with_bn:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.ReLU(inplace=True))
            prev_dim = dim

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, N, in_dim)
        """
        B, N, D = x.shape
        x = x.view(B * N, D)  # (B*N, in_dim)
        x = self.model(x)  # (B*N, out_dim)
        out_dim = x.shape[-1]
        x = x.view(B, N, out_dim)  # (B, N, out_dim)
        return x


class FoldingNetDec(nn.Module):
    def __init__(self, latent_dim, layer_config, num_points, code_word_dim=512, grid_size=1.0, batch_parallel=True):
        super(FoldingNetDec, self).__init__()
        self.latent_dim = latent_dim
        self.code_word_dim = code_word_dim
        self.layer_config = layer_config
        self.num_points = num_points
        self.grid_size = grid_size
        self.batch_parallel = batch_parallel

        # linear mapping from latent space to code word space
        self.input_mapping = nn.Sequential(
            nn.Linear(latent_dim, code_word_dim),
            nn.ReLU(True),
        )

        # reference grid
        side = int(num_points ** 0.5)   # G
        assert side * side == num_points, "num_points is better to be a square number"
        linspace = torch.linspace(-self.grid_size, self.grid_size, side)
        grid_u, grid_v = torch.meshgrid(linspace, linspace, indexing='xy')  # (G, G)
        grid = torch.stack([grid_u, grid_v], dim=-1)  # (G, G, 2)
        grid = grid.view(-1, 2) # (N, 2)
        self.register_buffer('reference_grid', grid)

        # folding layers
        self.mlp_layers = nn.ModuleList()
        prev_output_dim = 2
        for i, hidden_list in enumerate(layer_config):
            in_dim = self.code_word_dim + prev_output_dim
            mlp = MLP(in_dim, hidden_list)
            self.mlp_layers.append(mlp)
            prev_output_dim = hidden_list[-1]
        self.final_feature_dim = prev_output_dim



    def forward(self, x):
        B, Z = x.shape
        # map into code word space
        x = self.input_mapping(x)
        # replicate to [B, N, C]
        x_replicated = x.unsqueeze(1).expand(-1, self.num_points, -1)
        # concatenate with reference grid
        x = torch.cat([x_replicated, self.reference_grid.unsqueeze(0).expand(B, -1, -1)], dim=-1) # [B, N, C+2]

        # folding layers
        for i, mlp in enumerate(self.mlp_layers):
            x = mlp(x)
            if i < len(self.mlp_layers) - 1:
                x = torch.cat([x_replicated, x], dim=-1)

        # (B, N, F) -> (B, F, N)
        x = x.permute(0, 2, 1)
        return x

