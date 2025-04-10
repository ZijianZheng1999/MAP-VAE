import torch
from torch import nn


class VRNN(nn.Module):

    def __init__(self, encoder:nn.Module, decoder:nn.Module, prior:nn.Module, rnn_cell_list:[], latent_dim:int, hidden_dim:int):
        super(VRNN, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = encoder                  # [feature_dim + hidden_dim] -> [latent_dim]
        self.decoder = decoder                  # [latent_dim + hidden_dim] -> [feature_dim]
        self.rnn_cell_list = rnn_cell_list      # must be cell, [batch, feature_dim + latent_dim] -> [batch, hidden_dim]
        self.prior = prior
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # init the hidden state
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        x_recon_seq = []
        z_seq = []
        kld_loss = 0.0

        for t in range(seq_len):
            x_t = x[:, t, :]

            # encoder
            encoder_input = torch.cat((x_t, h), dim=1)
            encoder_output = self.encoder(encoder_input)
            z_mu = self.mean(encoder_output)
            z_logvar = self.logvar(encoder_output)

            # sample from the posterior
            z_std = torch.exp(0.5 * z_logvar)
            z_eps = torch.randn_like(z_std)
            z_t = z_mu + z_eps * z_std

            # update prior
            prior_output = self.prior(h)
            prior_mu = self.mean(prior_output)
            prior_logvar = self.logvar(prior_output)

            # get the KL divergence
            kld = 0.5 * torch.sum(
                prior_logvar - z_logvar +
                (torch.exp(z_logvar) + (z_mu - prior_mu) ** 2) / torch.exp(prior_logvar) - 1,
                dim=1  # sum over latent dim
            )  # (batch,)
            kld_loss += kld.mean()

            # reconstruct the input
            dec_input = torch.cat((z_t, h), dim=1)
            x_recon = self.decoder(dec_input)         # (batch, feature_dim)
            x_recon_seq.append(x_recon.unsqueeze(1))  # (batch, 1(seq_dim), feature)

            # update the hidden state
            rnn_input = torch.cat((z_t, x_t), dim=1)
            h = self.rnn_cell(rnn_input, h)           # (batch, hidden_dim)

