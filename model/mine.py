import torch
from torch import nn
import math


class MINE(nn.Module):
    def __init__(self, total_dim):
        super(MINE, self).__init__()
        self.distribution_dim = total_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(total_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, dis_1, dis_2):
        x = torch.cat([dis_1, dis_2], dim=1)
        out = self.model(x)
        return out


# input joint_data: (batch_dim, dim_of_distribution)
def get_mi(mine, max_epoch, dis_1, dis_2, need_print=False):
    mi_lb = None
    optimizer = torch.optim.Adam(mine.parameters())
    shuffled_dis_1 = dis_1[torch.randperm(dis_1.shape[0])]
    for epoch in range(max_epoch):
        mine.train()
        optimizer.zero_grad()
        # joint distribution output
        joint_pred = mine(dis_1, dis_2)
        # marginal distribution output
        marginal_pred = mine(shuffled_dis_1, dis_2)
        # derive lower bound of mutual information (-1)
        loss = -(joint_pred.mean(1) - torch.logsumexp(marginal_pred, 1)).sum()
        if need_print:
            print(f'\tMINE, Iteration: {epoch}, loss: {loss.item()}')
        # back propagation
        loss.backward(retain_graph=True)
        optimizer.step()
    # derive the lower bound of mutual information
    mine.eval()
    joint_pred = mine(dis_1, dis_2)
    marginal_pred = mine(shuffled_dis_1, dis_2)
    mi_lb = (joint_pred.mean(1) - torch.logsumexp(marginal_pred, 1)).sum()
    # finish training, loss = -1 * lower bound of mutual information (keep gradient)
    return mi_lb
