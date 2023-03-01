from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# transformer block as shown in Figure 4(a) in paper
class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        # linear projection that reduces dimensionality for faster computing
        self.fc1 = nn.Linear(d_points, d_model)
        # linear layer that projects back to input dimension
        self.fc2 = nn.Linear(d_model, d_points)
        # position encoding MLP that has two linear layers and one ReLU
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # mapping function gamma that is an MLP with two linear layers and one ReLU
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # phi and psi in paper that correspond to pointwise feature transformation
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        # alpha in paper
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        # k-nearest neighbor parameter
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        # k nearest neighbor
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # position encoding
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        # to compute gamma
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # to compute y
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        # residual
        res = self.fc2(res) + pre
        return res, attn
