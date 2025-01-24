import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

import warnings
import math


def phi(width: int, height: int, p_q: torch.Tensor):
    new_point = p_q.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point


def generate_ref_points(width: int, height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False

    return grid


class DeformableHeadAttention(nn.Module):
    """Deformable Attention Module"""
    def __init__(self, last_height, last_width, C, M=8, K=4, L=1, dropout=0.1, return_attentions=False):
        """
        Args:
            - param C: embedding size of the x's
            - param M: number of attention heads
            - param K: number of sampling points per attention head per feature level
            - param L: number of scale
            - param last_height: smallest feature height
            - param last_width: smallest feature width
            - param dropout: dropout ratio, default = 0.1
            - param return_attentions: return attentions or not (boolean), default = False
        """
        super().__init__()
        assert C % M == 0 # check if C is divisible by M

        self.C_v = C // M
        self.M = M
        self.L = L
        self.K = K

        self.q_proj = nn.Linear(C, C)
        self.delta_proj = nn.Linear(C, 2 * M * K * L) # delta p_q 2 * M * K * L
        self.Attention_projection = nn.Linear(C, M * K * L) # K probabilities per M and L

        self.W_prim = nn.Linear(C, C)
        self.W_m = nn.Linear(C, C)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.dimensions = [[last_height * 2**i, last_width * 2**i] for i in range(self.L)]
        self.return_attentions = True

        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.constant_(self.delta_proj.weight, 0.0)
        torch.nn.init.constant_(self.Attention_projection.weight, 0.0)
        torch.nn.init.constant_(self.Attention_projection.bias, 1 / (self.L * self.K))

        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        bias = self.delta_proj.bias.view(self.M, self.L, self.K, 2)

        init_xy(bias[0], x=-self.K, y=-self.K)
        init_xy(bias[1], x=-self.K, y=0)
        init_xy(bias[2], x=-self.K, y=self.K)
        init_xy(bias[3], x=0, y=-self.K)
        init_xy(bias[4], x=0, y=self.K)
        init_xy(bias[5], x=self.K, y=-self.K)
        init_xy(bias[6], x=self.K, y=0)
        init_xy(bias[7], x=self.K, y=self.K)

    def forward(self, z_q, Xs, p_q, query_mask=None, x_masks=None):
        """
        Args:
            - param x_masks: batch, height, width
            - param query_mask: batch, H, W
            - param z_q: batch, H, W, C, query tensors
            - param Xs: List[batch, H, W, C] list of tensors representing multiscale image
            - param p_q: reference point 1 per pixel B, H, W, 2
        Returns:
            - features      batch, height, width, C
            - Attention     batch, height, width, L, M, K
        """
        if x_masks is None:
            x_masks = [None] * len(Xs)

        output = {'attentions': None, 'deltas': None}

        """Sampling Offsets (delta p_mqk) and Attention Weights (A_mqk)"""
        B, H, W, _ = z_q.shape
        z_q = self.q_proj(z_q) # B, H, W, C
        deltas = self.delta_proj(z_q) # B, H, W, 2MLK
        deltas = deltas.view(B, H, W, self.M, -1) # B, H, W, M, 2LK
        A = self.Attention_projection(z_q) # B, H, W, MLK

        # put as -infinity probas masked (batch, H, W, 1)
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1)
            _, _, _, M_L_K = A.shape
            query_mask_ = query_mask_.expand(B, H, W, M_L_K)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))

        A = A.view(B, H, W, self.M, -1) # B, H, W, M, LK
        A = F.softmax(A, dim=-1) # softmax over the LK probabilities

        # mask nan position
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1) # B, H, W, 1, 1
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0) # mask the possible nan values

        if self.return_attentions:
            output['attentions'] = A # B, H, W, M, LK
            output['deltas'] = deltas # B, H, W, M, 2LK

        deltas = deltas.view(B, H, W, self.M, self.L, self.K, 2) # B, H, W, M, L, K, 2
        deltas = deltas.permute(0, 3, 4, 5, 1, 2, 6).contiguous() # B, M, L, K, H, W, 2
        deltas = deltas.view(B * self.M, self.L, self.K, H, W, 2) # B*M, L, K, H, W, 2
        A = A.permute(0, 3, 1, 2, 4).contiguous() # B, M, H, W, LK
        A = A.view(B * self.M, H * W, -1) # B*M, H*W, LK
        sampled_features_scale_list = []

        """Compute Sampling Using Multi-scale Feature Map"""
        for l in range(self.L):
            x_l = Xs[l] # N, H, W, C
            _, h, w, _ = x_l.shape
            x_l_mask = x_masks[l]

            phi_p_q = phi(height=h, width=w, p_q=p_q) # phi multiscale / B, H, W, 2
            phi_p_q = phi_p_q.repeat(self.M, 1, 1, 1) # repeat M points for every attention head / B*M, H, W, 2
            W_prim_x = self.W_prim(x_l)
            W_prim_x = W_prim_x.view(B, h, w, self.M, self.C_v) # separate the C features into M*C_v vectors / B, h(x_l), w(x_l), M, C_v

            if x_l_mask is not None:
                x_l_mask = x_l_mask.unsqueeze(dim=-1).unsqueeze(dim=-1) # B, h, w, 1, 1
                x_l_mask = x_l_mask.expand(B, h, w, self.M, self.C_v)
                W_prim_x = torch.masked_fill(W_prim_x, mask=x_l_mask, value=0)

            W_prim_x = W_prim_x.permute(0, 3, 4, 1, 2).contiguous() # B, M, C_v, h, w
            W_prim_x = W_prim_x.view(-1, self.C_v, h, w) # B*M, C_v, h, w
            sampled_features = self._compute_sampling(W_prim_x, phi_p_q, deltas, l, h, w)
            sampled_features_scale_list.append(sampled_features)

        """Aggregate Sampled Values"""
        sampled_features_scaled = torch.stack(sampled_features_scale_list, dim=1) # stack L (B*M, K, C_v, H, W) sampled features / B*M, L, K, C_v, H, W
        sampled_features_scaled = sampled_features_scaled.permute(0, 4, 5, 3, 1, 2).contiguous() # B*M, H, W, C_v, L, K
        sampled_features_scaled = sampled_features_scaled.view(B * self.M, H * W, self.C_v, -1) # B*M, H*W, C_v, L*K

        Attention_W_prim_x_plus_delta = torch.einsum('nlds, nls -> nld', sampled_features_scaled, A)
        Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.view(B, self.M, H, W, self.C_v) # B, M, H, W, C_v
        Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.permute(0, 2, 3, 1, 4).contiguous() # B, H, W, M, C_v
        Attention_W_prim_x_plus_delta = Attention_W_prim_x_plus_delta.view(B, H, W, self.C_v * self.M) # B, H, W, M*C_v

        final_features = self.W_m(Attention_W_prim_x_plus_delta)

        if self.dropout:
            final_features = self.dropout(final_features)

        return final_features, output

    def _compute_sampling(self, W_prim_x, phi_p_q, deltas, layer, h, w):
        offseted_features = []

        for k in range(self.K):
            phi_p_q_plus_deltas = phi_p_q + deltas[:, layer, k, :, :, :] # p_q + delta p_mqk
            vgrid_x = 2.0 * phi_p_q_plus_deltas[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * phi_p_q_plus_deltas[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

            sampled = F.grid_sample(W_prim_x, vgrid_scaled, mode='bilinear', padding_mode='zeros') # bilinear interpolation (as explained in deformable convolution)
            offseted_features.append(sampled)

        return torch.stack(offseted_features, dim=3)