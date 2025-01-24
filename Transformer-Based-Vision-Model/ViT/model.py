import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim) # patch_vec_size: P^2 * C, latent_vec_dim: D

        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim))

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1) # attach cls_token as much as batch_size (by repeat() method)
        x += self.pos_embedding # pytorch broadcasting
        x = self.dropout(x)

        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.latent_vec_dim = latent_vec_dim
        self.num_heads = num_heads # num_heads: k
        self.head_dim = int(latent_vec_dim / num_heads) # head_dim: D_h = D / k

        # Single Head dim: D_h, Multi-head dim = k * D_h (can process each head at once)
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim) # dim: k * D_h = D
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)   # dim: k * D_h = D
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim) # dim: k * D_h = D

        self.scale = torch.sqrt(latent_vec_dim * torch.ones(1)).to(device)

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # size: ..., (N+1), D
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1) # size: ..., D, (N+1) (K^T)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # size: ..., (N+1), D

        attention = torch.softmax(q @ k / self.scale, dim=-1) # @: dot product
        x = self.dropout(attention) @ v
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention


class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(latent_vec_dim)
        self.layernorm2 = nn.LayerNorm(latent_vec_dim)

        self.msa = MultiheadSelfAttention(latent_vec_dim=latent_vec_dim,
                                          num_heads=num_heads,
                                          drop_rate=drop_rate)

        self.dropout = nn.Dropout(drop_rate)

        self.mlp = nn.Sequential(
            nn.Linear(latent_vec_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, latent_vec_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        z1 = self.layernorm1(x)
        z1, attention = self.msa(z1)
        z1 = self.dropout(z1)
        x = x + z1

        z2 = self.ln2(x)
        z2 = self.mlp(z2)
        x = x + z2

        return x, attention


class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        self.patchembedding = LinearProjection(patch_vec_size=patch_vec_size,
                                               latent_vec_dim=latent_vec_dim,
                                               drop_rate=drop_rate)

        self.transformer = nn.ModuleList([
            TFencoderLayer(latent_vec_dim=latent_vec_dim,
                           num_heads=num_heads,
                           mlp_hidden_dim=mlp_hidden_dim,
                           drop_rate=drop_rate) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(latent_vec_dim),
            nn.Linear(latent_vec_dim, num_classes)
        )

    def forward(self, x):
        attention_list = []

        x = self.patchembedding(x)

        for layer in self.transformer:
            x, attention = layer(x)
            attention_list.append(attention)

        x = self.mlp_head(x[:,0])

        return x, attention_list