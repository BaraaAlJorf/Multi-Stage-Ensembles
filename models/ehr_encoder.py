import torch
from torch import nn

# Define the transformer and its components
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# EHR-specific transformer with CLS token
class EHRTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., dim_head=64):
        super().__init__()
        self.dim = dim
        self.to_ehr_embedding = nn.Linear(48, dim)  # Assuming EHR input size of 48 as given
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # CLS token initialization
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, ehr):
        ehr = self.to_ehr_embedding(ehr)  # Embed raw EHR data
        b, _, _ = ehr.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # Repeat CLS token for batch
        ehr = torch.cat((cls_tokens, ehr), dim=1)  # Prepend CLS token to embedded EHR
        ehr = self.transformer(ehr)  # Pass through transformer
        v_ehr = self.to_latent(ehr[:, 0])  # Extract the representation from the CLS token
        return v_ehr