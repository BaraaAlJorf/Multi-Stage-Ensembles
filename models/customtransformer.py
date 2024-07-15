import torch
from torch import nn

class CustomTransformerLayer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers=1, dropout=0.1):
        super(CustomTransformerLayer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            dropout=dropout
        )
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(self, src):
        # Project the input to the model dimension
        src = self.input_proj(src)
        
        # Since nn.Transformer expects (S, N, E) format, we permute src to fit this
        src = src.permute(1, 0, 2)  # (N, S, E) -> (S, N, E)
        
        # Pass through the transformer
        src = self.transformer(src)
        
        # Permute back to original format
        src = src.permute(1, 0, 2)  # (S, N, E) -> (N, S, E)
        
        # Project back to the original dimension
        src = self.output_proj(src)
        return src
