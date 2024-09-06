import torch
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=384, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layernorm(x)
        return self.fc(x)