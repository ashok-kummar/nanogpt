import torch
import torch.nn as nn
import torch.nn.functional as F

from head import Head

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attentation
    """

    def __init__(self, n_embed, n_heads=2, head_size=6, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out