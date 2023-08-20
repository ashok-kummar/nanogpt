import torch
import torch.nn as nn
from torch.nn import functional as F

"""
input params:

n_embed, block_size, head_size, dropout

"""


class Head(nn.Module):
    """
    Head of the Self-attention
    """

    def __init__(self, n_embed=384, block_size=256, head_size=6, dropout=0.2, debug=False):
        
        super().__init__()
        # adding key, query, value layers
        self.K = nn.Linear(n_embed, head_size, bias=False)
        self.Q = nn.Linear(n_embed, head_size, bias=False)
        self.V = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

        self.debug = debug

    def forward(self, x):

        B, T, C = x.shape

        k = self.K(x)                                                   # [B, T, C]   
        q = self.Q(x)                                                   # [B, T, C]        
        v = self.V(x)                                                   # [B, T, C]
        
        W = q @k.transpose(-2, -1) * C**-0.5                            # [B, T, C] @ [B, C, T] -> [B, T, T]    
        W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf'))        # [B, T, T]
        W = F.softmax(W, dim=-1)                                        # [B, T, T]
        W = self.dropout(W)
     
        out = W @ v                                                     # [B, T, T] @ [B, T, C] -> [B, T, C]
        
        if self.debug:
            print(f"k shape: {k.shape}")
            print(f"q shape: {q.shape}")
            print(f"v shape: {v.shape}")
            print(f"W shape: {W.shape}")
            print(f"out shape: {out.shape}")
        
        return out