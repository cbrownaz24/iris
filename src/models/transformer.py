"""
Mini Transformer Encoder for within-block observation encoding,
plus TransformerConfig for the hybrid mini-transformer + STU world model.
"""

from dataclasses import dataclass
import math

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str       # unused by STU, kept for config compatibility

    # Mini transformer encoder (within-block spatial attention)
    num_layers: int      # mini transformer layers
    num_heads: int       # mini transformer heads
    embed_dim: int       # per-token embedding dimension

    # STU backbone (across-block temporal modeling)
    stu_obs_dim: int     # pooled obs vector dimension (mini transformer output)
    stu_dim: int         # STU hidden dimension
    num_stu_layers: int  # number of STU blocks
    num_filters: int     # Hankel eigenvector filters

    # Dropout
    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class MiniSelfAttention(nn.Module):
    """Full (non-causal) self-attention for the mini encoder."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, nh, T, hd)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hd)
        y = rearrange(y, 'b h t e -> b t (h e)')
        return self.resid_drop(self.proj(y))


class MiniBlock(nn.Module):
    """Pre-norm transformer block for the mini encoder."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MiniSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerEncoder(nn.Module):
    """
    Small transformer encoder that processes K observation tokens within a
    single block and pools them into a single vector.

    Input:  (B, K, embed_dim)  — already-embedded observation tokens
    Output: (B, output_dim)    — pooled observation vector
    """

    def __init__(self, num_layers: int, num_heads: int, embed_dim: int,
                 output_dim: int, num_tokens: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(num_tokens, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MiniBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.pool_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, embed_dim) — embedded observation tokens
        Returns:
            (B, output_dim) — pooled observation vector
        """
        B, K, E = x.shape
        x = x + self.pos_emb(torch.arange(K, device=x.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)            # mean pool over K tokens -> (B, embed_dim)
        return self.pool_proj(x)      # (B, output_dim)
