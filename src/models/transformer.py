"""
Mini Transformer Encoder (CLS-based) for within-block observation encoding,
Decoder Transformer for next-frame prediction,
plus TransformerConfig for the hybrid architecture.
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

    # Working dimension (codebook vectors are projected to this)
    embed_dim: int

    # Mini transformer encoder (within-block spatial attention, CLS output)
    num_encoder_layers: int
    num_encoder_heads: int
    num_cls_tokens: int      # number of learnable CLS tokens prepended

    # STU backbone (across-block temporal modeling)
    stu_input_dim: int   # Koopman lifted dimension (M_k input)
    stu_output_dim: int  # M_k output dimension
    num_stu_layers: int
    num_filters: int     # Hankel eigenvector filters

    # Decoder transformer (full self-attention, predicts next frame)
    num_decoder_layers: int
    num_decoder_heads: int

    # Dropout
    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


# ======================================================================
# Shared building blocks
# ======================================================================

class SelfAttention(nn.Module):
    """Full (non-causal) self-attention."""

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


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + MLP."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout)
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


# ======================================================================
# Mini Transformer Encoder (CLS token output)
# ======================================================================

class MiniTransformerEncoder(nn.Module):
    """
    Small transformer encoder that processes K observation tokens within a
    single block. Learnable CLS tokens are prepended; their outputs are returned.

    Input:  (B, K, embed_dim)
    Output: (B, num_cls, embed_dim)
    """

    def __init__(self, num_layers: int, num_heads: int, embed_dim: int,
                 num_obs_tokens: int, num_cls_tokens: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, embed_dim))
        nn.init.normal_(self.cls_tokens, std=0.02)
        self.pos_emb = nn.Embedding(num_cls_tokens + num_obs_tokens, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, embed_dim) -- embedded observation tokens
        Returns:
            (B, num_cls, embed_dim) -- CLS outputs
        """
        B, K, E = x.shape
        C = self.num_cls_tokens
        cls = self.cls_tokens.expand(B, -1, -1)              # (B, C, E)
        x = torch.cat([cls, x], dim=1)                       # (B, C+K, E)
        x = x + self.pos_emb(torch.arange(C + K, device=x.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x[:, :C]                                       # (B, C, E)


# ======================================================================
# Decoder Transformer (full self-attention)
# ======================================================================

class DecoderTransformer(nn.Module):
    """
    Full self-attention transformer for next-frame prediction.

    Input:  (B, N, embed_dim) -- concatenated [obs_tokens, temporal, action]
    Output: (B, N, embed_dim)
    """

    def __init__(self, num_layers: int, num_heads: int, embed_dim: int,
                 num_tokens: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(num_tokens, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, E = x.shape
        x = x + self.pos_emb(torch.arange(N, device=x.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
