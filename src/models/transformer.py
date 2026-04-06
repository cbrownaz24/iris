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
    stu_obs_dim: int     # frame_summary dimension (mini transformer output)
    stu_dim: int         # STU hidden dimension (between MLP_in and MLP_out)
    temporal_state_dim: int  # output dimension of the STU sandwich (MLP_out output)
    num_stu_layers: int  # number of STU blocks
    num_filters: int     # Hankel eigenvector filters

    # Cross-attention decoder (z_t tokens attend to temporal_state)
    num_decoder_layers: int  # number of cross-attention decoder layers
    num_decoder_heads: int   # number of heads in cross-attention
    num_context_tokens: int  # temporal_state is expanded into this many KV tokens

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
                 output_dim: int, num_tokens: int, dropout: float = 0.1,
                 pool: bool = True) -> None:
        super().__init__()
        self.pool = pool
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
            If pool=True:  (B, output_dim) — pooled observation vector
            If pool=False: (B, K, output_dim) — per-token output
        """
        B, K, E = x.shape
        x = x + self.pos_emb(torch.arange(K, device=x.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if self.pool:
            x = x.mean(dim=1)        # mean pool over K tokens -> (B, embed_dim)
        return self.pool_proj(x)      # (B, output_dim) or (B, K, output_dim)


# ======================================================================
# Cross-Attention Decoder: z_t tokens attend to temporal_state
# ======================================================================

class CrossAttention(nn.Module):
    """Queries attend to context (key/value). Context does NOT attend to queries."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, K, embed_dim)  — query tokens (z_t)
            context: (B, S, embed_dim)  — key/value (projected temporal_state)
        Returns:
            (B, K, embed_dim)
        """
        B, K, C = x.shape
        S = context.size(1)
        nh, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, K, nh, hd).permute(0, 2, 1, 3)          # (B, nh, K, hd)
        kv = self.kv_proj(context).reshape(B, S, 2, nh, hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                     # (B, nh, S, hd)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                                              # (B, nh, K, hd)
        y = rearrange(y, 'b h t e -> b t (h e)')
        return self.resid_drop(self.out_proj(y))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_ctx = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.ln_q(x), self.ln_ctx(context))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttentionDecoder(nn.Module):
    """
    z_t observation tokens (queries) attend to temporal_state (key/value).
    temporal_state does NOT attend to z_t — one-directional cross-attention.

    temporal_state is expanded into ``num_context_tokens`` KV vectors so that
    different query tokens can selectively attend to different aspects of the
    temporal context.

    Input:  z_tokens (B, K, embed_dim), temporal_state (B, temporal_state_dim)
    Output: (B, K, embed_dim) — enriched token representations
    """

    def __init__(self, embed_dim: int, temporal_state_dim: int,
                 num_heads: int, num_layers: int,
                 num_context_tokens: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_context_tokens = num_context_tokens
        self.context_expand = nn.Linear(temporal_state_dim, num_context_tokens * embed_dim)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, z_tokens: torch.Tensor, temporal_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tokens:       (B, K, embed_dim) — embedded observation tokens
            temporal_state: (B, temporal_state_dim) — output of STU sandwich
        Returns:
            (B, K, embed_dim) — cross-attention enriched tokens
        """
        B = temporal_state.size(0)
        M = self.num_context_tokens
        context = self.context_expand(temporal_state).view(B, M, -1)  # (B, M, embed_dim)
        x = z_tokens
        for layer in self.layers:
            x = layer(x, context)
        return self.ln_f(x)
