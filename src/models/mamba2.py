"""
Mamba2 (SSD – State Space Duality) backbone for IRIS world model.

Replaces the Transformer's self-attention with selective state-space layers
while preserving the same block structure (pre-norm + residual) and MLP.
With expand=1, d_state=128, nheads=4, d_conv=4, the Mamba2 layer has ~265K
params per layer vs the transformer attention's ~263K (<1% difference).

Training uses a sequential selective scan (pure PyTorch); generation uses
the equivalent single-step recurrence with conv + SSM state caching.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2Cache:
    """Cache for Mamba2 incremental generation, drop-in replacement for KeysValues."""

    def __init__(self, n: int, num_layers: int, nheads: int, headdim: int,
                 d_state: int, d_inner: int, d_conv: int, device: torch.device) -> None:
        self._ssm_states = [torch.zeros(n, nheads, headdim, d_state, device=device)
                            for _ in range(num_layers)]
        self._conv_states = [torch.zeros(n, d_inner, d_conv - 1, device=device)
                             for _ in range(num_layers)]
        self._size = 0
        self._n = n

    @property
    def size(self) -> int:
        return self._size

    def get_ssm_state(self, layer: int) -> torch.Tensor:
        return self._ssm_states[layer]

    def set_ssm_state(self, layer: int, state: torch.Tensor) -> None:
        self._ssm_states[layer] = state

    def get_conv_state(self, layer: int) -> torch.Tensor:
        return self._conv_states[layer]

    def set_conv_state(self, layer: int, state: torch.Tensor) -> None:
        self._conv_states[layer] = state

    def __len__(self) -> int:
        return len(self._ssm_states)

    def reset(self) -> None:
        for s in self._ssm_states:
            s.zero_()
        for s in self._conv_states:
            s.zero_()
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        self._ssm_states = [s[mask] for s in self._ssm_states]
        self._conv_states = [s[mask] for s in self._conv_states]
        self._n = self._ssm_states[0].shape[0]


class Mamba2Layer(nn.Module):
    """
    Selective state-space layer (Mamba2 / SSD).

    Architecture per forward call:
        1. in_proj -> (x_ssm, gate z, B, C, dt)
        2. causal depthwise conv1d on x_ssm + SiLU
        3. multi-head selective scan  (input-dependent A_bar, B, C)
        4. RMSNorm(y) * SiLU(z)  (gating)
        5. out_proj

    Parameters per layer (d_model=256, expand=1, d_state=128, nheads=4, d_conv=4):
        in_proj  : 256 * 772        = 197,632  (no bias)
        conv1d   : 256*4 + 256      =   1,280
        A_log    : 4                 =       4
        D        : 4                 =       4
        dt_bias  : 4                 =       4
        norm     : 256               =     256
        out_proj : 256 * 256         =  65,536  (no bias)
        ────────────────────────────────────────
        Total                        = 264,716  (~0.6% above attention's 263,168)
    """

    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4,
                 expand: int = 1, nheads: int = 4, ngroups: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        d_inner = expand * d_model
        headdim = d_inner // nheads

        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.nheads = nheads
        self.ngroups = ngroups
        self.headdim = headdim

        # in_proj -> x_ssm(d_inner) + z(d_inner) + B(ngroups*N) + C(ngroups*N) + dt(nheads)
        d_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
        self.in_proj = nn.Linear(d_model, d_proj, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)

        # SSM parameters
        self.A_log = nn.Parameter(torch.zeros(nheads))
        self.D = nn.Parameter(torch.ones(nheads))
        self.dt_bias = nn.Parameter(torch.randn(nheads) * 0.1)

        # Gating norm
        self.norm = RMSNorm(d_inner)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_proj(self, proj: torch.Tensor):
        return proj.split(
            [self.d_inner, self.d_inner,
             self.ngroups * self.d_state, self.ngroups * self.d_state,
             self.nheads],
            dim=-1,
        )

    # ---- full-sequence forward (training) --------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        B, T, _ = x.shape

        proj = self.in_proj(x)                                       # (B, T, d_proj)
        x_ssm, z, B_in, C_in, dt_in = self._split_proj(proj)

        # Causal conv1d
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # Reshape for multi-head SSM
        x_heads = x_ssm.view(B, T, self.nheads, self.headdim)       # (B, T, H, P)
        B_in = B_in.view(B, T, self.ngroups, self.d_state)          # (B, T, G, N)
        C_in = C_in.view(B, T, self.ngroups, self.d_state)          # (B, T, G, N)
        dt = F.softplus(dt_in + self.dt_bias)                        # (B, T, H)
        A = -self.A_log.exp()                                        # (H,)

        # Selective scan
        y = self._selective_scan(x_heads, dt, A, B_in, C_in)        # (B, T, H, P)
        y = y + self.D.view(1, 1, self.nheads, 1) * x_heads         # skip
        y = y.view(B, T, self.d_inner)

        # Gate and project
        y = self.norm(y) * F.silu(z)
        return self.dropout(self.out_proj(y))

    def _selective_scan(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                        B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Sequential selective scan.
        x:  (B, T, H, P)   dt: (B, T, H)   A: (H,)
        B:  (B, T, G, N)   C:  (B, T, G, N)
        Returns: (B, T, H, P)
        """
        Bs, T, H, P = x.shape
        N = self.d_state
        hpg = H // self.ngroups

        h = torch.zeros(Bs, H, P, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            dt_t = dt[:, t]                                          # (B, H)
            A_bar = (A.unsqueeze(0) * dt_t).exp()                    # (B, H)

            # Expand groups -> heads
            B_t = B[:, t].unsqueeze(2).expand(-1, -1, hpg, -1).reshape(Bs, H, N)
            C_t = C[:, t].unsqueeze(2).expand(-1, -1, hpg, -1).reshape(Bs, H, N)
            x_t = x[:, t]                                            # (B, H, P)

            # h = A_bar * h + dt * B * x  (outer product over P, N)
            h = (A_bar.unsqueeze(-1).unsqueeze(-1) * h
                 + dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(2) * x_t.unsqueeze(-1))

            # y = C . h  (contract over N)
            y_t = (C_t.unsqueeze(2) * h).sum(-1)                    # (B, H, P)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)                           # (B, T, H, P)

    # ---- single-step recurrence (generation) -----------------------------

    def step(self, x: torch.Tensor, ssm_state: torch.Tensor,
             conv_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process T>=1 tokens using recurrence + conv state.
        x:          (B, T, d_model)
        ssm_state:  (B, H, P, N)
        conv_state: (B, d_inner, d_conv-1)
        Returns:    y (B, T, d_model), new_ssm_state, new_conv_state
        """
        B, T, _ = x.shape
        A = -self.A_log.exp()
        hpg = self.nheads // self.ngroups
        outputs = []

        for t in range(T):
            proj = self.in_proj(x[:, t])                             # (B, d_proj)
            x_ssm, z, B_in, C_in, dt_in = self._split_proj(proj)

            # Conv step: concat history + current, apply depthwise kernel
            conv_input = torch.cat([conv_state, x_ssm.unsqueeze(-1)], dim=-1)  # (B, d_inner, d_conv)
            x_conv = (conv_input * self.conv1d.weight.squeeze(1).unsqueeze(0)).sum(-1) + self.conv1d.bias
            conv_state = conv_input[:, :, 1:]                        # shift window
            x_conv = F.silu(x_conv)                                  # (B, d_inner)

            # SSM step
            x_heads = x_conv.view(B, self.nheads, self.headdim)     # (B, H, P)
            B_t = B_in.view(B, self.ngroups, self.d_state)
            B_t = B_t.unsqueeze(2).expand(-1, -1, hpg, -1).reshape(B, self.nheads, self.d_state)
            C_t = C_in.view(B, self.ngroups, self.d_state)
            C_t = C_t.unsqueeze(2).expand(-1, -1, hpg, -1).reshape(B, self.nheads, self.d_state)
            dt = F.softplus(dt_in + self.dt_bias)                    # (B, H)
            A_bar = (A.unsqueeze(0) * dt).exp()                      # (B, H)

            ssm_state = (A_bar.unsqueeze(-1).unsqueeze(-1) * ssm_state
                         + dt.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(2) * x_heads.unsqueeze(-1))
            y_t = (C_t.unsqueeze(2) * ssm_state).sum(-1)            # (B, H, P)
            y_t = y_t + self.D.view(1, self.nheads, 1) * x_heads

            y_t = y_t.view(B, self.d_inner)
            y_t = self.norm(y_t) * F.silu(z)
            outputs.append(self.out_proj(y_t))

        y = torch.stack(outputs, dim=1)                              # (B, T, d_model)
        return self.dropout(y), ssm_state, conv_state


class Mamba2Block(nn.Module):
    """Pre-norm Mamba2 + MLP block with residual connections (mirrors transformer Block)."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 nheads: int, ngroups: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba2 = Mamba2Layer(d_model, d_state, d_conv, expand, nheads, ngroups, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba2(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x: torch.Tensor, ssm_state: torch.Tensor,
             conv_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, new_ssm, new_conv = self.mamba2.step(self.ln1(x), ssm_state, conv_state)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, new_ssm, new_conv


class Mamba2Backbone(nn.Module):
    """Stack of Mamba2 blocks, drop-in replacement for the Transformer backbone."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        d_inner = config.expand * config.embed_dim
        self.d_inner = d_inner
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([
            Mamba2Block(config.embed_dim, config.d_state, config.d_conv,
                        config.expand, config.num_heads, 1, config.resid_pdrop)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> Mamba2Cache:
        device = self.ln_f.weight.device
        headdim = self.d_inner // self.config.num_heads
        return Mamba2Cache(n, self.config.num_layers, self.config.num_heads,
                           headdim, self.config.d_state, self.d_inner,
                           self.config.d_conv, device)

    def forward(self, sequences: torch.Tensor, past_keys_values=None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        if past_keys_values is None:
            # Training: full-sequence selective scan
            for block in self.blocks:
                x = block(x)
        else:
            # Generation: recurrent mode with state caching
            for i, block in enumerate(self.blocks):
                x, new_ssm, new_conv = block.step(
                    x,
                    past_keys_values.get_ssm_state(i),
                    past_keys_values.get_conv_state(i),
                )
                past_keys_values.set_ssm_state(i, new_ssm)
                past_keys_values.set_conv_state(i, new_conv)
            past_keys_values._size += sequences.size(1)
        return self.ln_f(x)
