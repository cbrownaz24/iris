"""
S4D (Diagonal State Space Model) backbone for IRIS world model.

Replaces the Transformer's self-attention with S4D layers while preserving
the same block structure (pre-norm + residual) and MLP. With d_state=256
and d_model=256, the S4D layer has exactly the same parameter count as
the multi-head self-attention it replaces (~263K per layer).

Training uses FFT-based convolution; generation uses the equivalent
recurrent form with a lightweight state cache.
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class S4Cache:
    """Recurrent state cache for S4, drop-in replacement for KeysValues."""

    def __init__(self, n: int, num_layers: int, d_model: int, d_state: int, device: torch.device) -> None:
        self._states = [torch.zeros(n, d_model, d_state, device=device) for _ in range(num_layers)]
        self._size = 0
        self._n = n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._states[idx]

    def __setitem__(self, idx: int, val: torch.Tensor) -> None:
        self._states[idx] = val

    def __len__(self) -> int:
        return len(self._states)

    @property
    def size(self) -> int:
        return self._size

    def reset(self) -> None:
        for s in self._states:
            s.zero_()
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        self._states = [s[mask] for s in self._states]
        self._n = self._states[0].shape[0]


class S4DLayer(nn.Module):
    """
    Diagonal State Space layer (S4D).

    Continuous SSM:  x'(t) = A x(t) + B u(t),  y(t) = C x(t) + D u(t)
    A is diagonal and negative (stable).  B is fixed to 1/sqrt(d_state).
    Discretized via zero-order hold with learned step size dt.

    Parameters per layer (d_model=256, d_state=256):
        input_proj  : 256*256 + 256 = 65,792
        A_log       : 256*256       = 65,536
        C           : 256*256       = 65,536
        D           : 256           =    256
        log_dt      : 256           =    256
        output_proj : 256*256 + 256 = 65,792
        ──────────────────────────────────────
        Total                       = 263,168  (matches self-attention)
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # A = -exp(A_log), initialized with HiPPO-inspired spacing
        A_init = torch.log(0.5 + torch.arange(d_state, dtype=torch.float32))
        self.A_log = nn.Parameter(A_init.unsqueeze(0).expand(d_model, -1).clone())

        # Output matrix C
        self.C = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / d_state ** 0.5))

        # Skip connection D
        self.D = nn.Parameter(torch.ones(d_model))

        # Discretization step dt, initialized in [0.001, 0.1]
        log_dt_init = torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.log_dt = nn.Parameter(log_dt_init)

        self.dropout = nn.Dropout(dropout)

    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return discretized (A_bar, B_bar)."""
        dt = self.log_dt.exp()                                  # (D,)
        A = -self.A_log.exp()                                   # (D, N)  negative
        A_bar = (A * dt.unsqueeze(-1)).exp()                    # (D, N)  in (0, 1)
        B_bar = (A_bar - 1) / (A * (self.d_state ** 0.5))      # (D, N)
        return A_bar, B_bar

    def _get_kernel(self, T: int) -> torch.Tensor:
        """Compute the causal SSM convolution kernel of length T."""
        A_bar, B_bar = self._discretize()

        # K[t] = sum_n C[d,n] * B_bar[d,n] * A_bar[d,n]^t
        CB = self.C * B_bar                                     # (D, N)
        log_A_bar = torch.log(A_bar.clamp(min=1e-10))           # (D, N)
        powers = torch.arange(T, device=A_bar.device, dtype=A_bar.dtype)
        A_bar_pow = torch.exp(log_A_bar.unsqueeze(-1) * powers) # (D, N, T)
        kernel = (CB.unsqueeze(-1) * A_bar_pow).sum(dim=1)      # (D, T)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolution mode for full sequences. x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape
        u = self.input_proj(x).transpose(1, 2)                  # (B, D, T)

        K = self._get_kernel(T)                                  # (D, T)

        # FFT-based causal convolution
        fft_len = 2 * T
        u_f = torch.fft.rfft(u, n=fft_len, dim=-1)
        K_f = torch.fft.rfft(K, n=fft_len, dim=-1)
        y = torch.fft.irfft(u_f * K_f.unsqueeze(0), n=fft_len, dim=-1)[..., :T]

        y = y + self.D.view(1, -1, 1) * u                       # skip connection
        y = self.dropout(self.output_proj(y.transpose(1, 2)))
        return y

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent mode for generation (handles T >= 1).
        x: (B, T, D), state: (B, D, N) -> y: (B, T, D), new_state: (B, D, N)
        """
        u = self.input_proj(x)                                   # (B, T, D)
        B, T, D = u.shape

        A_bar, B_bar = self._discretize()

        outputs = []
        for t in range(T):
            u_t = u[:, t]                                        # (B, D)
            state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * u_t.unsqueeze(-1)
            y_t = (self.C.unsqueeze(0) * state).sum(-1) + self.D.unsqueeze(0) * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                          # (B, T, D)
        y = self.dropout(self.output_proj(y))
        return y, state


class S4Block(nn.Module):
    """Pre-norm S4D + MLP block with residual connections (mirrors transformer Block)."""

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.s4 = S4DLayer(d_model, d_state, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.s4(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent step. x: (B, T, D), state: (B, D, N)."""
        y, new_state = self.s4.step(self.ln1(x), state)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, new_state


class S4Backbone(nn.Module):
    """Stack of S4 blocks, drop-in replacement for the Transformer backbone."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([
            S4Block(config.embed_dim, config.d_state, config.resid_pdrop)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> S4Cache:
        device = self.ln_f.weight.device
        return S4Cache(n, self.config.num_layers, self.config.embed_dim, self.config.d_state, device)

    def forward(self, sequences: torch.Tensor, past_keys_values=None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        if past_keys_values is None:
            # Training: efficient FFT convolution
            for block in self.blocks:
                x = block(x)
        else:
            # Generation: recurrent mode
            for i, block in enumerate(self.blocks):
                x, new_state = block.step(x, past_keys_values[i])
                past_keys_values[i] = new_state
            past_keys_values._size += sequences.size(1)
        return self.ln_f(x)
