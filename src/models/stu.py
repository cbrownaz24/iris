import torch
import torch.nn as nn


def compute_hankel_eigenvectors(T: int, num_filters: int) -> torch.Tensor:
    """
    Compute top-k eigenvectors of the Hankel matrix Z.
    Z_{ij} = 2 / ((i+j)^3 - (i+j)), with 1-indexed i,j in {1,...,T}.
    Returns: (T, num_filters) float32 tensor of eigenvectors.
    """
    num_filters = min(num_filters, T)  # can't have more filters than matrix dimension
    indices = torch.arange(1, T + 1, dtype=torch.float64)
    ij = indices.unsqueeze(0) + indices.unsqueeze(1)  # (T, T), values from 2 to 2T
    Z = 2.0 / (ij ** 3 - ij)

    eigenvalues, eigenvectors = torch.linalg.eigh(Z)  # ascending order
    top_k_idx = eigenvalues.abs().argsort(descending=True)[:num_filters]
    return eigenvectors[:, top_k_idx].float()  # (T, num_filters)


class STULayer(nn.Module):
    """
    Spectral Transform Unit layer with causal filtering.

    For each of k learned filters (Hankel eigenvectors phi_i):
        c_i[t] = sum_{s<=t} phi_i[s] * x[s]       (causal cumulative projection)
        y[t]  += phi_i[t] * M_i(c_i[t])            (modulate and sum)
    """

    def __init__(self, d_model: int, num_filters: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_filters = num_filters

        filters = compute_hankel_eigenvectors(max_seq_len, num_filters)
        self.register_buffer('filters', filters)  # (max_seq_len, num_filters)

        # One linear per filter (nn.Linear so configure_optimizer classifies the weight)
        self.M = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_filters)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D), causal."""
        B, T, D = x.shape
        phi = self.filters[:T]  # (T, num_filters)

        output = torch.zeros_like(x)
        for k in range(self.num_filters):
            phi_k = phi[:, k].view(1, T, 1)                 # (1, T, 1)
            cumsum_k = torch.cumsum(phi_k * x, dim=1)       # (B, T, D)
            output = output + phi_k * self.M[k](cumsum_k)   # (B, T, D)

        return self.dropout(output)


class STUBlock(nn.Module):
    """Pre-norm STU + MLP block with residual connections."""

    def __init__(self, d_model: int, num_filters: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.stu = STULayer(d_model, num_filters, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class STUBackbone(nn.Module):
    """Stack of STU blocks, drop-in replacement for the Transformer backbone."""

    def __init__(self, d_model: int, num_layers: int, num_filters: int,
                 max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            STUBlock(d_model, num_filters, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
