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

    M_i maps from d_in to d_out, allowing rectangular projections.
    """

    def __init__(self, d_in: int, d_out: int, num_filters: int, max_seq_len: int) -> None:
        super().__init__()
        self.num_filters = num_filters

        filters = compute_hankel_eigenvectors(max_seq_len, num_filters)
        self.register_buffer('filters', filters)  # (max_seq_len, num_filters)

        # One linear per filter: d_in -> d_out
        self.M = nn.ModuleList([
            nn.Linear(d_in, d_out, bias=False) for _ in range(num_filters)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        phi = self.filters[:T]                              # (T, num_filters)

        # Compute all cumulative projections at once
        phi_expanded = phi.unsqueeze(0).unsqueeze(-1)       # (1, T, num_filters, 1)
        x_expanded = x.unsqueeze(2)                          # (B, T, 1, D)
        weighted = phi_expanded * x_expanded                 # (B, T, num_filters, D)
        cumsum = torch.cumsum(weighted, dim=1)               # (B, T, num_filters, D)

        # Apply all M_k projections — stack weights into one matmul
        # M_weights: (num_filters, d_out, d_in)
        M_weights = torch.stack([self.M[k].weight for k in range(self.num_filters)])
        modulated = torch.einsum('btnf,nof->btno', cumsum, M_weights)  # (B, T, num_filters, d_out)

        output = (phi_expanded * modulated).sum(dim=2)       # (B, T, d_out)
        return output


class STUBackbone(nn.Module):
    """Stack of STU layers."""

    def __init__(self, d_in: int, d_out: int, num_layers: int, num_filters: int,
                 max_seq_len: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            STULayer(d_in, d_out, num_filters, max_seq_len)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> (B, T, d_out)"""
        for layer in self.layers:
            x = layer(x)
        return x
