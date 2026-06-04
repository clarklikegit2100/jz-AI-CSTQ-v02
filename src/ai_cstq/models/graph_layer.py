"""
Cell Graph Layer: GATv2-style spatial message passing for cell-cell interactions.

Builds a k-nearest-neighbour graph from cell reference positions and runs
multi-head graph attention to propagate spatial context between queries.

Reference: "How Attentive are Graph Attention Networks?" (GATv2)
           Brody et al., ICLR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


def knn_graph(
    positions: Tensor,   # (B, N, 2)  — normalised (cx, cy)
    k: int,
    exclude_self: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Build k-NN graph by L2 distance over reference positions.

    Returns:
        indices: (B, N, k)   — neighbour indices for each node
        dists:   (B, N, k)   — corresponding distances
    """
    B, N, _ = positions.shape
    # Pairwise squared distances: (B, N, N)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 2)
    dist2 = (diff ** 2).sum(-1)                              # (B, N, N)
    if exclude_self:
        dist2.diagonal(dim1=1, dim2=2).fill_(float("inf"))

    actual_k = min(k, N - 1 if exclude_self else N)
    dists, indices = torch.topk(dist2, k=actual_k, dim=-1, largest=False)
    dists = torch.sqrt(dists.clamp(min=0))
    return indices, dists


class CellGraphLayer(nn.Module):
    """
    GATv2 message passing over the kNN cell graph.

    For each query (cell) node i:
        e_ij = LeakyReLU( a^T [ W_l h_i || W_r h_j ] )
        α_ij = softmax_j( e_ij / sqrt(head_dim) )
        h_i' = ELU( Σ_j α_ij W_v h_j )
    Multi-head: concat heads → project → add residual.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        k_neighbors: int = 16,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k = k_neighbors
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # GATv2 linear transforms (shared across heads via reshape)
        self.W_l = nn.Linear(d_model, d_model, bias=False)   # left (target)
        self.W_r = nn.Linear(d_model, d_model, bias=False)   # right (source)
        self.attn_vec = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.W_v = nn.Linear(d_model, d_model, bias=False)   # value
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(leaky_relu_slope)

    def forward(
        self,
        x: Tensor,                          # (B, N, d_model)
        ref_pts: Optional[Tensor] = None,  # (B, N, 2) or (B, N, 4)  normalised
    ) -> Tensor:
        """
        Returns updated query embeddings (B, N, d_model).
        If ref_pts is None, falls back to sequential ordering as positions.
        """
        B, N, C = x.shape
        residual = x
        x = self.norm(x)

        # Build positions for kNN
        if ref_pts is not None:
            pos = ref_pts[..., :2].detach()  # (B, N, 2) — use (cx, cy)
        else:
            # Fallback: linear positions along N
            pos = torch.linspace(0, 1, N, device=x.device)
            pos = pos.unsqueeze(-1).expand(N, 2).unsqueeze(0).expand(B, -1, -1)

        # kNN graph
        indices, dists = knn_graph(pos, k=self.k)  # (B, N, k), (B, N, k)
        k_actual = indices.shape[-1]

        # Project left (target) and right (source)
        h_l = self.W_l(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
        h_r = self.W_r(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
        h_v = self.W_v(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)

        # Gather neighbour embeddings
        # indices: (B, N, k)  →  expand to (B, N, k, H, D)
        idx_expand = indices.unsqueeze(-1).unsqueeze(-1).expand(B, N, k_actual, self.num_heads, self.head_dim)

        # h_r_neigh: (B, N, k, H, D)
        h_r_neigh = h_r.unsqueeze(2).expand(B, N, N, self.num_heads, self.head_dim)
        # Avoid materializing full N×N — gather from the indexed nodes
        # Reshape h_r: (B, N, H, D) → (B, 1, N, H, D) → gather along dim=2
        h_r_exp = h_r.unsqueeze(1).expand(B, N, N, self.num_heads, self.head_dim)
        del h_r_neigh  # release early

        idx3 = indices.unsqueeze(-1).unsqueeze(-1).expand(B, N, k_actual, self.num_heads, self.head_dim)
        h_r_k = torch.gather(
            h_r.unsqueeze(1).expand(B, N, N, self.num_heads, self.head_dim),
            dim=2,
            index=idx3,
        )  # (B, N, k, H, D)

        h_v_k = torch.gather(
            h_v.unsqueeze(1).expand(B, N, N, self.num_heads, self.head_dim),
            dim=2,
            index=idx3,
        )  # (B, N, k, H, D)

        # GATv2 attention coefficient
        # e_ij = LeakyReLU( h_l_i + h_r_j ), then dot with a
        h_l_expand = h_l.unsqueeze(2).expand(B, N, k_actual, self.num_heads, self.head_dim)
        e = self.leaky(h_l_expand + h_r_k)               # (B, N, k, H, D)
        # Dot with attention vector per head
        attn_vec = self.attn_vec.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, D)
        e = (e * attn_vec).sum(-1)                        # (B, N, k, H)

        # Distance-based bias (closer neighbours get higher weight)
        dist_bias = -dists.unsqueeze(-1) * 5.0           # (B, N, k, 1)
        e = e + dist_bias

        alpha = torch.softmax(e, dim=2)                   # (B, N, k, H)
        alpha = self.dropout(alpha)

        # Aggregate: weighted sum of neighbour values
        alpha_exp = alpha.unsqueeze(-1)                   # (B, N, k, H, 1)
        agg = (alpha_exp * h_v_k).sum(dim=2)              # (B, N, H, D)
        agg = F.elu(agg)

        # Concat heads and project
        agg = agg.reshape(B, N, C)
        out = self.out_proj(agg)
        return residual + self.dropout(out)
