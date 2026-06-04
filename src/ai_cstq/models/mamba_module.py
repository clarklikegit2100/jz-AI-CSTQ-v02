"""
Mamba-style Selective State Space Model (SSM) for temporal cell tracking.

Pure PyTorch implementation — no mamba-ssm CUDA dependency.
Compatible with Windows and any CUDA/CPU setup.

Architecture follows:
  Mamba: Linear-Time Sequence Modeling with Selective State Spaces
  Gu & Dao, arXiv:2312.00752

Key design: A, B, C matrices are INPUT-DEPENDENT (selective), enabling
the model to decide which temporal information to retain per cell.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Core: Selective Scan (pure PyTorch, causal)
# ---------------------------------------------------------------------------

def selective_scan_sequential(
    u: Tensor,           # (B, L, D)  — input
    delta: Tensor,       # (B, L, D)  — timescale (input-dependent)
    A: Tensor,           # (D, N)     — state matrix (log parameterized)
    B: Tensor,           # (B, L, N) — input matrix (input-dependent)
    C: Tensor,           # (B, L, N) — output matrix (input-dependent)
    D: Tensor,           # (D,)       — skip connection
) -> Tensor:
    """
    Sequential selective scan (recurrence).

    Discretization via Zero-Order Hold (ZOH):
        A_bar = exp(A * Δ)
        B_bar = (A_bar - 1) / A * B * Δ  ≈ Δ * B  (for simplicity, Euler)

    Returns y: (B, L, D)
    """
    B_batch, L, D = u.shape
    N = A.shape[1]
    device = u.device

    # A is (D, N), log-parameterized: actual A = -exp(A_log) (stable negative)
    A_real = -torch.exp(A)  # (D, N)

    # Discretize A: A_bar = exp(A * delta)   shape: (B, L, D, N)
    # delta: (B, L, D)  →  expand to (B, L, D, 1) for broadcast with (D, N)
    delta_A = torch.einsum("bld,dn->bldn", delta, A_real)  # (B, L, D, N)
    A_bar = torch.exp(delta_A)                              # (B, L, D, N)

    # Discretize B: B_bar = delta * B        shape: (B, L, D, N)
    delta_B_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)  # (B, L, D, N)

    # Recurrence
    h = torch.zeros(B_batch, D, N, device=device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + delta_B_u[:, t]        # (B, D, N)
        y_t = torch.einsum("bdn,bn->bd", h, C[:, t])  # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)                         # (B, L, D)
    y = y + u * D                                       # skip connection
    return y


# ---------------------------------------------------------------------------
# Mamba Block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    One Mamba block:
      - Expand projection (expand_factor)
      - Depthwise conv (local context, causal)
      - Selective SSM
      - Gate + output projection
      - Residual

    Input/output: (B, L, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        dt_rank = dt_rank or math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.norm = nn.LayerNorm(d_model)

        # Input projection: d_model → d_inner * 2 (gate + value)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal depthwise conv (conv1d, groups=d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=conv_bias,
        )

        # SSM parameters
        # x → (dt, B, C) projection
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)

        # dt projection (rank → d_inner)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        # Initialize dt_proj bias so softplus(bias) ≈ uniform in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A: (d_inner, d_state), log-parameterized, initialized to -0.5 on average
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: skip connection scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_val, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # Causal depthwise conv (along L)
        x_conv = self.conv1d(x_val.transpose(1, 2))  # (B, d_inner, L+pad)
        x_conv = x_conv[:, :, :x_val.shape[1]]       # trim causal padding
        x_conv = F.silu(x_conv).transpose(1, 2)      # (B, L, d_inner)

        # SSM
        x_proj_out = self.x_proj(x_conv)             # (B, L, dt_rank + 2*d_state)
        dt, B_mat, C_mat = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))            # (B, L, d_inner)

        y = selective_scan_sequential(
            u=x_conv,
            delta=dt,
            A=self.A_log,       # (d_inner, d_state)
            B=B_mat,            # (B, L, d_state)
            C=C_mat,            # (B, L, d_state)
            D=self.D,           # (d_inner,)
        )

        y = y * F.silu(z)                           # gating
        return residual + self.out_proj(y)


# ---------------------------------------------------------------------------
# Temporal Mamba: fuse 3-frame features
# ---------------------------------------------------------------------------

class TemporalMamba(nn.Module):
    """
    Apply Mamba SSM along the temporal (frame) dimension.

    Input:  list of T feature tensors, each (B, d_model, H, W)
    Output: (B, d_model, H, W)  — temporally-fused features for current frame
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, frame_features: list, current_idx: int = 1) -> Tensor:
        """
        frame_features: list of T tensors, each (B, d_model, H, W)
        current_idx:    index of the current frame (default 1 for [t-1, t, t+1])
        Returns: (B, d_model, H, W) — temporally-attended current frame features
        """
        T = len(frame_features)
        B, C, H, W = frame_features[0].shape

        # Reshape: (B, C, H, W) → (B*H*W, C) → assemble temporal sequence
        # x_seq: (B*H*W, T, C)
        x_seq = torch.stack(
            [f.permute(0, 2, 3, 1).reshape(B * H * W, C) for f in frame_features],
            dim=1
        )  # (B*H*W, T, C)

        for layer in self.layers:
            x_seq = layer(x_seq)

        # Extract current frame
        x_cur = x_seq[:, current_idx, :]          # (B*H*W, C)
        x_cur = x_cur.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x_cur


# ---------------------------------------------------------------------------
# Multi-scale Temporal Mamba: apply per FPN level
# ---------------------------------------------------------------------------

class MultiScaleTemporalMamba(nn.Module):
    """
    Apply TemporalMamba independently at each FPN scale.

    Input:  List[List[Tensor]] — outer list = T frames, inner list = L levels
    Output: List[Tensor]       — L fused feature maps for current frame
    """

    def __init__(self, d_model: int, num_levels: int = 4, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.temporal_fusers = nn.ModuleList([
            TemporalMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, num_layers=1)
            for _ in range(num_levels)
        ])

    def forward(self, all_frame_features: list, current_idx: int = 1) -> list:
        """
        all_frame_features: list of T lists, each inner list has num_levels tensors
        Returns: list of num_levels fused tensors
        """
        fused = []
        for lvl in range(self.num_levels):
            level_feats = [all_frame_features[t][lvl] for t in range(len(all_frame_features))]
            fused_lvl = self.temporal_fusers[lvl](level_feats, current_idx)
            fused.append(fused_lvl)
        return fused


# ---------------------------------------------------------------------------
# Query-level Mamba (applied in decoder over N query tokens)
# ---------------------------------------------------------------------------

class QueryMamba(nn.Module):
    """
    Apply Mamba along the query dimension (L = N queries).
    Used inside decoder layers to model query-to-query dependencies
    beyond the kNN graph (global, sequential mixing).

    Input/output: (B, N, d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.block = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N, d_model)"""
        return self.block(x)
