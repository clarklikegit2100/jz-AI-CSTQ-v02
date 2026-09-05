"""
BSGM Decoder: Bayesian Swin Graph Mamba Decoder.

Each decoder layer applies (in order):
  1. CellGraphLayer   — kNN GATv2: spatial cell-cell relationships
  2. QueryMamba       — Mamba SSM along query dimension
  3. BayesianDropout
  4. Self-Attention   — masked (keeps DN groups separate)
  5. Deformable Cross-Attention — attends to encoder memory
  6. BayesianDropout
  7. FFN

Track queries (from previous frame) and object queries (new detections)
are concatenated and processed jointly, following Cell-TRACTR DN-Track.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph_layer import CellGraphLayer
from .mamba_module import QueryMamba


# ---------------------------------------------------------------------------
# Bayesian Dropout (active at both train and MC-eval time)
# ---------------------------------------------------------------------------

class BayesianDropout(nn.Module):
    def __init__(self, p: float = 0.1, active_in_eval: bool = False):
        super().__init__()
        self.p = p
        self.active_in_eval = active_in_eval

    def forward(self, x: Tensor) -> Tensor:
        if self.p <= 0:
            return x
        return F.dropout(x, p=self.p, training=(self.training or self.active_in_eval))


# ---------------------------------------------------------------------------
# Deformable Cross-Attention (lightweight, pure PyTorch fallback)
# ---------------------------------------------------------------------------

class DeformableCrossAttention(nn.Module):
    """
    Multi-scale deformable cross-attention between decoder queries and
    encoder memory.

    Pure PyTorch implementation (no CUDA ops required).
    Samples `n_points` offsets per query per level, bilinear-interpolates
    from the flattened spatial memory.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_levels: int = 4,
        n_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.n_points = n_points
        self.head_dim = d_model // num_heads

        # Sampling offsets: query → (num_heads * num_levels * n_points * 2)
        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * n_points * 2)
        # Attention weights: query → (num_heads * num_levels * n_points)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * n_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        # Initialize offsets in a circle pattern per head
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).expand(
            self.num_heads, self.num_levels, self.n_points, 2
        ).clone()
        for i in range(self.n_points):
            grid_init[:, :, i, :] /= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: Tensor,                          # (B, Nq, d_model)
        reference_points: Tensor,               # (B, Nq, 2) or (B, Nq, 4) normalised
        memory: Tensor,                         # (B, sum(HiWi), d_model)
        spatial_shapes: Tensor,                 # (num_levels, 2)  [(H1,W1), ...]
        level_start_index: Tensor,              # (num_levels,)
        memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, Nq, _ = query.shape
        Nm = memory.shape[1]

        value = self.value_proj(memory)  # (B, Nm, d_model)
        if memory_padding_mask is not None:
            value = value.masked_fill(memory_padding_mask.unsqueeze(-1), 0)

        value = value.view(B, Nm, self.num_heads, self.head_dim)  # (B, Nm, H, Dh)
        # NOTE: sampling below must read from `value` (the value_proj output),
        # not the raw `memory` -- the projection was previously computed and
        # then ignored (resolution_scaling_plan.md Phase 2 decoder fix).

        offsets = self.sampling_offsets(query)  # (B, Nq, H*L*P*2)
        offsets = offsets.view(B, Nq, self.num_heads, self.num_levels, self.n_points, 2)

        attn_w = self.attention_weights(query)   # (B, Nq, H*L*P)
        attn_w = attn_w.view(B, Nq, self.num_heads, self.num_levels * self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = attn_w.view(B, Nq, self.num_heads, self.num_levels, self.n_points)

        # Reference points: use (cx, cy)
        ref = reference_points[..., :2].unsqueeze(2).unsqueeze(2).unsqueeze(2)  # (B,Nq,1,1,1,2)

        # Compute sample locations per level
        output = query.new_zeros(B, Nq, self.d_model)
        for lvl, (H_lvl, W_lvl) in enumerate(spatial_shapes.tolist()):
            H_lvl, W_lvl = int(H_lvl), int(W_lvl)
            start = int(level_start_index[lvl].item())
            end = start + H_lvl * W_lvl
            mem_lvl = value[:, start:end].reshape(B, H_lvl, W_lvl, self.num_heads, self.head_dim)

            # Sampling locations for this level
            off = offsets[:, :, :, lvl, :, :]   # (B, Nq, H, P, 2)
            ref_lvl = reference_points[..., :2]  # (B, Nq, 2)
            # Offset scale relative to feature map size
            off_scaled = off / torch.tensor([W_lvl, H_lvl], device=off.device, dtype=off.dtype)
            # Sample points: (B, Nq, H, P, 2) in [0,1]
            pts = ref_lvl.unsqueeze(2).unsqueeze(2) + off_scaled  # (B, Nq, H, P, 2)
            pts = pts.clamp(0, 1)
            # Convert to pixel coordinates
            pts_px = pts * torch.tensor([W_lvl - 1, H_lvl - 1], device=pts.device, dtype=pts.dtype)
            pts_px_norm = pts_px / torch.tensor([W_lvl - 1, H_lvl - 1], device=pts.device, dtype=pts.dtype) * 2 - 1
            # grid_sample on mem_lvl
            # mem_lvl: (B, H_lvl, W_lvl, num_heads, head_dim)
            # Reshape to (B*H, head_dim, H_lvl, W_lvl) for grid_sample
            mem_lvl_gs = mem_lvl.permute(0, 3, 4, 1, 2).reshape(
                B * self.num_heads, self.head_dim, H_lvl, W_lvl
            )
            # pts for grid sample: (B, H, Nq, P, 2) → (B*H, Nq*P, 1, 2)
            pts_gs = pts_px_norm.permute(0, 2, 1, 3, 4).reshape(
                B * self.num_heads, Nq * self.n_points, 1, 2
            )
            # Use (x, y) = (W, H) convention for grid_sample
            pts_gs = pts_gs[..., [0, 1]]

            sampled = F.grid_sample(
                mem_lvl_gs, pts_gs, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # (B*H, head_dim, Nq*P, 1)
            sampled = sampled.squeeze(-1).view(B, self.num_heads, self.head_dim, Nq, self.n_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)  # (B, Nq, H, P, Dh)

            # Weighted sum
            w = attn_w[:, :, :, lvl, :]               # (B, Nq, H, P)
            weighted = (sampled * w.unsqueeze(-1)).sum(dim=3)  # (B, Nq, H, Dh)
            output += weighted.reshape(B, Nq, self.d_model)

        output = self.output_proj(output)
        return self.dropout(output)


# ---------------------------------------------------------------------------
# BSGM Decoder Layer
# ---------------------------------------------------------------------------

class BSGMDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_levels: int = 4,
        n_points: int = 4,
        graph_topk: int = 16,
        graph_heads: int = 4,
        mamba_d_state: int = 16,
        bayesian_p: float = 0.1,
        bayesian_eval: bool = False,
        use_graph: bool = True,
        use_query_mamba: bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_query_mamba = use_query_mamba

        # 1. Cell Graph Layer (spatial kNN GATv2)
        self.graph_layer = CellGraphLayer(
            d_model=d_model,
            num_heads=graph_heads,
            k_neighbors=graph_topk,
            dropout=dropout,
        )
        self.norm_graph = nn.LayerNorm(d_model)

        # 2. Query-level Mamba SSM
        self.query_mamba = QueryMamba(
            d_model=d_model,
            d_state=mamba_d_state,
            d_conv=4,
            expand=2,
        )
        self.norm_mamba = nn.LayerNorm(d_model)

        # 3. Bayesian Dropout (post graph+mamba)
        self.bayes_drop1 = BayesianDropout(bayesian_p, bayesian_eval)

        # 4. Self-Attention (masked)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(d_model)
        self.drop_sa = nn.Dropout(dropout)

        # 5. Deformable Cross-Attention
        self.cross_attn = DeformableCrossAttention(
            d_model=d_model,
            num_heads=nhead,
            num_levels=num_levels,
            n_points=n_points,
            dropout=dropout,
        )
        self.norm_ca = nn.LayerNorm(d_model)
        self.bayes_drop2 = BayesianDropout(bayesian_p, bayesian_eval)

        # 6. FFN
        act = F.relu if activation == "relu" else F.gelu
        self._act = act
        self.ffn_l1 = nn.Linear(d_model, dim_feedforward)
        self.ffn_l2 = nn.Linear(dim_feedforward, d_model)
        self.drop_ffn1 = nn.Dropout(dropout)
        self.drop_ffn2 = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,                            # (B, N_total, d_model)
        reference_points: Tensor,               # (B, N_total, 2 or 4)
        memory: Tensor,                         # (B, Nm, d_model)
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        memory_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,  # (N_total, N_total) bool
        query_pos: Optional[Tensor] = None,     # positional embedding
    ) -> Tensor:
        # 1. Graph layer (spatial cell relationships)
        if self.use_graph:
            tgt2 = self.graph_layer(tgt, reference_points)
            tgt = self.norm_graph(tgt2)

        # 2. Query-level Mamba
        if self.use_query_mamba:
            tgt2 = self.query_mamba(tgt)
            tgt = self.norm_mamba(tgt2)

        # 3. Bayesian dropout
        tgt = self.bayes_drop1(tgt)

        # 4. Self-attention (with optional query positional embedding)
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(
            q, k, tgt,
            attn_mask=self_attn_mask,
        )
        tgt = self.norm_sa(tgt + self.drop_sa(tgt2))

        # 5. Deformable cross-attention
        tgt2 = self.cross_attn(
            query=tgt if query_pos is None else tgt + query_pos,
            reference_points=reference_points,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            memory_padding_mask=memory_padding_mask,
        )
        tgt = self.norm_ca(tgt + self.bayes_drop2(tgt2))

        # 6. FFN
        tgt2 = self.ffn_l2(self.drop_ffn1(self._act(self.ffn_l1(tgt))))
        tgt = self.norm_ffn(tgt + self.drop_ffn2(tgt2))

        return tgt


# ---------------------------------------------------------------------------
# BSGM Decoder (stack of N layers)
# ---------------------------------------------------------------------------

class BSGMDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_levels: int = 4,
        n_points: int = 4,
        graph_topk: int = 16,
        graph_heads: int = 4,
        mamba_d_state: int = 16,
        bayesian_p: float = 0.1,
        bayesian_eval: bool = False,
        return_intermediate: bool = True,
        # Box/query refinement
        use_dab: bool = True,
        use_graph: bool = True,
        use_query_mamba: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BSGMDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                num_levels=num_levels,
                n_points=n_points,
                graph_topk=graph_topk,
                graph_heads=graph_heads,
                mamba_d_state=mamba_d_state,
                use_graph=use_graph,
                use_query_mamba=use_query_mamba,
                bayesian_p=bayesian_p,
                bayesian_eval=bayesian_eval,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        self.use_dab = use_dab

    def forward(
        self,
        tgt: Tensor,                       # (B, N, d_model)
        reference_points: Tensor,          # (B, N, 2 or 4)
        memory: Tensor,                    # (B, Nm, d_model)
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        memory_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # Iterative refinement hooks (optional, for box-refinement head)
        refine_fn=None,                    # callable(hs, ref) → new_ref per layer
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            hs: intermediate hidden states (num_layers, B, N, d_model)
                or just the last layer if return_intermediate=False
            ref: refined reference points (num_layers+1, B, N, 2 or 4)
        """
        intermediate_hs = []
        intermediate_ref = [reference_points]
        ref = reference_points

        for layer in self.layers:
            # Sigmoid the reference for cross-attention (normalised coords)
            ref_input = ref.sigmoid() if not self.use_dab else ref

            tgt = layer(
                tgt=tgt,
                reference_points=ref_input,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_padding_mask=memory_padding_mask,
                self_attn_mask=self_attn_mask,
                query_pos=query_pos,
            )

            if refine_fn is not None:
                ref = refine_fn(tgt, ref)
                intermediate_ref.append(ref)

            if self.return_intermediate:
                intermediate_hs.append(tgt)

        if self.return_intermediate:
            return torch.stack(intermediate_hs, dim=0), torch.stack(intermediate_ref, dim=0)
        return tgt.unsqueeze(0), torch.stack(intermediate_ref, dim=0)
