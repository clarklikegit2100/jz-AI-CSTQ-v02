"""
Multi-scale deformable attention (resolution_scaling_plan.md Phase 2).

Pure-PyTorch implementation of Deformable-DETR's `MSDeformAttn` (Zhu et al.,
2021), used to replace the model's *global* encoder self-attention -- which is
`O(S^2)` in the number of multi-scale tokens and OOMs above 256x256 input --
with sparse per-token sampling, `O(S * heads * levels * points)`.

The same module also serves the decoder's cross-attention (queries attend to
the encoder memory with box/point reference points), so encoder and decoder
share one attention primitive.

No custom CUDA op: sampling is done with `F.grid_sample`. Correctness parity
with the reference implementation is checked in `tests/test_ms_deform_attn.py`.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Reference points
# ---------------------------------------------------------------------------

def get_valid_ratios(mask: Tensor, spatial_shapes: Tensor) -> Tensor:
    """
    mask : (B, sum(Hi*Wi)) bool, True = padding.
    Returns (B, n_levels, 2) = (valid_W_fraction, valid_H_fraction) per level.
    With no padding (the default here) this is all ones.
    """
    B = mask.shape[0]
    ratios = []
    start = 0
    for (H_, W_) in spatial_shapes.tolist():
        H_, W_ = int(H_), int(W_)
        m = mask[:, start:start + H_ * W_].view(B, H_, W_)
        valid_h = (~m[:, :, 0]).sum(1).float()
        valid_w = (~m[:, 0, :]).sum(1).float()
        ratios.append(torch.stack([valid_w / W_, valid_h / H_], -1))
        start += H_ * W_
    return torch.stack(ratios, 1)  # (B, n_levels, 2)


def get_encoder_reference_points(
    spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device
) -> Tensor:
    """
    One reference point per encoder token, at its own grid-cell centre,
    normalised to [0, 1] and then broadcast across every level.
    Returns (B, sum(Hi*Wi), n_levels, 2).
    """
    ref_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes.tolist()):
        H_, W_ = int(H_), int(W_)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref_list.append(torch.stack((ref_x, ref_y), -1))       # (B, H*W, 2)
    reference_points = torch.cat(ref_list, 1)                   # (B, sum, 2)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points                                     # (B, sum, n_levels, 2)


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def ms_deform_attn_core_pytorch(
    value: Tensor,                 # (B, S, n_heads, head_dim)
    spatial_shapes: Tensor,        # (n_levels, 2)
    sampling_locations: Tensor,    # (B, Lq, n_heads, n_levels, n_points, 2) in [0, 1]
    attention_weights: Tensor,     # (B, Lq, n_heads, n_levels, n_points)
) -> Tensor:
    B, _, n_heads, head_dim = value.shape
    _, Lq, _, n_levels, n_points, _ = sampling_locations.shape
    split = [int(H) * int(W) for H, W in spatial_shapes.tolist()]
    value_list = value.split(split, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled = []
    for lvl, (H_, W_) in enumerate(spatial_shapes.tolist()):
        H_, W_ = int(H_), int(W_)
        # (B, H*W, n_heads, head_dim) -> (B*n_heads, head_dim, H_, W_)
        v = value_list[lvl].flatten(2).transpose(1, 2).reshape(B * n_heads, head_dim, H_, W_)
        # (B, Lq, n_heads, n_points, 2) -> (B*n_heads, Lq, n_points, 2)
        g = sampling_grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        s = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled.append(s)   # (B*n_heads, head_dim, Lq, n_points)
    # (B*n_heads, 1, Lq, n_levels*n_points)
    attn = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Lq, n_levels * n_points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attn).sum(-1)
    out = out.view(B, n_heads * head_dim, Lq).transpose(1, 2).contiguous()
    return out  # (B, Lq, n_heads*head_dim)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        ).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,                          # (B, Lq, d_model)  pos already added by caller
        reference_points: Tensor,               # (B, Lq, n_levels, 2) or (B, Lq, n_levels, 4)
        input_flatten: Tensor,                  # (B, S, d_model)  the value source (memory)
        input_spatial_shapes: Tensor,           # (n_levels, 2)
        input_level_start_index: Tensor,        # (n_levels,)
        input_padding_mask: Optional[Tensor] = None,   # (B, S) bool, True = pad
    ) -> Tensor:
        B, Lq, _ = query.shape
        S = input_flatten.shape[1]

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(B, S, self.n_heads, self.d_model // self.n_heads)

        offsets = self.sampling_offsets(query).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn = self.attention_weights(query).view(
            B, Lq, self.n_heads, self.n_levels * self.n_points
        )
        attn = F.softmax(attn, -1).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            ).to(offsets.dtype)                                   # (n_levels, 2) = (W, H)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        out = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attn)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Deformable encoder
# ---------------------------------------------------------------------------

class MSDeformEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(
            src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask
        )
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


class MSDeformEncoder(nn.Module):
    def __init__(self, num_layers=4, d_model=256, d_ffn=1024, dropout=0.1,
                 n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.layers = nn.ModuleList([
            MSDeformEncoderLayer(d_model, d_ffn, dropout, n_heads, n_levels, n_points)
            for _ in range(num_layers)
        ])

    def forward(self, src, pos, spatial_shapes, level_start_index, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(
                src.shape[:2], dtype=torch.bool, device=src.device
            )
        valid_ratios = get_valid_ratios(padding_mask, spatial_shapes)
        reference_points = get_encoder_reference_points(
            spatial_shapes, valid_ratios, src.device
        )
        for layer in self.layers:
            src = layer(src, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return src
