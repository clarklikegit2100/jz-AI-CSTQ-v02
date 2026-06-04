"""
Swin Transformer backbone for BSGM-CellTrack.

Full from-scratch implementation of Swin-T/S/B in pure PyTorch.
Outputs 4-level FPN feature maps at strides {4, 8, 16, 32}.

Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
           Liu et al., ICCV 2021 (arXiv:2103.14030)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
    Partition (B, H, W, C) feature map into non-overlapping windows.
    Returns (B*num_windows, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """
    Reverse window partitioning back to (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Window Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA / SW-MSA) with relative
    position bias table.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww (square)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Relative position bias: table size = (2*Wh-1) × (2*Ww-1), one per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position indices
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (B*num_windows, N, C)   N = window_size^2
        mask: (num_windows, N, N) or None
        """
        BW, N, C = x.shape
        qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (BW, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BW, heads, N, N)

        # Relative position bias
        idx = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[idx].view(N, N, -1)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + bias

        if mask is not None:
            # mask: (num_windows, N, N) — large negative for masked positions
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BW, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """
    One Swin Transformer block: W-MSA or SW-MSA + MLP, with pre-norm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )

        # Stochastic depth (DropPath) — simplified as Dropout on residual scale
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Cyclic-shift attention mask (computed lazily in forward)
        self._attn_mask: Optional[Tensor] = None
        self._mask_hw: Tuple[int, int] = (-1, -1)

    def _build_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[Tensor]:
        if self.shift_size == 0:
            return None
        if self._attn_mask is not None and self._mask_hw == (H, W):
            return self._attn_mask.to(device)

        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, Wh, Ww, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        self._attn_mask = attn_mask
        self._mask_hw = (H, W)
        return attn_mask

    def _pad_to_window(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """Pad H and W to be divisible by window_size."""
        B, H, W, C = x.shape
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        return x, pad_b, pad_r

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad if needed
        x, pad_b, pad_r = self._pad_to_window(x)
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        attn_mask = self._build_attn_mask(Hp, Wp, x.device)

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, Wh, Ww, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().add_(keep)
        return x / keep * random_tensor


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image → patch tokens via non-overlapping 4×4 convolution."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """x: (B, C, H, W)  →  tokens: (B, H/4*W/4, embed_dim), H/4, W/4"""
        B, C, H, W = x.shape
        # Pad to be divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)  # B, embed_dim, H/4, W/4
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B, H_out*W_out, embed_dim
        x = self.norm(x)
        return x, H_out, W_out


# ---------------------------------------------------------------------------
# Patch Merging (downsampling between stages)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """2× spatial downsampling: concat 2×2 neighbour patches → linear project."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor, H: int, W: int) -> Tuple[Tensor, int, int]:
        """x: (B, H*W, C)  →  (B, H/2*W/2, 2C)"""
        B, L, C = x.shape
        # Pad if H or W is odd
        x = x.view(B, H, W, C)
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        H2, W2 = x.shape[1] // 2, x.shape[2] // 2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, H2 * W2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H2, W2


# ---------------------------------------------------------------------------
# Swin Stage
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        downsample: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: Tensor, H: int, W: int) -> Tuple[Tensor, int, int, Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x, H, W)
        x_out = x
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, x_out, x_out.shape[1] // (H * W // H // W), 1  # placeholder


    def forward_with_output(self, x: Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        x_before_merge = x
        if self.downsample is not None:
            x, H_new, W_new = self.downsample(x, H, W)
        else:
            H_new, W_new = H, W
        return x, H_new, W_new, x_before_merge, H, W


# ---------------------------------------------------------------------------
# Full Swin Transformer Backbone
# ---------------------------------------------------------------------------

CONFIGS = {
    "swin_t": dict(embed_dim=96,  depths=[2, 2, 6,  2], num_heads=[3, 6, 12, 24]),
    "swin_s": dict(embed_dim=96,  depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    "swin_b": dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
}


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone producing 4 FPN feature maps.

    Returns:
        List of 4 tensors: [C2, C3, C4, C5]
        C_i has shape (B, out_channels, H/stride_i, W/stride_i)
        strides: [4, 8, 16, 32]
    """

    def __init__(
        self,
        arch: str = "swin_t",
        in_channels: int = 3,
        out_channels: int = 256,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_size: int = 4,
    ):
        super().__init__()
        cfg = CONFIGS[arch]
        embed_dim: int = cfg["embed_dim"]
        depths: List[int] = cfg["depths"]
        num_heads: List[int] = cfg["num_heads"]

        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # 4 Swin stages; all but last have downsample
        self.stages = nn.ModuleList()
        block_idx = 0
        dims = [embed_dim * (2 ** i) for i in range(4)]
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            stage = SwinStage(
                dim=dims[i],
                depth=d,
                num_heads=nh,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[block_idx: block_idx + d],
                downsample=(i < 3),  # no downsample after last stage
            )
            self.stages.append(stage)
            block_idx += d

        # Layer norms for each stage output
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(4)])

        # Lateral projections: each stage dim → out_channels
        self.laterals = nn.ModuleList([
            nn.Conv2d(dims[i], out_channels, kernel_size=1)
            for i in range(4)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        """Load pretrained Swin weights (from official or timm checkpoint)."""
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        # Strip any "backbone." prefix
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            print(f"[SwinBackbone] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[SwinBackbone] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        x: (B, C_in, H, W)
        Returns: [c2, c3, c4, c5]  each (B, out_channels, H/s, W/s)
        """
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        stage_outputs = []
        for i, stage in enumerate(self.stages):
            x, H, W, x_before, H_before, W_before = stage.forward_with_output(x, H, W)
            # Apply layer norm to the stage output (before downsampling)
            normed = self.norms[i](x_before)
            # Reshape to spatial map
            feat = normed.view(-1, H_before, W_before, normed.shape[-1])
            feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            stage_outputs.append(feat)

        # Apply lateral 1×1 convolutions
        outs = [self.laterals[i](stage_outputs[i]) for i in range(4)]
        return outs  # [C2, C3, C4, C5] at strides [4, 8, 16, 32]


# ---------------------------------------------------------------------------
# FPN Neck (top-down feature pyramid)
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """
    Top-down FPN: fuses C5→C4→C3→C2 with upsampling + lateral add.
    All outputs at `out_channels` (= d_model for the transformer).
    """

    def __init__(self, out_channels: int = 256, num_levels: int = 4):
        super().__init__()
        self.out_channels = out_channels
        self.num_levels = num_levels
        # Output convolutions (3×3, groups=1) for each level
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        features: [C2, C3, C4, C5]  all (B, out_channels, H/s, W/s)
        Returns:  [P2, P3, P4, P5]  with top-down fusion
        """
        assert len(features) == self.num_levels
        # Top-down path
        laterals = list(features)  # copy
        for i in range(self.num_levels - 1, 0, -1):
            target_h, target_w = laterals[i - 1].shape[-2:]
            up = F.interpolate(laterals[i], size=(target_h, target_w), mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up

        outs = [self.output_convs[i](laterals[i]) for i in range(self.num_levels)]
        return outs
