"""
EmbedTrack-style model for BSGM-CellTrack comparison.

Architecture inspired by:
  Loffler & Mikut (2021), "EmbedTrack - Simultaneous Cell Segmentation and
  Tracking Through Learning Offsets and Clustering Bandwidths"
  https://git.scc.kit.edu/kit-loe-ge/embedtrack

Key differences from BSGM-CellTrack:
  - Dense pixel-level prediction (no object queries / DETR)
  - Branched encoder-decoder (ERFNet-style, here simplified UNet)
  - Input: frame pair [t, t-1] (2 frames)
  - Outputs: segmentation offsets, clustering bandwidths, seediness, tracking offsets
  - Tracking by nearest-neighbor on warped centroids (not track queries)
  - ~5-8M params vs BSGM's ~38M

Outputs per forward pass:
  seg_offsets  : (B, 2, H, W)  — (dx, dy) from pixel to cell center at frame t
  bandwidth    : (B, 2, H, W)  — (sx, sy) clustering radius at frame t
  seediness    : (B, 1, H, W)  — foreground score [0,1] at frame t
  track_offsets: (B, 2, H, W)  — (dx, dy) from pixel at t to cell center at t-1
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """2x spatial downsampling with two conv-bn-relu layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """2x spatial upsampling (bilinear) + skip connection + two conv-bn-relu."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Shared Encoder (UNet-style)
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """
    4-level UNet encoder.  ch = [base, 2x, 4x, 8x, 16x]
    Returns skip connections at each level.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.stem  = nn.Sequential(ConvBnRelu(in_channels, c), ConvBnRelu(c, c))
        self.down1 = DownBlock(c,       c * 2)
        self.down2 = DownBlock(c * 2,   c * 4)
        self.down3 = DownBlock(c * 4,   c * 8)
        self.bottleneck = DownBlock(c * 8, c * 16)
        self.out_ch = c * 16

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Returns (bottleneck, [s0, s1, s2, s3]) skip connections."""
        s0 = self.stem(x)          # (B, c,    H,   W)
        s1 = self.down1(s0)        # (B, 2c,   H/2, W/2)
        s2 = self.down2(s1)        # (B, 4c,   H/4, W/4)
        s3 = self.down3(s2)        # (B, 8c,   H/8, W/8)
        b  = self.bottleneck(s3)   # (B, 16c,  H/16,W/16)
        return b, [s0, s1, s2, s3]


# ---------------------------------------------------------------------------
# Segmentation decoder (offsets + bandwidth + seediness)
# ---------------------------------------------------------------------------

class SegDecoder(nn.Module):
    """
    Decodes a single frame's encoder output into dense pixel predictions.
    Outputs: seg_offsets (2), bandwidth (2), seediness (1) → 5 channels total.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.up3 = UpBlock(c * 16, c * 8,  c * 8)
        self.up2 = UpBlock(c * 8,  c * 4,  c * 4)
        self.up1 = UpBlock(c * 4,  c * 2,  c * 2)
        self.up0 = UpBlock(c * 2,  c,      c)
        self.head = nn.Conv2d(c, 5, kernel_size=1)   # dx, dy, sx, sy, seediness

    def forward(self, bottleneck: Tensor, skips: List[Tensor]) -> Tensor:
        """Returns (B, 5, H, W)."""
        s0, s1, s2, s3 = skips
        x = self.up3(bottleneck, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


# ---------------------------------------------------------------------------
# Tracking decoder (uses concatenated features of frame t and t-1)
# ---------------------------------------------------------------------------

class TrackDecoder(nn.Module):
    """
    Predicts per-pixel tracking offsets (dx, dy) from frame t to cell center in t-1.
    Input: concatenated bottleneck of [frame_t, frame_t-1].
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        in_ch = c * 16 * 2   # concatenated bottlenecks
        # Reduce to single stream then decode
        self.reduce = ConvBnRelu(in_ch, c * 16)
        self.up3 = UpBlock(c * 16, c * 8 * 2, c * 8)   # skip = cat of both frames
        self.up2 = UpBlock(c * 8,  c * 4 * 2, c * 4)
        self.up1 = UpBlock(c * 4,  c * 2 * 2, c * 2)
        self.up0 = UpBlock(c * 2,  c * 2,     c)
        self.head = nn.Conv2d(c, 2, kernel_size=1)      # (dx, dy) tracking offsets

    def forward(
        self,
        bott_t: Tensor, skips_t: List[Tensor],
        bott_tm1: Tensor, skips_tm1: List[Tensor],
    ) -> Tensor:
        """Returns (B, 2, H, W) tracking offsets."""
        x = self.reduce(torch.cat([bott_t, bott_tm1], dim=1))

        s0_t, s1_t, s2_t, s3_t = skips_t
        s0_m, s1_m, s2_m, s3_m = skips_tm1

        x = self.up3(x, torch.cat([s3_t, s3_m], dim=1))
        x = self.up2(x, torch.cat([s2_t, s2_m], dim=1))
        x = self.up1(x, torch.cat([s1_t, s1_m], dim=1))
        x = self.up0(x, torch.cat([s0_t, s0_m], dim=1))  # single skip here
        return self.head(x)


# ---------------------------------------------------------------------------
# Full EmbedTrack-style model
# ---------------------------------------------------------------------------

class EmbedTrackNet(nn.Module):
    """
    EmbedTrack-inspired single-model simultaneous cell segmentation + tracking.

    Input:  frames — list of 2 tensors: [frame_t (B,C,H,W), frame_tm1 (B,C,H,W)]
    Output: dict with keys:
        'seg_offsets'  : (B, 2, H, W)  pixel→cell-center offset at frame t
        'bandwidth'    : (B, 2, H, W)  clustering bandwidth (sx, sy) at frame t
        'seediness'    : (B, 1, H, W)  foreground score [0,1]
        'track_offsets': (B, 2, H, W)  pixel→prev-frame cell-center offset
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
    ):
        super().__init__()
        self.encoder    = SharedEncoder(in_channels, base_ch)
        self.seg_dec    = SegDecoder(base_ch)
        self.track_dec  = TrackDecoder(base_ch)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, frames: List[Tensor]) -> Dict[str, Tensor]:
        """
        frames: [frame_t, frame_tm1]  — current and previous frame.
        """
        assert len(frames) >= 2, "EmbedTrackNet needs at least 2 frames [t, t-1]"
        frame_t   = frames[0]   # current
        frame_tm1 = frames[1]   # previous

        # Encode both frames independently (shared weights)
        bott_t,   skips_t   = self.encoder(frame_t)
        bott_tm1, skips_tm1 = self.encoder(frame_tm1)

        # Segmentation predictions for frame t
        seg_pred = self.seg_dec(bott_t, skips_t)  # (B, 5, H, W)
        seg_offsets = seg_pred[:, :2]              # (B, 2, H, W)  dx, dy
        bandwidth   = F.softplus(seg_pred[:, 2:4]) # (B, 2, H, W)  sx, sy > 0
        seediness   = seg_pred[:, 4:5].sigmoid()  # (B, 1, H, W)  [0,1]

        # Tracking offsets (pixel t → cell center t-1)
        track_offsets = self.track_dec(bott_t, skips_t, bott_tm1, skips_tm1)

        return {
            "seg_offsets":   seg_offsets,
            "bandwidth":     bandwidth,
            "seediness":     seediness,
            "track_offsets": track_offsets,
        }


# ---------------------------------------------------------------------------
# EmbedTrack-style losses (simplified, compatible with dry-run synthetic data)
# ---------------------------------------------------------------------------

def embedtrack_loss(
    outputs: Dict[str, Tensor],
    targets: List[Dict],
    w_seg: float = 1.0,
    w_track: float = 1.0,
) -> Dict[str, Tensor]:
    """
    Compute EmbedTrack-style losses on synthetic targets.

    Expected keys in each target dict:
        'masks'      : (M, H, W) binary instance masks
        'boxes'      : (M, 4) normalised cxcywh bounding boxes
        'track_boxes': (M, 4) boxes of same cells in t-1 (optional)

    Losses computed:
        loss_seed     : BCE on seediness map vs foreground
        loss_offset   : L1 on seg_offsets vs true pixel→center direction
        loss_bandwidth: variance loss — pixels in same instance should agree on bw
        loss_track    : L1 on track_offsets vs true pixel→prev-center direction
    """
    device = outputs["seg_offsets"].device
    B, _, H, W = outputs["seg_offsets"].shape
    losses: Dict[str, Tensor] = {}

    total_seed   = torch.tensor(0.0, device=device)
    total_offset = torch.tensor(0.0, device=device)
    total_bw_var = torch.tensor(0.0, device=device)
    total_track  = torch.tensor(0.0, device=device)

    for b in range(B):
        tgt = targets[b]
        masks = tgt["masks"].to(device)           # (M, H, W) bool or float
        boxes = tgt["boxes"].to(device)           # (M, 4) cxcywh [0,1]

        if masks.shape[0] == 0:
            continue

        # Resize masks to match output if needed
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(
                masks.float().unsqueeze(1), size=(H, W), mode="nearest"
            ).squeeze(1)

        # Foreground map (union of all instance masks)
        fg_map = (masks.sum(0) > 0).float()    # (H, W)

        # 1. Seediness loss — BCE against foreground mask
        seed = outputs["seediness"][b, 0]       # (H, W)
        total_seed = total_seed + F.binary_cross_entropy(
            seed.clamp(1e-6, 1 - 1e-6), fg_map, reduction="mean"
        )

        # 2. Offset loss — for each instance, pixel→center GT
        yy = torch.arange(H, device=device, dtype=torch.float32)
        xx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy / H, xx / W, indexing="ij")

        gt_dx = torch.zeros(H, W, device=device)
        gt_dy = torch.zeros(H, W, device=device)
        gt_track_dx = torch.zeros(H, W, device=device)
        gt_track_dy = torch.zeros(H, W, device=device)

        for m_idx in range(masks.shape[0]):
            mask_m = masks[m_idx] > 0.5          # (H, W)
            if not mask_m.any():
                continue
            cx, cy, bw_m, bh_m = boxes[m_idx].tolist()
            # GT offset: pixel grid → cell center
            gt_dx[mask_m] = cx - grid_x[mask_m]
            gt_dy[mask_m] = cy - grid_y[mask_m]

            # Track: assume same center in t-1 shifted slightly (identity for dry-run)
            if "track_boxes" in tgt and tgt["track_boxes"].shape[0] > m_idx:
                cx_p = tgt["track_boxes"][m_idx, 0].item()
                cy_p = tgt["track_boxes"][m_idx, 1].item()
            else:
                cx_p, cy_p = cx, cy   # fallback: identity motion
            gt_track_dx[mask_m] = cx_p - grid_x[mask_m]
            gt_track_dy[mask_m] = cy_p - grid_y[mask_m]

        pred_dx = outputs["seg_offsets"][b, 0]   # (H, W)
        pred_dy = outputs["seg_offsets"][b, 1]

        if fg_map.sum() > 0:
            total_offset = total_offset + (
                F.l1_loss(pred_dx[fg_map > 0], gt_dx[fg_map > 0]) +
                F.l1_loss(pred_dy[fg_map > 0], gt_dy[fg_map > 0])
            )

        # 3. Bandwidth variance loss
        bw = outputs["bandwidth"][b]             # (2, H, W)
        for m_idx in range(masks.shape[0]):
            mask_m = (masks[m_idx] > 0.5)
            if mask_m.sum() < 2:
                continue
            bw_m_pixels = bw[:, mask_m]          # (2, N_pix)
            total_bw_var = total_bw_var + bw_m_pixels.var(dim=1).mean()

        # 4. Tracking offset loss
        pred_tdx = outputs["track_offsets"][b, 0]
        pred_tdy = outputs["track_offsets"][b, 1]
        if fg_map.sum() > 0:
            total_track = total_track + (
                F.l1_loss(pred_tdx[fg_map > 0], gt_track_dx[fg_map > 0]) +
                F.l1_loss(pred_tdy[fg_map > 0], gt_track_dy[fg_map > 0])
            )

    n = max(B, 1)
    losses["loss_seed"]      = total_seed   / n
    losses["loss_offset"]    = total_offset / n
    losses["loss_bw_var"]    = total_bw_var / n
    losses["loss_track"]     = total_track  / n
    losses["loss_total"] = (
        w_seg   * (losses["loss_seed"] + losses["loss_offset"] + losses["loss_bw_var"]) +
        w_track * losses["loss_track"]
    )
    return losses


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_embedtrack(cfg: dict) -> EmbedTrackNet:
    return EmbedTrackNet(
        in_channels=cfg.get("backbone_in_channels", 3),
        base_ch=cfg.get("embed_base_ch", 32),
    )
