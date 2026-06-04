"""
HybridCellTracker — EmbedTrack-style segmentation + BSGM-style tracking.

Architecture:
  Shared backbone (Swin-T) + FPN + Temporal Mamba
      ├── Segmentation branch: pixel-level offset embeddings (EmbedTrack)
      │       EmbedSegDecoder → seg_offsets, bandwidth, seediness, track_offsets
      └── Tracking branch:    sparse track queries   (BSGM)
              MaskPoolQueryInit (bridge) → BSGMDecoder → cls + box + uncertainty

Key innovation — MaskPoolQueryInit (bridge module):
  Maps instance masks from t-1 to track query content for frame t by
  mask-average-pooling FPN features.  During training uses GT masks
  (differentiable); at inference uses predicted clustering masks.

Reference papers:
  EmbedTrack  — Löffler & Mikut, IEEE TMI 2022
  Cell-TRACTR — O'Connor & Dunlop, bioRxiv 2024
  MOTR        — Zeng et al., ECCV 2022  (track query concept)
  Kaiser 2025 — Mitosis-aware MHT built on EmbedTrack
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .swin_backbone import SwinTransformerBackbone, FPNNeck
from .mamba_module import MultiScaleTemporalMamba
from .bsgm_decoder import BSGMDecoder, BayesianDropout
from .bsgm_net import (
    PositionEmbeddingSine,
    DeformableEncoderLayer, DeformableEncoder,
    MLP, inverse_sigmoid,
)


# ---------------------------------------------------------------------------
# Segmentation branch: pixel-level embedding decoder
# ---------------------------------------------------------------------------

class EmbedSegDecoder(nn.Module):
    """
    Decodes FPN features into dense per-pixel predictions (EmbedTrack style).

    Uses a 4-level top-down decoder with skip connections from the FPN.

    Outputs per pixel (all at full H×W resolution):
        seg_offsets  : (B, 2, H, W)  — dx,dy to cell centre
        bandwidth    : (B, 2, H, W)  — sx,sy clustering radius (> 0)
        seediness    : (B, 1, H, W)  — foreground score [0,1]
        track_offsets: (B, 2, H, W)  — dx,dy to cell centre at t-1
    """

    def __init__(self, d_model: int = 256, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels

        # Lateral 1×1 projections to keep everything at d_model channels
        self.up_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
                nn.GroupNorm(32, d_model),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels)
        ])

        # Final prediction head at stride-4 (highest resolution level)
        # Predicts: dx, dy (seg), sx, sy (bw), seed, track_dx, track_dy → 7 ch
        self.pred_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, 7, 1),
        )

    def forward(self, fpn_features: List[Tensor]) -> Dict[str, Tensor]:
        """
        fpn_features: [P2, P3, P4, P5]  each (B, d_model, Hi, Wi)
        Returns dict of dense predictions at P2 resolution (stride 4).
        """
        # Top-down fusion: P5→P4→P3→P2
        x = fpn_features[-1]   # coarsest level
        laterals = list(fpn_features)

        for i in range(self.num_levels - 1, 0, -1):
            up = F.interpolate(x, size=laterals[i - 1].shape[-2:], mode="nearest")
            x = self.up_convs[i](laterals[i - 1] + up)

        # x is now at P2 resolution (stride 4, highest res)
        pred = self.pred_head(x)          # (B, 7, H/4, W/4)

        seg_offsets   = pred[:, 0:2]
        bandwidth     = F.softplus(pred[:, 2:4])   # sx, sy > 0
        seediness     = pred[:, 4:5].sigmoid()
        track_offsets = pred[:, 5:7]

        return {
            "seg_offsets":    seg_offsets,
            "bandwidth":      bandwidth,
            "seediness":      seediness,
            "track_offsets":  track_offsets,
        }


# ---------------------------------------------------------------------------
# Bridge module: MaskPoolQueryInit
# ---------------------------------------------------------------------------

class MaskPoolQueryInit(nn.Module):
    """
    Initialise track query content for frame t by mask-average-pooling
    the FPN features of frame t-1 using the corresponding instance masks.

    This is the key bridge connecting the two branches:
      - seg branch  →  instance masks (GT during training, predicted at infer)
      - track branch ←  mask-pooled FPN features as query content

    Inputs
    ------
    fpn_feat  : (B, d_model, H_p, W_p)  highest-res FPN level of frame t-1
    masks     : (B, M, H_m, W_m)        binary instance masks (0/1 float)
    boxes     : (B, M, 4)               normalised cxcywh boxes (for pos embed)

    Returns
    -------
    query_content : (B, M, d_model)
    query_pos     : (B, M, 4)           same as input boxes (passed through)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        fpn_feat: Tensor,
        masks: Tensor,
        boxes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        B, d, Hf, Wf = fpn_feat.shape
        _, M, Hm, Wm = masks.shape

        # Resize masks to FPN feature resolution
        if (Hm, Wm) != (Hf, Wf):
            masks = F.interpolate(
                masks.float(), size=(Hf, Wf), mode="bilinear", align_corners=False
            )

        # Flatten spatial dims for batched gather
        feat_flat = fpn_feat.view(B, d, -1)          # (B, d, Hf*Wf)
        mask_flat = masks.view(B, M, -1)             # (B, M, Hf*Wf)

        # Weighted average pool (soft mask)
        weight = mask_flat / (mask_flat.sum(-1, keepdim=True).clamp(min=1.0))
        query_content = torch.bmm(weight, feat_flat.permute(0, 2, 1))  # (B, M, d)

        query_content = self.proj(query_content)     # (B, M, d_model)
        return query_content, boxes                  # boxes used as pos embed


# ---------------------------------------------------------------------------
# Main hybrid model
# ---------------------------------------------------------------------------

class HybridCellTracker(nn.Module):
    """
    Hybrid cell tracker combining:
      - EmbedTrack-style pixel embedding segmentation
      - BSGM-style track query tracking

    Parameters
    ----------
    (see build_hybrid_model factory at the bottom)
    """

    def __init__(
        self,
        # Backbone
        backbone_arch: str = "swin_t",
        backbone_in_channels: int = 3,
        swin_window_size: int = 7,
        swin_pretrained: Optional[str] = None,
        # Transformer shared
        d_model: int = 256,
        num_feature_levels: int = 4,
        # Temporal Mamba
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        # Segmentation branch
        # (uses EmbedSegDecoder, no extra params needed)
        # Tracking branch (BSGM decoder)
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_points: int = 4,
        num_queries: int = 300,
        num_classes: int = 1,
        graph_topk: int = 16,
        graph_heads: int = 4,
        bayesian_p: float = 0.1,
        bayesian_eval: bool = False,
        with_div: bool = True,
        with_box_refine: bool = True,
        # Training flags
        tracking: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.tracking = tracking
        self.with_div = with_div
        self.with_box_refine = with_box_refine
        self.num_feature_levels = num_feature_levels
        self.bayesian_eval = bayesian_eval

        # ── Shared backbone ────────────────────────────────────────────────
        self.backbone = SwinTransformerBackbone(
            arch=backbone_arch,
            in_channels=backbone_in_channels,
            out_channels=d_model,
            window_size=swin_window_size,
        )
        if swin_pretrained:
            self.backbone.load_pretrained(swin_pretrained)

        self.fpn_neck = FPNNeck(out_channels=d_model, num_levels=num_feature_levels)

        self.temporal_mamba = MultiScaleTemporalMamba(
            d_model=d_model,
            num_levels=num_feature_levels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
        )

        # ── Segmentation branch (EmbedTrack style) ─────────────────────────
        self.seg_decoder = EmbedSegDecoder(d_model=d_model, num_levels=num_feature_levels)

        # ── Bridge: mask → track query init ────────────────────────────────
        self.mask_pool = MaskPoolQueryInit(d_model=d_model)

        # ── Tracking branch (BSGM style) ───────────────────────────────────
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model // 2)
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, d_model))
        nn.init.normal_(self.level_embed)

        enc_layer = DeformableEncoderLayer(d_model, dim_feedforward, dropout, nhead)
        self.encoder = DeformableEncoder(enc_layer, num_encoder_layers)
        self.enc_bayes_drop = BayesianDropout(bayesian_p, bayesian_eval)

        # Object queries: separate content (d_model) and 2D reference pos
        self.query_embed = nn.Embedding(num_queries, d_model)   # content
        self.query_ref   = nn.Embedding(num_queries, 2)          # (x, y) reference

        self.track_decoder = BSGMDecoder(
            d_model=d_model,
            num_layers=num_decoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_levels=num_feature_levels,
            n_points=n_points,
            graph_topk=graph_topk,
            graph_heads=graph_heads,
            mamba_d_state=mamba_d_state,
            bayesian_p=bayesian_p,
            bayesian_eval=bayesian_eval,
            return_intermediate=True,
        )

        # Prediction heads (one per decoder layer)
        box_dim = 8 if with_div else 4
        self.cls_head = nn.ModuleList([
            nn.Linear(d_model, num_classes + 1) for _ in range(num_decoder_layers)
        ])
        self.box_head = nn.ModuleList([
            MLP(d_model, d_model, box_dim, 3) for _ in range(num_decoder_layers)
        ])
        if with_box_refine:
            self.ref_point_head = MLP(d_model * 2, d_model, 2, 2)

        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        self._init_heads()

    def _init_heads(self):
        for h in self.cls_head:
            nn.init.constant_(h.bias, -math.log((1 - 0.01) / 0.01))
        for h in self.box_head:
            nn.init.constant_(h.layers[-1].weight, 0)
            nn.init.constant_(h.layers[-1].bias, 0)

    # -----------------------------------------------------------------------
    # Feature extraction (shared)
    # -----------------------------------------------------------------------

    def extract_features(self, frames: List[Tensor]):
        """
        frames: list of T tensors each (B, C, H, W)
        Returns fused_fpn (list of FPN tensors for current frame)
        and all_fpn (list per frame).
        """
        all_fpn = []
        for frame in frames:
            swin_feats = self.backbone(frame)
            fpn_feats  = self.fpn_neck(swin_feats)
            all_fpn.append(fpn_feats)

        current_idx = len(frames) // 2
        fused_fpn = self.temporal_mamba(all_fpn, current_idx)
        return fused_fpn, all_fpn

    # -----------------------------------------------------------------------
    # Encoder helpers
    # -----------------------------------------------------------------------

    def _flatten_fpn(self, fpn_features: List[Tensor]):
        srcs, poses, masks, shapes = [], [], [], []
        for lvl, feat in enumerate(fpn_features):
            B, C, H, W = feat.shape
            pos = self.pos_enc(feat).flatten(2).transpose(1, 2)
            src = feat.flatten(2).transpose(1, 2) + self.level_embed[lvl]
            srcs.append(src)
            poses.append(pos)
            masks.append(torch.zeros(B, H * W, dtype=torch.bool, device=feat.device))
            shapes.append((H, W))
        src  = torch.cat(srcs,  dim=1)
        pos  = torch.cat(poses, dim=1)
        mask = torch.cat(masks, dim=1)
        shp  = torch.tensor(shapes, dtype=torch.long, device=src.device)
        lvl_start = torch.cat([shp.new_zeros(1), shp.prod(1).cumsum(0)[:-1]])
        return src, pos, mask, shp, lvl_start

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        frames: List[Tensor],
        targets: Optional[List[Dict]] = None,
        # Track queries from previous frame (masks + boxes)
        track_masks: Optional[Tensor] = None,   # (B, Nt, H_m, W_m)
        track_boxes: Optional[Tensor] = None,   # (B, Nt, 4) normalised cxcywh
        # FPN features of previous frame (for MaskPool bridge)
        prev_fpn_feat: Optional[Tensor] = None, # (B, d_model, H_p, W_p)
    ) -> Dict:
        """
        frames: [frame_{t-1}, frame_t, frame_{t+1}]

        Returns
        -------
        seg_offsets   : (B, 2, H/4, W/4)
        bandwidth     : (B, 2, H/4, W/4)
        seediness     : (B, 1, H/4, W/4)
        track_offsets : (B, 2, H/4, W/4)
        pred_logits   : (B, N_total, C+1)        last decoder layer
        pred_boxes    : (B, N_total, 4)
        pred_logits_aux / pred_boxes_aux          intermediate layers
        uncertainty   : (B, N_total, 1)
        hs_embed      : (B, N_total, d_model)
        num_track     : int   number of track queries prepended
        """
        B = frames[0].shape[0]

        # ── Shared feature extraction ──────────────────────────────────────
        fused_fpn, all_fpn = self.extract_features(frames)
        fpn_t   = fused_fpn                    # current frame FPN (temporally fused)
        # Highest-res FPN level of current frame (stride-4)
        fpn_hr  = fpn_t[0]                     # (B, d_model, H/4, W/4)

        # ── Segmentation branch ────────────────────────────────────────────
        seg_out = self.seg_decoder(fpn_t)

        # ── Encoder (tracking branch) ──────────────────────────────────────
        src, pos, padding_mask, spatial_shapes, level_start = self._flatten_fpn(fpn_t)
        src = self.enc_bayes_drop(src)
        memory = self.encoder(src, pos, padding_mask)

        # ── Query initialisation ───────────────────────────────────────────
        # Object queries (detect new cells)
        tgt_obj = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # (B, N, d)
        ref_obj = inverse_sigmoid(
            self.query_ref.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        )

        num_track = 0
        if self.tracking and track_masks is not None and track_boxes is not None:
            # Bridge: pool FPN_t-1 features using prev-frame masks
            fpn_src = prev_fpn_feat if prev_fpn_feat is not None else fpn_hr
            tgt_trk, ref_trk_boxes = self.mask_pool(fpn_src, track_masks, track_boxes)
            # Reference points from track boxes: take (cx, cy) only → 2D
            ref_trk = inverse_sigmoid(ref_trk_boxes[..., :2].clamp(0.01, 0.99))  # (B, Nt, 2)
            # Prepend track queries
            tgt = torch.cat([tgt_trk, tgt_obj], dim=1)
            ref = torch.cat([ref_trk, ref_obj], dim=1)
            num_track = tgt_trk.shape[1]
        else:
            tgt = tgt_obj
            ref = ref_obj

        # Self-attention mask: keep track / object groups separate
        N_total = tgt.shape[1]
        self_attn_mask = None
        if num_track > 0:
            self_attn_mask = torch.zeros(N_total, N_total, dtype=torch.bool,
                                         device=tgt.device)
            self_attn_mask[:num_track, num_track:] = True
            self_attn_mask[num_track:, :num_track] = True

        # Box refinement closure
        if self.with_box_refine:
            layer_idx = [0]
            def refine_fn(hs: Tensor, ref_pts: Tensor) -> Tensor:
                # ref_pts: (B, N, 2) — only x,y reference coords
                li = layer_idx[0]; layer_idx[0] += 1
                delta = self.box_head[li](hs)[..., :2]    # (B, N, 2)
                return inverse_sigmoid(ref_pts[..., :2].sigmoid()) + delta
        else:
            refine_fn = None

        # ── Decoder ────────────────────────────────────────────────────────
        all_hs, all_refs = self.track_decoder(
            tgt=tgt,
            reference_points=ref,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start,
            memory_padding_mask=padding_mask,
            self_attn_mask=self_attn_mask,
            refine_fn=refine_fn,
        )

        # ── Prediction heads ───────────────────────────────────────────────
        pred_logits_list, pred_boxes_list = [], []
        for li in range(all_hs.shape[0]):
            hs_i   = all_hs[li]
            logits = self.cls_head[li](hs_i)
            boxes  = self.box_head[li](hs_i)
            ref_sig = all_refs[li].sigmoid()
            boxes[..., :2] = boxes[..., :2] + ref_sig[..., :2]
            boxes[..., :4] = boxes[..., :4].sigmoid()
            if self.with_div and boxes.shape[-1] == 8:
                boxes[..., 4:8] = boxes[..., 4:8].sigmoid()
            pred_logits_list.append(logits)
            pred_boxes_list.append(boxes)

        pred_logits = torch.stack(pred_logits_list, 0)
        pred_boxes  = torch.stack(pred_boxes_list, 0)
        last_hs     = all_hs[-1]

        out = {
            # Segmentation branch outputs
            **seg_out,
            # Tracking branch outputs
            "pred_logits":     pred_logits[-1],
            "pred_boxes":      pred_boxes[-1],
            "pred_logits_aux": pred_logits[:-1],
            "pred_boxes_aux":  pred_boxes[:-1],
            "hs_embed":        last_hs,
            "uncertainty":     self.uncertainty_head(last_hs),
            "ref_pts":         all_refs,
            "num_track":       num_track,
            # Current-frame highest-res FPN (to be used as prev_fpn_feat next frame)
            "fpn_hr":          fpn_hr.detach(),
        }
        return out

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def cluster_instances(
        self,
        seg_out: Dict[str, Tensor],
        seed_threshold: float = 0.5,
        min_size: int = 10,
    ) -> Tensor:
        """
        Simple greedy clustering of predicted pixel embeddings
        (simplified version of EmbedTrack clustering for inference).

        Returns binary instance mask tensor (B, M, H, W) and
        centroid boxes (B, M, 4) in normalised cxcywh format.
        """
        seediness    = seg_out["seediness"]    # (B, 1, H, W)
        seg_offsets  = seg_out["seg_offsets"]  # (B, 2, H, W)
        bandwidth    = seg_out["bandwidth"]    # (B, 2, H, W)

        B, _, H, W = seediness.shape
        device = seediness.device

        all_masks  = []
        all_boxes  = []

        for b in range(B):
            fg = (seediness[b, 0] > seed_threshold)         # (H, W) bool

            if not fg.any():
                all_masks.append(seediness.new_zeros(0, H, W))
                all_boxes.append(seediness.new_zeros(0, 4))
                continue

            # Shift pixels to cluster space
            yy = torch.arange(H, device=device, dtype=torch.float32) / H
            xx = torch.arange(W, device=device, dtype=torch.float32) / W
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

            shifted_x = (grid_x + seg_offsets[b, 0]).clamp(0, 1)
            shifted_y = (grid_y + seg_offsets[b, 1]).clamp(0, 1)

            # Foreground pixel coords in shifted space
            fg_idx = fg.nonzero(as_tuple=False)            # (N, 2) row,col
            if fg_idx.shape[0] == 0:
                all_masks.append(seediness.new_zeros(0, H, W))
                all_boxes.append(seediness.new_zeros(0, 4))
                continue

            ex = shifted_x[fg_idx[:, 0], fg_idx[:, 1]]   # (N,)
            ey = shifted_y[fg_idx[:, 0], fg_idx[:, 1]]
            bw_x = bandwidth[b, 0][fg_idx[:, 0], fg_idx[:, 1]].mean().item()
            bw_y = bandwidth[b, 1][fg_idx[:, 0], fg_idx[:, 1]].mean().item()
            bw_x = max(bw_x, 1.0 / W)
            bw_y = max(bw_y, 1.0 / H)

            # Greedy clustering
            assigned = torch.zeros(fg_idx.shape[0], dtype=torch.bool, device=device)
            instance_masks = []
            instance_boxes = []

            for _ in range(fg_idx.shape[0]):
                free = (~assigned).nonzero(as_tuple=True)[0]
                if free.numel() == 0:
                    break
                # Pick the free pixel with highest seediness as seed
                seed_vals = seediness[b, 0][fg_idx[free, 0], fg_idx[free, 1]]
                seed_i = free[seed_vals.argmax()]
                cx_s, cy_s = ex[seed_i].item(), ey[seed_i].item()

                dist2 = ((ex - cx_s) / bw_x) ** 2 + ((ey - cy_s) / bw_y) ** 2
                in_cluster = (dist2 <= 1.0) & (~assigned)

                if in_cluster.sum() < min_size:
                    assigned[seed_i] = True
                    continue

                assigned |= in_cluster
                pix_idx = fg_idx[in_cluster]               # (K, 2) row,col
                mask = torch.zeros(H, W, dtype=torch.bool, device=device)
                mask[pix_idx[:, 0], pix_idx[:, 1]] = True

                rows = pix_idx[:, 0].float()
                cols = pix_idx[:, 1].float()
                cy_box = (rows.mean() / H).clamp(0.01, 0.99)
                cx_box = (cols.mean() / W).clamp(0.01, 0.99)
                bh_box = ((rows.max() - rows.min() + 1) / H).clamp(0.01, 0.99)
                bw_box = ((cols.max() - cols.min() + 1) / W).clamp(0.01, 0.99)

                instance_masks.append(mask)
                instance_boxes.append(torch.stack([cx_box, cy_box, bw_box, bh_box]))

                if assigned.all():
                    break

            if instance_masks:
                all_masks.append(torch.stack(instance_masks, 0).float())  # (M, H, W)
                all_boxes.append(torch.stack(instance_boxes, 0))           # (M, 4)
            else:
                all_masks.append(seediness.new_zeros(0, H, W))
                all_boxes.append(seediness.new_zeros(0, 4))

        return all_masks, all_boxes


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_hybrid_model(cfg: dict) -> HybridCellTracker:
    return HybridCellTracker(
        backbone_arch=cfg.get("backbone", "swin_t"),
        backbone_in_channels=cfg.get("backbone_in_channels", 3),
        swin_window_size=cfg.get("swin_window_size", 7),
        swin_pretrained=cfg.get("swin_pretrained", None),
        d_model=cfg.get("hidden_dim", 256),
        num_feature_levels=cfg.get("num_feature_levels", 4),
        mamba_d_state=cfg.get("mamba_d_state", 16),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        nhead=cfg.get("nheads", 8),
        num_encoder_layers=cfg.get("enc_layers", 2),
        num_decoder_layers=cfg.get("dec_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 1024),
        dropout=cfg.get("dropout", 0.1),
        n_points=cfg.get("dec_n_points", 4),
        num_queries=cfg.get("num_queries", 300),
        num_classes=cfg.get("num_classes", 1),
        graph_topk=cfg.get("graph_topk", 16),
        graph_heads=cfg.get("graph_heads", 4),
        bayesian_p=cfg.get("bayesian_dropout", 0.1),
        bayesian_eval=cfg.get("bayesian_eval", False),
        with_div=cfg.get("with_div", True),
        with_box_refine=cfg.get("with_box_refine", True),
        tracking=cfg.get("tracking", True),
    )
