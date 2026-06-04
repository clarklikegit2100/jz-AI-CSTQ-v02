"""
BSGMCellTrack: Bayesian Swin Graph Mamba Cell Tracker — main model.

End-to-end CTC cell tracking: simultaneous detection, segmentation, and tracking.

Architecture:
  Swin-T Backbone → FPN Neck → Temporal Mamba fusion →
  Deformable Encoder → BSGM Decoder →
  [Classification | Box | Mask | Uncertainty] Heads

Supports:
  - Track queries (previous-frame cells → current-frame tracking)
  - Cell divisions (8D bounding boxes for parent→2 children)
  - DN-Track (denoised tracking supervision during training)
  - MC Dropout uncertainty (multiple forward passes at inference)
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


# ---------------------------------------------------------------------------
# Position Encoding (2D sine)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, C, H, W)  →  pos: (B, 2*num_pos_feats, H, W)"""
        B, C, H, W = x.shape
        if mask is None:
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # (B, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 2*num_pos_feats, H, W)
        return pos


# ---------------------------------------------------------------------------
# Deformable Encoder (pure PyTorch, no CUDA ops)
# ---------------------------------------------------------------------------

class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, d_ffn: int = 1024, dropout: float = 0.1, nhead: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(d_model, d_ffn)
        self.lin2 = nn.Linear(d_ffn, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, pos: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        q = k = src + pos
        src2, _ = self.self_attn(q, k, src, key_padding_mask=padding_mask)
        src = self.norm1(src + self.drop1(src2))
        src2 = self.lin2(self.drop2(F.relu(self.lin1(src))))
        src = self.norm2(src + self.drop3(src2))
        return src


class DeformableEncoder(nn.Module):
    def __init__(self, layer: DeformableEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(
                layer.self_attn.embed_dim,
                layer.lin1.out_features,
                layer.drop1.p,
                layer.self_attn.num_heads,
            )
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, pos: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            src = layer(src, pos, padding_mask)
        return src


# ---------------------------------------------------------------------------
# FPN Pixel Decoder (for instance segmentation masks)
# ---------------------------------------------------------------------------

class FPNPixelDecoder(nn.Module):
    """
    Light FPN pixel decoder: fuses P2–P5 → high-res mask features.
    Returns (B, mask_channels, H/4, W/4) — same resolution as C2.
    """

    def __init__(self, d_model: int = 256, mask_channels: int = 128, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        # Lateral convolutions (already at d_model from backbone+FPN)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, 1) for _ in range(num_levels)
        ])
        # Output convolutions (applied after top-down fusion)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, 3, padding=1),
                nn.GroupNorm(32, d_model),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels)
        ])
        # Final projection to mask_channels at C2 scale
        self.mask_proj = nn.Conv2d(d_model, mask_channels, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fpn_features: List[Tensor]) -> Tensor:
        """
        fpn_features: [P2, P3, P4, P5] (B, d_model, H/s, W/s)
        Returns: mask_features (B, mask_channels, H/4, W/4)
        """
        laterals = [self.lateral_convs[i](fpn_features[i]) for i in range(self.num_levels)]

        # Top-down fusion
        for i in range(self.num_levels - 1, 0, -1):
            h, w = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=(h, w), mode="nearest")

        outs = [self.output_convs[i](laterals[i]) for i in range(self.num_levels)]
        # Use the highest resolution (C2) as the final mask features
        return self.mask_proj(outs[0])


# ---------------------------------------------------------------------------
# Mask Head (per-query mask generation)
# ---------------------------------------------------------------------------

class MaskHead(nn.Module):
    """
    Generate per-instance masks from query embeddings + pixel decoder features.
    query_embed: (B, N, d_model)
    mask_features: (B, mask_channels, H, W)
    Returns: (B, N, H, W) logits
    """

    def __init__(self, d_model: int = 256, mask_channels: int = 128):
        super().__init__()
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, mask_channels),
        )

    def forward(self, query_embed: Tensor, mask_features: Tensor) -> Tensor:
        """
        query_embed:  (B, N, d_model)
        mask_features: (B, C, H, W)
        Returns: (B, N, H, W)
        """
        mask_embed = self.mask_embed(query_embed)  # (B, N, mask_channels)
        B, C, H, W = mask_features.shape
        # Dot product: (B, N, C) × (B, C, H*W) → (B, N, H*W) → (B, N, H, W)
        masks = torch.einsum("bnc,bchw->bnhw", mask_embed, mask_features)
        return masks


# ---------------------------------------------------------------------------
# MLP head (box regression)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Inverse sigmoid helper
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# ---------------------------------------------------------------------------
# BSGMCellTrack: Main Model
# ---------------------------------------------------------------------------

class BSGMCellTrack(nn.Module):
    """
    End-to-end CTC cell tracker with simultaneous detection, segmentation,
    and tracking.

    Key parameters
    --------------
    backbone_arch : str
        One of 'swin_t', 'swin_s', 'swin_b'.
    num_queries : int
        Number of object queries (max cells per frame). Default 400.
    d_model : int
        Transformer hidden dimension. Default 256.
    with_mask : bool
        Enable instance segmentation head.
    with_div : bool
        Enable cell division (8D boxes). Doubles box output dim for division queries.
    track_query_false_positive_prob : float
        Probability of injecting FP track queries during training (robustness).
    """

    def __init__(
        self,
        # Backbone
        backbone_arch: str = "swin_t",
        backbone_in_channels: int = 3,
        swin_window_size: int = 7,
        swin_pretrained: Optional[str] = None,
        # Transformer
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
        n_points: int = 4,
        # Queries
        num_queries: int = 400,
        num_classes: int = 1,
        # Tracking
        tracking: bool = True,
        with_div: bool = True,
        with_mask: bool = True,
        mask_channels: int = 128,
        # Bayesian
        bayesian_p: float = 0.1,
        bayesian_eval: bool = False,
        # Mamba temporal
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_num_temporal_layers: int = 1,
        # Graph
        graph_topk: int = 16,
        graph_heads: int = 4,
        # Training
        two_stage: bool = True,
        with_box_refine: bool = True,
        # DN-Track
        dn_track: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.tracking = tracking
        self.with_div = with_div
        self.with_mask = with_mask
        self.two_stage = two_stage
        self.with_box_refine = with_box_refine
        self.dn_track = dn_track
        self.bayesian_p = bayesian_p
        self.bayesian_eval = bayesian_eval
        self.num_feature_levels = num_feature_levels

        # ----- Backbone -----
        self.backbone = SwinTransformerBackbone(
            arch=backbone_arch,
            in_channels=backbone_in_channels,
            out_channels=d_model,
            window_size=swin_window_size,
        )
        if swin_pretrained:
            self.backbone.load_pretrained(swin_pretrained)

        self.fpn_neck = FPNNeck(out_channels=d_model, num_levels=num_feature_levels)

        # ----- Temporal Mamba -----
        self.temporal_mamba = MultiScaleTemporalMamba(
            d_model=d_model,
            num_levels=num_feature_levels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
        )

        # ----- Positional Encoding -----
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # Level embed: distinguish feature levels
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, d_model))
        nn.init.normal_(self.level_embed)

        # ----- Encoder -----
        enc_layer = DeformableEncoderLayer(d_model, dim_feedforward, dropout, nhead)
        self.encoder = DeformableEncoder(enc_layer, num_encoder_layers)
        self.enc_bayes_drop = BayesianDropout(bayesian_p, bayesian_eval)

        # ----- Two-stage (region proposals from encoder) -----
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            # Enc cls + bbox for proposal generation
            self.enc_cls_head = nn.Linear(d_model, num_classes)
            self.enc_box_head = MLP(d_model, d_model, 4, 3)

        # ----- Object Queries -----
        if not two_stage:
            if with_div:
                self.query_embed = nn.Embedding(num_queries, d_model * 2)  # (content + pos)
            else:
                self.query_embed = nn.Embedding(num_queries, d_model * 2)

        # ----- Decoder -----
        self.decoder = BSGMDecoder(
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

        # ----- Prediction Heads -----
        num_pred_layers = num_decoder_layers + (1 if two_stage else 0)

        # Classification head (shared or per-layer)
        self.cls_head = nn.ModuleList([
            nn.Linear(d_model, num_classes + 1)  # +1 for background / no-object
            for _ in range(num_pred_layers)
        ])

        # Box regression head
        # 4D: (cx, cy, w, h); 8D for divisions: two 4D boxes
        box_dim = 8 if with_div else 4
        self.box_head = nn.ModuleList([
            MLP(d_model, d_model, box_dim, 3)
            for _ in range(num_pred_layers)
        ])

        # Box refinement: predict delta from reference points
        if with_box_refine:
            self.ref_point_head = MLP(d_model * 2, d_model, 2, 2)

        # Mask heads
        if with_mask:
            self.pixel_decoder = FPNPixelDecoder(
                d_model=d_model, mask_channels=mask_channels, num_levels=num_feature_levels
            )
            self.mask_head = MaskHead(d_model=d_model, mask_channels=mask_channels)

        # Uncertainty head: outputs a scalar per query
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        self._init_heads()

    def _init_heads(self):
        for cls_h in self.cls_head:
            nn.init.constant_(cls_h.bias, -math.log((1 - 0.01) / 0.01))  # prior: few cells
        for box_h in self.box_head:
            nn.init.constant_(box_h.layers[-1].weight, 0)
            nn.init.constant_(box_h.layers[-1].bias, 0)

    # -------------------------------------------------------------------------
    # Feature extraction: multi-frame
    # -------------------------------------------------------------------------

    def extract_features(self, frames: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        frames: list of T tensors, each (B, C, H, W)
        Returns:
            fused_fpn: list of `num_feature_levels` tensors (B, d_model, H/s, W/s)
                       temporally fused for current frame (index T//2)
            all_fpn:   list of T lists, each with num_feature_levels tensors
        """
        all_fpn = []
        for frame in frames:
            swin_feats = self.backbone(frame)   # [C2, C3, C4, C5]
            fpn_feats = self.fpn_neck(swin_feats)  # [P2, P3, P4, P5]
            all_fpn.append(fpn_feats)

        # Temporal Mamba fusion (focus on current frame = T//2)
        current_idx = len(frames) // 2
        fused_fpn = self.temporal_mamba(all_fpn, current_idx)
        return fused_fpn, all_fpn

    # -------------------------------------------------------------------------
    # Flatten multi-scale features → encoder input
    # -------------------------------------------------------------------------

    def prepare_encoder_input(
        self, fpn_features: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        """
        Returns:
            src_flat: (B, sum(HiWi), d_model)
            pos_flat: (B, sum(HiWi), d_model)
            mask_flat: (B, sum(HiWi))  — all False (no padding by default)
            spatial_shapes: (L, 2)
            level_start_index: (L,)
        """
        srcs, poses, masks, spatial_shapes = [], [], [], []
        for lvl, feat in enumerate(fpn_features):
            B, C, H, W = feat.shape
            pos = self.pos_enc(feat)
            pos_flat = pos.flatten(2).transpose(1, 2)      # (B, H*W, d_model)
            src_flat = feat.flatten(2).transpose(1, 2)     # (B, H*W, d_model)
            src_flat = src_flat + self.level_embed[lvl]    # level-specific embedding
            mask = torch.zeros(B, H * W, dtype=torch.bool, device=feat.device)

            srcs.append(src_flat)
            poses.append(pos_flat)
            masks.append(mask)
            spatial_shapes.append((H, W))

        src = torch.cat(srcs, dim=1)
        pos = torch.cat(poses, dim=1)
        mask = torch.cat(masks, dim=1)
        shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src.device)
        level_start = torch.cat([
            shapes.new_zeros(1),
            shapes.prod(1).cumsum(0)[:-1],
        ])
        return src, pos, mask, shapes, level_start

    # -------------------------------------------------------------------------
    # Two-stage proposal generation
    # -------------------------------------------------------------------------

    def gen_proposals(self, memory: Tensor, spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate region proposals from encoder memory for two-stage detection.
        Returns:
            topk_coords: (B, num_queries, 4)  normalised reference points
            topk_feats:  (B, num_queries, d_model) query initialisation
        """
        B, S, C = memory.shape
        # Grid proposals per level
        proposals, proposal_scores = [], []
        _cur = 0
        for H_, W_ in spatial_shapes.tolist():
            H_, W_ = int(H_), int(W_)
            mem_lvl = memory[:, _cur: _cur + H_ * W_]  # (B, H*W, C)
            mem_lvl = self.enc_output_norm(self.enc_output(mem_lvl))
            scores = self.enc_cls_head(mem_lvl)         # (B, H*W, num_classes)
            # Grid reference points
            ys, xs = torch.meshgrid(
                (torch.arange(H_, dtype=torch.float32, device=memory.device) + 0.5) / H_,
                (torch.arange(W_, dtype=torch.float32, device=memory.device) + 0.5) / W_,
                indexing="ij",
            )
            wh = torch.tensor([1.0 / W_, 1.0 / H_], device=memory.device)
            grid = torch.stack([xs.flatten(), ys.flatten()], -1)  # (H*W, 2)
            grid_wh = torch.cat([grid, wh.unsqueeze(0).expand(H_ * W_, -1)], -1)  # (H*W, 4)
            proposals.append(grid_wh.unsqueeze(0).expand(B, -1, -1))
            proposal_scores.append(scores.max(-1).values)
            _cur += H_ * W_

        all_proposals = torch.cat(proposals, 1)           # (B, S, 4)
        all_scores = torch.cat(proposal_scores, 1)        # (B, S)

        # Select top-k proposals
        _, topk_idx = torch.topk(all_scores, self.num_queries, dim=1)
        topk_coords = torch.gather(all_proposals, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
        topk_feats = torch.gather(memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
        topk_feats = self.enc_output_norm(self.enc_output(topk_feats))
        return topk_coords, topk_feats

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        frames: List[Tensor],
        targets: Optional[List[Dict]] = None,
        track_query_hs_embeds: Optional[Tensor] = None,  # (B, Nt, d_model)
        track_query_boxes: Optional[Tensor] = None,      # (B, Nt, 4)
    ) -> Dict:
        """
        frames: list of T frame tensors each (B, C_in, H, W).
                Typically T=3: [frame_t-1, frame_t, frame_t+1].
        targets: Optional ground-truth for training.
        track_query_hs_embeds: hidden states from previous frame (for tracking).
        track_query_boxes: bounding boxes from previous frame.

        Returns dict with keys:
            'pred_logits':  (num_layers, B, N_total, num_classes+1)
            'pred_boxes':   (num_layers, B, N_total, 4 or 8)
            'pred_masks':   (B, N_total, H_mask, W_mask)  if with_mask
            'hs_embed':     (B, N_total, d_model)          last decoder layer
            'uncertainty':  (B, N_total, 1)
            'ref_pts':      (num_layers+1, B, N_total, 2 or 4)
            'enc_outputs':  two-stage encoder proposals (if two_stage)
        """
        B = frames[0].shape[0]

        # ---- Feature extraction ----
        fused_fpn, all_fpn = self.extract_features(frames)

        # ---- Pixel decoder for masks (use fused features) ----
        if self.with_mask:
            mask_features = self.pixel_decoder(fused_fpn)  # (B, mask_channels, H/4, W/4)

        # ---- Encoder ----
        src, pos, padding_mask, spatial_shapes, level_start = self.prepare_encoder_input(fused_fpn)
        src = self.enc_bayes_drop(src)
        memory = self.encoder(src, pos, padding_mask)

        # ---- Query initialisation ----
        num_track = 0
        if self.two_stage:
            tgt_coords, tgt_feats = self.gen_proposals(memory, spatial_shapes)
            # tgt_coords: (B, num_queries, 4) in [0,1] normalised
            ref_pts = inverse_sigmoid(tgt_coords)
            tgt = tgt_feats
            enc_outputs = {
                "pred_logits": self.enc_cls_head(tgt_feats),
                "pred_boxes": self.enc_box_head(tgt_feats).sigmoid(),
            }
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            tgt, ref_pts_init = query_embed.chunk(2, dim=-1)
            ref_pts = ref_pts_init.sigmoid()
            ref_pts = inverse_sigmoid(ref_pts)
            enc_outputs = {}

        # ---- Prepend track queries (if any) ----
        if self.tracking and track_query_hs_embeds is not None and track_query_boxes is not None:
            num_track = track_query_hs_embeds.shape[1]
            # Prepend: [track_queries | object_queries]
            tgt = torch.cat([track_query_hs_embeds, tgt], dim=1)
            # ref_pts can be 2D (one-stage) or 4D (two-stage); match track ref dims
            ndim = ref_pts.shape[-1]
            track_ref = inverse_sigmoid(track_query_boxes[..., :ndim].clamp(1e-6, 1 - 1e-6))
            ref_pts = torch.cat([track_ref, ref_pts], dim=1)

        # ---- Build self-attention mask (keep track / detection groups separate) ----
        N_total = tgt.shape[1]
        self_attn_mask = None
        if num_track > 0:
            self_attn_mask = torch.zeros(N_total, N_total, dtype=torch.bool, device=tgt.device)
            # track queries don't attend to object queries and vice versa
            self_attn_mask[:num_track, num_track:] = True
            self_attn_mask[num_track:, :num_track] = True

        # ---- Box refinement closure ----
        if self.with_box_refine:
            # Create per-layer box refinement heads (closure captures layer index)
            layer_idx = [0]

            def refine_fn(hs: Tensor, ref: Tensor) -> Tensor:
                li = layer_idx[0]
                layer_idx[0] += 1
                # Predict delta from reference
                ref_sig = ref.sigmoid()
                delta = self.box_head[li](hs)[..., :4]
                new_ref = inverse_sigmoid(ref_sig) + delta
                return new_ref
        else:
            refine_fn = None

        # ---- Decoder ----
        all_hs, all_refs = self.decoder(
            tgt=tgt,
            reference_points=ref_pts,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start,
            memory_padding_mask=padding_mask,
            self_attn_mask=self_attn_mask,
            refine_fn=refine_fn,
        )
        # all_hs: (num_layers, B, N_total, d_model)
        # all_refs: (num_layers+1, B, N_total, 2)

        # ---- Prediction heads (per decoder layer) ----
        pred_logits_list, pred_boxes_list = [], []
        offset = 1 if self.two_stage else 0  # skip enc head in two_stage
        for li in range(all_hs.shape[0]):
            hs_i = all_hs[li]
            logits = self.cls_head[li + offset](hs_i)    # (B, N_total, num_classes+1)
            boxes = self.box_head[li + offset](hs_i)     # (B, N_total, 4 or 8)
            # Sigmoid for normalised coords
            ref_sig_i = all_refs[li].sigmoid()
            # Add reference offset for first 4 dims
            boxes_sig = boxes.clone()
            boxes_sig[..., :2] = boxes_sig[..., :2] + ref_sig_i[..., :2]
            boxes_sig[..., :4] = boxes_sig[..., :4].sigmoid()
            if self.with_div and boxes.shape[-1] == 8:
                boxes_sig[..., 4:6] = boxes_sig[..., 4:6].sigmoid()
                boxes_sig[..., 6:8] = boxes_sig[..., 6:8].sigmoid()

            pred_logits_list.append(logits)
            pred_boxes_list.append(boxes_sig)

        pred_logits = torch.stack(pred_logits_list, dim=0)  # (L, B, N, C)
        pred_boxes = torch.stack(pred_boxes_list, dim=0)    # (L, B, N, 4 or 8)

        # ---- Mask prediction (last decoder layer only for efficiency) ----
        last_hs = all_hs[-1]  # (B, N_total, d_model)
        pred_masks = None
        if self.with_mask:
            pred_masks = self.mask_head(last_hs, mask_features)  # (B, N_total, H_mask, W_mask)

        # ---- Uncertainty (last decoder layer) ----
        uncertainty = self.uncertainty_head(last_hs)  # (B, N_total, 1)

        out = {
            "pred_logits": pred_logits[-1],       # (B, N_total, num_classes+1) — final layer
            "pred_boxes": pred_boxes[-1],         # (B, N_total, 4 or 8)
            "pred_logits_aux": pred_logits[:-1],  # (L-1, B, N, ...)  auxiliary
            "pred_boxes_aux": pred_boxes[:-1],
            "hs_embed": last_hs,                  # (B, N_total, d_model) for next-frame track
            "uncertainty": uncertainty,
            "ref_pts": all_refs,
        }
        if pred_masks is not None:
            out["pred_masks"] = pred_masks
        if enc_outputs:
            out["enc_outputs"] = enc_outputs

        # Split track vs. object query outputs for tracking logic
        if num_track > 0:
            out["track_query_logits"] = out["pred_logits"][:, :num_track]
            out["object_query_logits"] = out["pred_logits"][:, num_track:]
            out["track_query_boxes"] = out["pred_boxes"][:, :num_track]
            out["object_query_boxes"] = out["pred_boxes"][:, num_track:]

        return out

    # -------------------------------------------------------------------------
    # MC Dropout inference (uncertainty estimation)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def mc_forward(
        self,
        frames: List[Tensor],
        track_query_hs_embeds: Optional[Tensor] = None,
        track_query_boxes: Optional[Tensor] = None,
        num_samples: int = 10,
    ) -> Dict:
        """
        Run `num_samples` stochastic forward passes with dropout active,
        then aggregate mean predictions + epistemic uncertainty (std of logits).
        """
        # Force dropout active
        orig_bayesian_eval = self.bayesian_eval
        for m in self.modules():
            if isinstance(m, BayesianDropout):
                m.active_in_eval = True

        logits_list, boxes_list, masks_list = [], [], []
        for _ in range(num_samples):
            out = self.forward(frames, track_query_hs_embeds=track_query_hs_embeds,
                               track_query_boxes=track_query_boxes)
            logits_list.append(out["pred_logits"])
            boxes_list.append(out["pred_boxes"])
            if "pred_masks" in out:
                masks_list.append(out["pred_masks"])

        logits_stack = torch.stack(logits_list, dim=0)  # (S, B, N, C)
        mean_logits = logits_stack.mean(0)
        std_logits = logits_stack.std(0)  # epistemic uncertainty

        boxes_stack = torch.stack(boxes_list, dim=0)    # (S, B, N, 4)
        mean_boxes = boxes_stack.mean(0)

        # Restore dropout state
        for m in self.modules():
            if isinstance(m, BayesianDropout):
                m.active_in_eval = orig_bayesian_eval

        result = {
            "pred_logits": mean_logits,
            "pred_boxes": mean_boxes,
            "uncertainty": std_logits.mean(-1, keepdim=True),  # (B, N, 1)
            "hs_embed": out["hs_embed"],
        }
        if masks_list:
            result["pred_masks"] = torch.stack(masks_list, 0).mean(0)
        return result


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> "BSGMCellTrack":
    """Build BSGMCellTrack from a config dict."""
    return BSGMCellTrack(
        backbone_arch=cfg.get("backbone", "swin_t"),
        backbone_in_channels=cfg.get("backbone_in_channels", 3),
        swin_window_size=cfg.get("swin_window_size", 7),
        swin_pretrained=cfg.get("swin_pretrained", None),
        d_model=cfg.get("hidden_dim", 256),
        nhead=cfg.get("nheads", 8),
        num_encoder_layers=cfg.get("enc_layers", 4),
        num_decoder_layers=cfg.get("dec_layers", 6),
        dim_feedforward=cfg.get("dim_feedforward", 1024),
        dropout=cfg.get("dropout", 0.1),
        num_feature_levels=cfg.get("num_feature_levels", 4),
        n_points=cfg.get("dec_n_points", 4),
        num_queries=cfg.get("num_queries", 400),
        num_classes=cfg.get("num_classes", 1),
        tracking=cfg.get("tracking", True),
        with_div=cfg.get("with_div", True),
        with_mask=cfg.get("masks", True),
        mask_channels=cfg.get("mask_channels", 128),
        bayesian_p=cfg.get("bayesian_dropout", 0.1),
        bayesian_eval=cfg.get("bayesian_eval", False),
        mamba_d_state=cfg.get("mamba_d_state", 16),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        graph_topk=cfg.get("graph_topk", 16),
        graph_heads=cfg.get("graph_heads", 4),
        two_stage=cfg.get("two_stage", True),
        with_box_refine=cfg.get("with_box_refine", True),
        dn_track=cfg.get("dn_track", False),
    )
