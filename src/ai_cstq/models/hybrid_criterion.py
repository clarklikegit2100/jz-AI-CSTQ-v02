"""
Joint loss for HybridCellTracker.

Combines:
  L_seg   — EmbedTrack-style dense pixel losses (seed, offset, bandwidth variance)
  L_track — BSGM-style sparse query losses (focal cls, L1 box, GIoU)
  L_aux   — auxiliary losses from intermediate decoder layers

Total:
  L = λ_seg * L_seg + λ_track * L_track + λ_aux * Σ L_aux_i
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .criterion import SetCriterion, sigmoid_focal_loss
from .matcher import HungarianMatcher
from .embedtrack_net import embedtrack_loss


# ---------------------------------------------------------------------------
# Hybrid criterion
# ---------------------------------------------------------------------------

class HybridCriterion(nn.Module):
    """
    Joint loss for the hybrid segmentation + tracking model.

    Parameters
    ----------
    matcher       : HungarianMatcher for the tracking branch
    loss_weights  : dict of scalar weights keyed by loss name prefix
    seg_weights   : dict with keys w_seed, w_offset, w_bw, w_track_offset
    lambda_seg    : overall weight for the segmentation branch losses
    lambda_track  : overall weight for the tracking branch losses
    lambda_aux    : weight for intermediate decoder-layer auxiliary losses
    num_classes   : number of foreground classes (1 = cell)
    focal_alpha, focal_gamma : focal loss hyper-parameters
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        loss_weights: Dict[str, float],
        seg_weights: Dict[str, float],
        lambda_seg: float = 1.0,
        lambda_track: float = 1.0,
        lambda_aux: float = 0.5,
        num_classes: int = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher      = matcher
        self.loss_weights = loss_weights
        self.seg_weights  = seg_weights
        self.lambda_seg   = lambda_seg
        self.lambda_track = lambda_track
        self.lambda_aux   = lambda_aux
        self.num_classes  = num_classes
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma

    # -----------------------------------------------------------------------
    # Tracking branch losses (same as SetCriterion, inlined for clarity)
    # -----------------------------------------------------------------------

    def _loss_labels(self, outputs: dict, targets: list, indices: list,
                     num_cells: float) -> Tensor:
        logits = outputs["pred_logits"]                # (B, N, C+1)
        B, N, Cp1 = logits.shape
        tgt_one_hot = torch.zeros(B, N, Cp1, device=logits.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            tgt_one_hot[b, src_idx, 0] = 1.0          # class 0 = cell
        return sigmoid_focal_loss(
            logits, tgt_one_hot, num_cells,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )

    def _loss_boxes(self, outputs: dict, targets: list, indices: list,
                    num_cells: float) -> Tuple[Tensor, Tensor]:
        from .matcher import box_cxcywh_to_xyxy, generalised_iou
        pred_boxes = outputs["pred_boxes"]
        l1_sum = torch.tensor(0.0, device=pred_boxes.device)
        giou_sum = torch.tensor(0.0, device=pred_boxes.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pb = pred_boxes[b, src_idx, :4]
            tb = targets[b]["boxes"][tgt_idx]
            l1_sum  = l1_sum  + F.l1_loss(pb, tb, reduction="sum") / num_cells
            p_xyxy  = box_cxcywh_to_xyxy(pb)
            t_xyxy  = box_cxcywh_to_xyxy(tb)
            giou_val = generalised_iou(p_xyxy, t_xyxy)
            giou_sum = giou_sum + (1 - giou_val.diag()).sum() / num_cells
        return l1_sum, giou_sum

    def _tracking_loss(self, outputs: dict, targets: list) -> Dict[str, Tensor]:
        num_cells = max(sum(len(t["labels"]) for t in targets), 1)
        indices   = self.matcher(outputs, targets)

        lw = self.loss_weights
        loss_cls  = self._loss_labels(outputs, targets, indices, num_cells)
        loss_l1, loss_giou = self._loss_boxes(outputs, targets, indices, num_cells)

        return {
            "loss_cls":  lw.get("loss_cls",  4.0) * loss_cls,
            "loss_bbox": lw.get("loss_bbox", 5.0) * loss_l1,
            "loss_giou": lw.get("loss_giou", 2.0) * loss_giou,
        }

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, outputs: dict, targets: list) -> Dict[str, Tensor]:
        """
        outputs : dict from HybridCellTracker.forward()
        targets : list of B target dicts with keys:
                    labels   (M,)
                    boxes    (M, 4) normalised cxcywh
                    masks    (M, H, W) binary
        """
        losses: Dict[str, Tensor] = {}

        # ── Segmentation branch losses ──────────────────────────────────────
        sw = self.seg_weights
        seg_targets = [
            {"masks": t["masks"], "boxes": t["boxes"]}
            for t in targets
        ]
        seg_loss_dict = embedtrack_loss(
            outputs={
                "seg_offsets":   outputs["seg_offsets"],
                "bandwidth":     outputs["bandwidth"],
                "seediness":     outputs["seediness"],
                "track_offsets": outputs["track_offsets"],
            },
            targets=seg_targets,
            w_seg=sw.get("w_seg", 1.0),
            w_track=sw.get("w_track_offset", 1.0),
        )
        losses["loss_seg_seed"]   = seg_loss_dict["loss_seed"]
        losses["loss_seg_offset"] = seg_loss_dict["loss_offset"]
        losses["loss_seg_bw_var"] = seg_loss_dict["loss_bw_var"]
        losses["loss_seg_track"]  = seg_loss_dict["loss_track"]

        # ── Tracking branch losses (main decoder layer) ─────────────────────
        track_out_main = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes":  outputs["pred_boxes"],
        }
        for k, v in self._tracking_loss(track_out_main, targets).items():
            losses[k] = v

        # ── Auxiliary layers ────────────────────────────────────────────────
        if "pred_logits_aux" in outputs:
            for li in range(outputs["pred_logits_aux"].shape[0]):
                aux_out = {
                    "pred_logits": outputs["pred_logits_aux"][li],
                    "pred_boxes":  outputs["pred_boxes_aux"][li],
                }
                for k, v in self._tracking_loss(aux_out, targets).items():
                    losses[f"{k}_aux{li}"] = self.lambda_aux * v

        # ── Weighted total ──────────────────────────────────────────────────
        seg_total = (
            losses["loss_seg_seed"]   +
            losses["loss_seg_offset"] +
            losses["loss_seg_bw_var"] +
            losses["loss_seg_track"]
        )
        track_total = (
            losses["loss_cls"]  +
            losses["loss_bbox"] +
            losses["loss_giou"]
        )
        losses["loss_total"] = (
            self.lambda_seg   * seg_total +
            self.lambda_track * track_total
        )
        return losses


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_hybrid_criterion(cfg: dict) -> HybridCriterion:
    matcher = HungarianMatcher(
        cost_class=cfg.get("set_cost_class", 1.0),
        cost_bbox=cfg.get("set_cost_bbox",   5.0),
        cost_giou=cfg.get("set_cost_giou",   2.0),
        cost_mask=cfg.get("set_cost_mask",   0.0),   # seg branch handles masks
        focal_alpha=cfg.get("focal_alpha",   0.25),
        focal_gamma=cfg.get("focal_gamma",   2.0),
    )
    loss_weights = {
        "loss_cls":  cfg.get("cls_loss_coef",  4.0),
        "loss_bbox": cfg.get("bbox_loss_coef", 5.0),
        "loss_giou": cfg.get("giou_loss_coef", 2.0),
    }
    seg_weights = {
        "w_seg":           cfg.get("seg_loss_coef",   1.0),
        "w_track_offset":  cfg.get("track_offset_coef", 1.0),
    }
    return HybridCriterion(
        matcher=matcher,
        loss_weights=loss_weights,
        seg_weights=seg_weights,
        lambda_seg=cfg.get("lambda_seg",   1.0),
        lambda_track=cfg.get("lambda_track", 1.0),
        lambda_aux=cfg.get("lambda_aux",   0.5),
        num_classes=cfg.get("num_classes", 1),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )
