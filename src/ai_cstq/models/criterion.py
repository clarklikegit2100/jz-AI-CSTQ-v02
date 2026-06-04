"""
SetCriterion for BSGM-CellTrack.

Computes the training loss over all decoder layers (auxiliary losses)
and the encoder proposals (two-stage).

Loss terms:
  - Sigmoid focal classification loss
  - L1 bounding-box regression loss
  - GIoU bounding-box loss
  - Sigmoid focal mask loss
  - Dice mask loss
  - Division loss (extra weight for division targets)
  - DN-Track denoised tracking auxiliary loss (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalised_iou


# ---------------------------------------------------------------------------
# Focal loss helpers
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_items: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> Tensor:
    """
    inputs:  (B, N, C) raw logits
    targets: (B, N, C) float 0/1 targets
    """
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = focal_weight * ((1 - pt) ** gamma)
    loss = focal_weight * ce
    if reduction == "sum":
        return loss.sum() / num_items
    elif reduction == "mean":
        return loss.mean()
    return loss


def dice_loss(inputs: Tensor, targets: Tensor, num_items: float) -> Tensor:
    """
    inputs:  (N, L) logits (sigmoid applied here)
    targets: (N, L) float 0/1
    """
    prob = inputs.sigmoid()
    numerator = 2 * (prob * targets).sum(-1)
    denominator = prob.sum(-1) + targets.sum(-1)
    loss = 1 - numerator / denominator.clamp(min=1e-6)
    return loss.sum() / num_items


# ---------------------------------------------------------------------------
# SetCriterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """
    Joint detection + tracking + segmentation loss.

    Parameters
    ----------
    matcher : HungarianMatcher
    loss_weights : dict of loss name → weight
    num_classes : int
    focal_alpha, focal_gamma : focal loss hyperparameters
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        loss_weights: Dict[str, float],
        num_classes: int = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.loss_weights = loss_weights
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    # -------------------------------------------------------------------------
    # Individual loss terms
    # -------------------------------------------------------------------------

    def loss_labels(
        self,
        outputs: dict,
        targets: list,
        indices: list,
        num_cells: float,
    ) -> Dict[str, Tensor]:
        logits = outputs["pred_logits"]                    # (B, N, C+1)
        B, N, C1 = logits.shape

        # Build targets tensor
        tgt_cls = torch.zeros(B, N, dtype=torch.long, device=logits.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx):
                tgt_cls[b, src_idx] = targets[b]["labels"][tgt_idx]

        # One-hot for focal loss: positive class = 0 (cell), background = 1
        tgt_one_hot = torch.zeros_like(logits)
        tgt_one_hot[tgt_cls == 0, 0] = 1.0  # cell class

        loss = sigmoid_focal_loss(logits, tgt_one_hot, num_cells,
                                  self.focal_alpha, self.focal_gamma)
        return {"loss_cls": loss}

    def loss_boxes(
        self,
        outputs: dict,
        targets: list,
        indices: list,
        num_cells: float,
    ) -> Dict[str, Tensor]:
        src_boxes, tgt_boxes = self._gather_matched(outputs["pred_boxes"], targets, indices, "boxes")
        if src_boxes.shape[0] == 0:
            device = outputs["pred_boxes"].device
            return {"loss_bbox": torch.tensor(0.0, device=device),
                    "loss_giou": torch.tensor(0.0, device=device)}

        # L1 on first 4 dims (normalised cx, cy, w, h)
        loss_l1 = F.l1_loss(src_boxes[:, :4], tgt_boxes[:, :4], reduction="sum") / num_cells

        # GIoU
        giou = generalised_iou(
            box_cxcywh_to_xyxy(src_boxes[:, :4]),
            box_cxcywh_to_xyxy(tgt_boxes[:, :4]),
        )
        loss_giou = (1 - giou.diagonal()).sum() / num_cells

        return {"loss_bbox": loss_l1, "loss_giou": loss_giou}

    def loss_masks(
        self,
        outputs: dict,
        targets: list,
        indices: list,
        num_cells: float,
    ) -> Dict[str, Tensor]:
        if "pred_masks" not in outputs:
            return {}
        pred_masks = outputs["pred_masks"]    # (B, N, H_m, W_m)
        H_m, W_m = pred_masks.shape[-2:]

        src_masks_list, tgt_masks_list = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_masks_list.append(pred_masks[b, src_idx])             # (k, H_m, W_m)
            tgt_m = targets[b]["masks"][tgt_idx].float()              # (k, H_gt, W_gt)
            tgt_m = F.interpolate(
                tgt_m.unsqueeze(1), size=(H_m, W_m), mode="bilinear", align_corners=False
            ).squeeze(1)                                               # (k, H_m, W_m)
            tgt_masks_list.append(tgt_m)

        if not src_masks_list:
            device = pred_masks.device
            return {"loss_mask_focal": torch.tensor(0.0, device=device),
                    "loss_mask_dice":  torch.tensor(0.0, device=device)}

        src_masks = torch.cat(src_masks_list, 0).flatten(1)   # (K, H*W)
        tgt_masks = torch.cat(tgt_masks_list, 0).flatten(1)   # (K, H*W)

        loss_focal = sigmoid_focal_loss(src_masks, tgt_masks, num_cells, self.focal_alpha, self.focal_gamma)
        loss_dice = dice_loss(src_masks, tgt_masks, num_cells)
        return {"loss_mask_focal": loss_focal, "loss_mask_dice": loss_dice}

    # -------------------------------------------------------------------------
    # Gather matched predictions
    # -------------------------------------------------------------------------

    def _gather_matched(
        self,
        preds: Tensor,        # (B, N, D)
        targets: list,
        indices: list,
        key: str,
    ) -> Tuple[Tensor, Tensor]:
        src_list, tgt_list = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_list.append(preds[b, src_idx])
            tgt_list.append(targets[b][key][tgt_idx])
        if not src_list:
            dummy = preds.new_zeros(0, preds.shape[-1])
            return dummy, dummy
        return torch.cat(src_list, 0), torch.cat(tgt_list, 0)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, outputs: dict, targets: list) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        outputs : dict from BSGMCellTrack.forward()
        targets : list of B target dicts (from dataloader)

        Returns
        -------
        Dict of weighted scalar losses (all backprop-able).
        """
        losses: Dict[str, Tensor] = {}

        # Count matched cells for normalisation
        num_cells = sum(len(t["labels"]) for t in targets)
        num_cells = max(num_cells, 1)

        # --- Main layer ---
        indices = self.matcher(outputs, targets)
        losses.update(self.loss_labels(outputs, targets, indices, num_cells))
        losses.update(self.loss_boxes(outputs, targets, indices, num_cells))
        losses.update(self.loss_masks(outputs, targets, indices, num_cells))

        # --- Auxiliary layers ---
        if "pred_logits_aux" in outputs:
            for li in range(outputs["pred_logits_aux"].shape[0]):
                aux_out = {
                    "pred_logits": outputs["pred_logits_aux"][li],
                    "pred_boxes": outputs["pred_boxes_aux"][li],
                }
                idx_aux = self.matcher(aux_out, targets)
                for k, v in self.loss_labels(aux_out, targets, idx_aux, num_cells).items():
                    losses[f"{k}_aux{li}"] = v
                for k, v in self.loss_boxes(aux_out, targets, idx_aux, num_cells).items():
                    losses[f"{k}_aux{li}"] = v

        # --- Encoder two-stage proposal loss ---
        if "enc_outputs" in outputs:
            enc = outputs["enc_outputs"]
            enc_out = {
                "pred_logits": enc["pred_logits"],
                "pred_boxes": enc["pred_boxes"],
            }
            enc_indices = self.matcher(enc_out, targets)
            for k, v in self.loss_labels(enc_out, targets, enc_indices, num_cells).items():
                losses[f"{k}_enc"] = v
            for k, v in self.loss_boxes(enc_out, targets, enc_indices, num_cells).items():
                losses[f"{k}_enc"] = v

        # --- Apply weights and sum ---
        total = torch.tensor(0.0, device=outputs["pred_logits"].device)
        loss_log: Dict[str, Tensor] = {}
        for k, v in losses.items():
            # Match by prefix to find weight
            weight = 1.0
            for wk, wv in self.loss_weights.items():
                if k.startswith(wk):
                    weight = wv
                    break
            loss_log[k] = v.detach()
            total = total + weight * v

        loss_log["loss_total"] = total.detach()
        losses["loss_total"] = total
        return losses


# ---------------------------------------------------------------------------
# Build criterion
# ---------------------------------------------------------------------------

def build_criterion(cfg: dict) -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=cfg.get("set_cost_class", 1.0),
        cost_bbox=cfg.get("set_cost_bbox", 5.0),
        cost_giou=cfg.get("set_cost_giou", 2.0),
        cost_mask=cfg.get("set_cost_mask", 1.0),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )

    loss_weights = {
        "loss_cls":        cfg.get("cls_loss_coef", 4.0),
        "loss_bbox":       cfg.get("bbox_loss_coef", 5.0),
        "loss_giou":       cfg.get("giou_loss_coef", 2.0),
        "loss_mask_focal": cfg.get("mask_loss_coef", 5.0),
        "loss_mask_dice":  cfg.get("dice_loss_coef", 5.0),
    }

    return SetCriterion(
        matcher=matcher,
        loss_weights=loss_weights,
        num_classes=cfg.get("num_classes", 1),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )
