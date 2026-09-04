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
    valid_mask: Optional[Tensor] = None,
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
    if valid_mask is not None:
        loss = loss * valid_mask.to(dtype=loss.dtype)
    if reduction == "sum":
        return loss.sum() / num_items
    elif reduction == "mean":
        return loss.mean()
    return loss


def dice_loss(
    inputs: Tensor, targets: Tensor, num_items: float,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    inputs:  (N, L) logits (sigmoid applied here)
    targets: (N, L) float 0/1
    """
    prob = inputs.sigmoid()
    if valid_mask is not None:
        valid_mask = valid_mask.to(dtype=prob.dtype)
        prob = prob * valid_mask
        targets = targets * valid_mask
    numerator = 2 * (prob * targets).sum(-1)
    denominator = prob.sum(-1) + targets.sum(-1)
    loss = 1 - numerator / denominator.clamp(min=1e-6)
    return loss.sum() / num_items


# ---------------------------------------------------------------------------
# Point-based mask sampling (Mask2Former / PointRend)
# ---------------------------------------------------------------------------

def point_sample(inp: Tensor, point_coords: Tensor, **kwargs) -> Tensor:
    """
    Sample `inp` at continuous normalised coordinates.

    inp          : (N, C, H, W)
    point_coords : (N, P, 2) in [0, 1]
    returns      : (N, C, P)
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)                 # (N, P, 1, 2)
    out = F.grid_sample(
        inp, 2.0 * point_coords - 1.0, align_corners=False, **kwargs
    )                                                            # (N, C, P, 1)
    if add_dim:
        out = out.squeeze(3)                                     # (N, C, P)
    return out


def _mask_point_uncertainty(logits: Tensor) -> Tensor:
    """Most uncertain where the logit is closest to zero. logits: (N, 1, P)."""
    return -logits.abs()


@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    coarse_logits: Tensor,
    num_points: int,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
) -> Tensor:
    """
    Mask2Former point sampling: oversample uniformly, then keep the points where
    the coarse prediction is most uncertain, plus a uniform-random remainder.

    coarse_logits : (N, 1, H, W)
    returns       : (N, num_points, 2) in [0, 1]
    """
    N = coarse_logits.shape[0]
    device = coarse_logits.device
    n_sampled = max(int(num_points * oversample_ratio), num_points)
    coords = torch.rand(N, n_sampled, 2, device=device)
    logits = point_sample(coarse_logits, coords)                  # (N, 1, n_sampled)
    unc = _mask_point_uncertainty(logits)[:, 0, :]                # (N, n_sampled)

    n_uncertain = int(importance_sample_ratio * num_points)
    n_random = num_points - n_uncertain
    idx = torch.topk(unc, k=n_uncertain, dim=1)[1]                # (N, n_uncertain)
    shift = (n_sampled * torch.arange(N, device=device))[:, None]
    coords = coords.view(-1, 2)[(idx + shift).view(-1)].view(N, n_uncertain, 2)
    if n_random > 0:
        coords = torch.cat(
            [coords, torch.rand(N, n_random, 2, device=device)], dim=1
        )
    return coords


@torch.no_grad()
def sample_points_in_target_bbox(tgt_masks: Tensor, num_points: int, pad: float = 0.5) -> Tensor:
    """
    Uniformly sample points inside each target's (padded) bounding box.

    For small objects the uniform / uncertainty samplers put almost no points on
    the foreground at cold start, so the dice signal is weak. Concentrating a
    share of the points on the GT region fixes that regardless of prediction
    state.

    tgt_masks : (K, 1, H, W) float 0/1
    returns   : (K, num_points, 2) in [0, 1]
    """
    K, _, H, W = tgt_masks.shape
    device = tgt_masks.device
    out = torch.rand(K, num_points, 2, device=device)
    m = tgt_masks[:, 0] > 0.5
    ys = torch.arange(H, device=device).float() / max(H - 1, 1)
    xs = torch.arange(W, device=device).float() / max(W - 1, 1)
    for k in range(K):
        rows = m[k].any(1).nonzero()
        cols = m[k].any(0).nonzero()
        if len(rows) == 0 or len(cols) == 0:
            continue
        y0, y1 = ys[rows[0, 0]], ys[rows[-1, 0]]
        x0, x1 = xs[cols[0, 0]], xs[cols[-1, 0]]
        hy, hx = (y1 - y0) * pad, (x1 - x0) * pad
        y0, y1 = (y0 - hy).clamp(0, 1), (y1 + hy).clamp(0, 1)
        x0, x1 = (x0 - hx).clamp(0, 1), (x1 + hx).clamp(0, 1)
        out[k, :, 0] = x0 + (x1 - x0) * out[k, :, 0]
        out[k, :, 1] = y0 + (y1 - y0) * out[k, :, 1]
    return out


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
        mask_focal_alpha: float = 0.75,
        mask_num_points: int = 12544,
        mask_oversample_ratio: float = 3.0,
        mask_importance_sample_ratio: float = 0.75,
        mask_gt_bbox_fraction: float = 0.5,
        mask_loss_type: str = "bce",
    ):
        super().__init__()
        self.matcher = matcher
        self.loss_weights = loss_weights
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mask_focal_alpha = mask_focal_alpha
        self.mask_num_points = mask_num_points
        self.mask_oversample_ratio = mask_oversample_ratio
        self.mask_importance_sample_ratio = mask_importance_sample_ratio
        self.mask_gt_bbox_fraction = mask_gt_bbox_fraction
        self.mask_loss_type = mask_loss_type
        # Frozen matching: once detection has converged the training loop can
        # snapshot the query->GT assignment ("capture") and then reuse it
        # ("frozen") so the mask head trains each query toward a fixed cell.
        # Without this the Hungarian assignment keeps permuting while masks are
        # learned, so every query is trained toward the *mean* cell and the
        # queries / masks collapse to one central blob.
        self.matching_mode = "hungarian"   # "hungarian" | "capture" | "frozen"
        self.frozen_indices: Dict[int, Tuple[Tensor, Tensor]] = {}
        self.dn_aux = True
        # Toggled per-epoch by the training loop: keep mask terms out of the
        # backward pass until the matcher (box/class driven) has stabilised.
        self.mask_enabled = True

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

        # One-hot focal target. Default = background (all zeros => channel-0 cell
        # target is 0). ONLY matched queries are marked as cells (channel-0 = 1).
        # Unmatched queries stay background so no-object supervision drives their
        # cell score down. Previously tgt_cls was zero-initialised and the cell
        # label is also 0, so EVERY query (matched or not) got cell-target=1,
        # collapsing the classification head to "always cell" regardless of
        # num_queries.
        tgt_one_hot = torch.zeros_like(logits)
        query_valid = torch.ones_like(logits)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx):
                tgt_one_hot[b, src_idx, 0] = 1.0  # matched cell only

            # An unmatched query centred inside an IGNORE region is unknown,
            # not background. Matched queries remain supervised regardless.
            if "ignore_mask" in targets[b] and "pred_boxes" in outputs:
                ignore = targets[b]["ignore_mask"].bool()
                h, w = ignore.shape[-2:]
                centres = outputs["pred_boxes"][b, :, :2].detach().clamp(0, 1)
                xs = (centres[:, 0] * max(w - 1, 0)).round().long()
                ys = (centres[:, 1] * max(h - 1, 0)).round().long()
                ignored_queries = ignore[ys, xs]
                if len(src_idx):
                    ignored_queries[src_idx] = False
                query_valid[b, ignored_queries] = 0

        loss = sigmoid_focal_loss(logits, tgt_one_hot, num_cells,
                                  self.focal_alpha, self.focal_gamma,
                                  valid_mask=query_valid)
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
        pred_masks = outputs["pred_masks"]    # (B, N, H_m, W_m) coarse logits

        src_list, tgt_list, ign_list = [], [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            if "mask_valid" in targets[b]:
                # Hungarian indices are CPU tensors; keep the selector on the
                # same device, then use the filtered indices on CUDA tensors.
                keep = targets[b]["mask_valid"][tgt_idx.to(targets[b]["mask_valid"].device)].bool().cpu()
                src_idx = src_idx[keep]
                tgt_idx = tgt_idx[keep]
            if len(src_idx) == 0:
                continue

            src_list.append(pred_masks[b, src_idx])                       # (k, H_m, W_m) logits
            tgt_list.append(targets[b]["masks"][tgt_idx].float())         # (k, H_t, W_t) 0/1
            if "ignore_mask" in targets[b]:
                ign = targets[b]["ignore_mask"].float()[None, None].expand(len(src_idx), 1, -1, -1)
                ign_list.append(ign)
            else:
                ign_list.append(None)

        if not src_list:
            device = pred_masks.device
            return {"loss_mask_focal": torch.zeros((), device=device),
                    "loss_mask_dice":  torch.zeros((), device=device),
                    "stat_mask_iou":   torch.zeros((), device=device),
                    "stat_mask_pred_fg": torch.zeros((), device=device),
                    "stat_mask_matched": torch.zeros((), device=device)}

        src_masks = torch.cat(src_list, 0).unsqueeze(1)                   # (K, 1, H_m, W_m)
        tgt_masks = torch.cat(tgt_list, 0).unsqueeze(1)                   # (K, 1, H_t, W_t)
        has_ignore = all(i is not None for i in ign_list)
        ignore_masks = torch.cat(ign_list, 0) if has_ignore else None    # (K, 1, H_t, W_t)

        # Point sampling: identical continuous coords for prediction and target,
        # so coarse (H/4) predictions and full-res targets stay aligned and the
        # per-point gradient is normalised by ~num_points instead of ~H*W.
        with torch.no_grad():
            n_bbox = int(self.mask_gt_bbox_fraction * self.mask_num_points)
            n_unc = self.mask_num_points - n_bbox
            parts = []
            if n_unc > 0:
                parts.append(get_uncertain_point_coords_with_randomness(
                    src_masks, n_unc,
                    self.mask_oversample_ratio, self.mask_importance_sample_ratio,
                ))
            if n_bbox > 0:
                parts.append(sample_points_in_target_bbox(tgt_masks, n_bbox))
            point_coords = torch.cat(parts, dim=1)                       # (K, P, 2)
            tgt_pts = point_sample(tgt_masks, point_coords).squeeze(1)   # (K, P)
            if ignore_masks is not None:
                valid_pts = (point_sample(ignore_masks, point_coords).squeeze(1) < 0.5)
            else:
                valid_pts = None
        src_pts = point_sample(src_masks, point_coords).squeeze(1)       # (K, P) with grad

        # Per-point classification term. Plain BCE (mean over points) rather than
        # focal: the point set is deliberately foreground-enriched (half the
        # points fall in the GT bbox), so it is already roughly balanced, and
        # BCE anchors the absolute mask probability. Dice alone is scale-free and
        # oscillates between all-foreground and a saturated all-background death.
        if self.mask_loss_type == "focal":
            ce_map = sigmoid_focal_loss(
                src_pts, tgt_pts, num_cells, self.mask_focal_alpha,
                self.focal_gamma, reduction="none", valid_mask=valid_pts,
            )
        else:
            ce_map = F.binary_cross_entropy_with_logits(
                src_pts, tgt_pts, reduction="none",
            )
            if valid_pts is not None:
                ce_map = ce_map * valid_pts.to(ce_map.dtype)
        if valid_pts is not None:
            denom = valid_pts.to(ce_map.dtype).sum(-1).clamp(min=1)
            loss_focal = (ce_map.sum(-1) / denom).sum() / num_cells
        else:
            loss_focal = ce_map.mean(1).sum() / num_cells
        loss_dice = dice_loss(src_pts, tgt_pts, num_cells, valid_mask=valid_pts)

        if not self.mask_enabled:
            loss_focal = loss_focal * 0.0
            loss_dice = loss_dice * 0.0

        with torch.no_grad():
            # Full-resolution mask IoU — the real L0 gate signal (the point
            # sample set is deliberately foreground-weighted so its IoU is not
            # comparable to a whole-image score).
            src_full = F.interpolate(
                src_masks, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False,
            )
            pf = src_full > 0                              # logit>0 <=> prob>0.5
            tf = tgt_masks > 0.5
            if ignore_masks is not None:
                keep = ignore_masks < 0.5
                pf = pf & keep
                tf = tf & keep
            inter = (pf & tf).flatten(1).sum(1).float()
            union = (pf | tf).flatten(1).sum(1).float().clamp(min=1)
            stats = {
                "stat_mask_iou": (inter / union).mean(),
                "stat_mask_pred_fg": (src_full > 0).float().mean(),
                "stat_mask_matched": torch.tensor(float(src_pts.shape[0]), device=src_pts.device),
            }
        return {"loss_mask_focal": loss_focal, "loss_mask_dice": loss_dice, **stats}

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
        if self.matching_mode == "frozen" and all(
            int(t["image_id"]) in self.frozen_indices for t in targets
        ):
            indices = [
                tuple(x.to(outputs["pred_logits"].device) for x in
                      self.frozen_indices[int(t["image_id"])])
                for t in targets
            ]
        else:
            indices = self.matcher(outputs, targets)
            if self.matching_mode == "capture" and self.training:
                for t, idx in zip(targets, indices):
                    self.frozen_indices[int(t["image_id"])] = (
                        idx[0].detach().cpu().clone(), idx[1].detach().cpu().clone(),
                    )
        losses.update(self.loss_labels(outputs, targets, indices, num_cells))
        losses.update(self.loss_boxes(outputs, targets, indices, num_cells))
        losses.update(self.loss_masks(outputs, targets, indices, num_cells))

        # --- Query denoising (fixed assignment, no matcher) ---
        if "dn_meta" in outputs and outputs["dn_meta"] is not None:
            dn = outputs["dn_meta"]
            gt_idx = dn["gt_idx"].detach().cpu()
            src_idx = torch.arange(dn["num_dn"])
            dn_indices = [(src_idx, gt_idx)]
            dn_out = {
                "pred_logits": outputs["dn_pred_logits"],
                "pred_boxes": outputs["dn_pred_boxes"],
            }
            if "dn_pred_masks" in outputs:
                dn_out["pred_masks"] = outputs["dn_pred_masks"]
            n_dn = max(dn["num_dn"], 1)
            for k, v in self.loss_labels(dn_out, targets, dn_indices, n_dn).items():
                losses[f"{k}_dn"] = v
            for k, v in self.loss_boxes(dn_out, targets, dn_indices, n_dn).items():
                losses[f"{k}_dn"] = v
            for k, v in self.loss_masks(dn_out, targets, dn_indices, n_dn).items():
                losses[f"{k}_dn"] = v
            # DN loss at every decoder layer (deep supervision)
            if self.dn_aux and "dn_pred_logits_aux" in outputs:
                for li in range(outputs["dn_pred_logits_aux"].shape[0]):
                    dn_aux = {
                        "pred_logits": outputs["dn_pred_logits_aux"][li],
                        "pred_boxes": outputs["dn_pred_boxes_aux"][li],
                    }
                    for k, v in self.loss_labels(dn_aux, targets, dn_indices, n_dn).items():
                        losses[f"{k}_dn_aux{li}"] = v
                    for k, v in self.loss_boxes(dn_aux, targets, dn_indices, n_dn).items():
                        losses[f"{k}_dn_aux{li}"] = v

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
            # Diagnostic-only quantities: logged, never added to the total.
            if k.startswith("stat_"):
                loss_log[k] = v.detach()
                continue
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
        cost_mask_points=cfg.get("cost_mask_points", 3000),
    )

    loss_weights = {
        "loss_cls":        cfg.get("cls_loss_coef", 4.0),
        "loss_bbox":       cfg.get("bbox_loss_coef", 5.0),
        "loss_giou":       cfg.get("giou_loss_coef", 2.0),
        "loss_mask_focal": cfg.get("mask_loss_coef", 5.0),
        "loss_mask_dice":  cfg.get("dice_loss_coef", 5.0),
    }

    crit = SetCriterion(
        matcher=matcher,
        loss_weights=loss_weights,
        num_classes=cfg.get("num_classes", 1),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        mask_focal_alpha=cfg.get("mask_focal_alpha", 0.75),
        mask_num_points=cfg.get("mask_num_points", 12544),
        mask_oversample_ratio=cfg.get("mask_oversample_ratio", 3.0),
        mask_importance_sample_ratio=cfg.get("mask_importance_sample_ratio", 0.75),
        mask_gt_bbox_fraction=cfg.get("mask_gt_bbox_fraction", 0.5),
        mask_loss_type=cfg.get("mask_loss_type", "bce"),
    )
    crit.dn_aux = cfg.get("dn_aux", True)
    return crit
