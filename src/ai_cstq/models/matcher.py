"""
Hungarian bipartite matcher for BSGM-CellTrack.

Matches model predictions to ground-truth annotations by minimising the
total cost across class + box + mask terms.

Adapted from DETR / Deformable-DETR; supports:
  - Division cells (8D bounding boxes)
  - Track queries (pre-matched, pass-through)
  - Focal loss cost for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Compute optimal bipartite matching between predictions and targets.

    Parameters
    ----------
    cost_class : float
        Weight for classification cost.
    cost_bbox : float
        Weight for L1 bounding-box cost.
    cost_giou : float
        Weight for GIoU cost.
    cost_mask : float
        Weight for mask cost (if masks provided).
    focal_alpha : float
        Alpha for focal classification cost.
    focal_gamma : float
        Gamma for focal classification cost.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_mask: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cost_mask_points: int = 3000,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cost_mask_points = cost_mask_points

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        Parameters
        ----------
        outputs : dict with keys:
            'pred_logits': (B, N, num_classes+1)
            'pred_boxes':  (B, N, 4 or 8)
            'pred_masks':  (B, N, H_m, W_m)  optional
        targets : list of B dicts, each with:
            'labels':   (num_gt,)
            'boxes':    (num_gt, 4 or 8)
            'masks':    (num_gt, H, W)    optional

        Returns
        -------
        list of B tuples (row_ind, col_ind) — matched (pred_idx, gt_idx) pairs
        """
        B, N, _ = outputs["pred_logits"].shape
        indices = []

        for b in range(B):
            logits = outputs["pred_logits"][b]   # (N, C+1)
            boxes = outputs["pred_boxes"][b]     # (N, 4 or 8)
            tgt_labels = targets[b]["labels"]    # (M,)
            tgt_boxes = targets[b]["boxes"]      # (M, 4 or 8)
            M = len(tgt_labels)

            if M == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                 torch.tensor([], dtype=torch.long)))
                continue

            # Classification cost (focal variant)
            probs = logits.sigmoid()             # (N, C+1)
            # Use first class probability (cell=0)
            prob_pos = probs[:, :1].expand(-1, M)  # (N, M)
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost = -(1 - alpha) * (prob_pos ** gamma) * torch.log(1 - prob_pos + 1e-8)
            pos_cost = -alpha * ((1 - prob_pos) ** gamma) * torch.log(prob_pos + 1e-8)
            cost_class = pos_cost - neg_cost    # (N, M)

            # Box cost: L1 over normalised coords (cx, cy, w, h)
            boxes_4d = boxes[:, :4]             # (N, 4)
            tgt_boxes_4d = tgt_boxes[:, :4]     # (M, 4)
            cost_bbox = torch.cdist(boxes_4d, tgt_boxes_4d, p=1)  # (N, M)

            # GIoU cost
            cost_giou = -generalised_iou(
                box_cxcywh_to_xyxy(boxes_4d),
                box_cxcywh_to_xyxy(tgt_boxes_4d),
            )  # (N, M)

            # Mask cost — sample the same uniform-random point set for every
            # prediction and target, matching how loss_masks supervises masks
            # (Mask2Former). Avoids the resolution mismatch of a full downsample.
            cost_mask = 0.0
            if "pred_masks" in outputs and "masks" in targets[b]:
                pred_masks = outputs["pred_masks"][b].unsqueeze(1)          # (N, 1, H_m, W_m)
                tgt_masks = targets[b]["masks"].float().unsqueeze(1)        # (M, 1, H_t, W_t)
                P = self.cost_mask_points
                pc = torch.rand(1, P, 2, device=pred_masks.device)
                pred_pts = _point_sample(
                    pred_masks, pc.expand(pred_masks.shape[0], -1, -1)
                ).squeeze(1)                                               # (N, P)
                tgt_pts = _point_sample(
                    tgt_masks, pc.expand(tgt_masks.shape[0], -1, -1)
                ).squeeze(1)                                               # (M, P)
                valid_flat = None
                if "ignore_mask" in targets[b]:
                    ign = targets[b]["ignore_mask"].float()[None, None]
                    valid_flat = (_point_sample(ign, pc).squeeze() < 0.5)  # (P,)
                cost_mask = batch_sigmoid_focal_cost(
                    pred_pts, tgt_pts, valid_mask=valid_flat,
                )  # (N, M)
                # Invalid/empty instance masks must not influence assignment;
                # their class and box annotations remain fully supervised.
                if "mask_valid" in targets[b]:
                    invalid = ~targets[b]["mask_valid"].bool()
                    cost_mask[:, invalid] = 0
                cost_mask = self.cost_mask * cost_mask

            # Total cost
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
                + cost_mask
            )
            C = C.detach().cpu().numpy()

            row, col = linear_sum_assignment(C)
            indices.append((
                torch.tensor(row, dtype=torch.long),
                torch.tensor(col, dtype=torch.long),
            ))

        return indices


# ---------------------------------------------------------------------------
# Box utility functions
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def generalised_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute generalised IoU between all pairs of boxes.

    Parameters
    ----------
    boxes1 : (N, 4)  xyxy format
    boxes2 : (M, 4)  xyxy format

    Returns
    -------
    giou : (N, M)
    """
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou  # (N, M)


def _point_sample(inp: Tensor, point_coords: Tensor, **kwargs) -> Tensor:
    """Sample inp (N,C,H,W) at point_coords (N,P,2) in [0,1] -> (N,C,P)."""
    add_dim = point_coords.dim() == 3
    if add_dim:
        point_coords = point_coords.unsqueeze(2)
    out = F.grid_sample(inp, 2.0 * point_coords - 1.0, align_corners=False, **kwargs)
    return out.squeeze(3) if add_dim else out


def batch_sigmoid_focal_cost(
    pred: Tensor, tgt: Tensor, valid_mask: Tensor = None,
) -> Tensor:
    """
    Sigmoid focal cost between pred (N, L) and tgt (M, L).
    Returns (N, M).
    """
    prob = pred.sigmoid().unsqueeze(1)    # (N, 1, L)
    tgt_e = tgt.unsqueeze(0)             # (1, M, L)
    if valid_mask is None:
        valid = 1.0
        denom = pred.shape[-1]
    else:
        valid = valid_mask.to(dtype=pred.dtype).view(1, 1, -1)
        denom = valid.sum().clamp(min=1.0)
    pos_cost = -(tgt_e * torch.log(prob + 1e-8) * valid).sum(-1) / denom
    neg_cost = -((1 - tgt_e) * torch.log(1 - prob + 1e-8) * valid).sum(-1) / denom
    return pos_cost + neg_cost           # (N, M)
