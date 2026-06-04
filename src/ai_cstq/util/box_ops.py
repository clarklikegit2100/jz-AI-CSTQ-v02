"""Box operation utilities."""
import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], -1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def generalised_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU between all pairs. boxes in xyxy format."""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(1e-6)
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_area = (enc_rb - enc_lt).clamp(0).prod(-1)
    return iou - (enc_area - union) / enc_area.clamp(1e-6)


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Convert binary masks (N, H, W) → bounding boxes (N, 4) in cxcywh."""
    N, H, W = masks.shape
    boxes = torch.zeros(N, 4, device=masks.device, dtype=torch.float32)
    for i, mask in enumerate(masks):
        y, x = mask.nonzero(as_tuple=True)
        if len(y):
            x0, x1 = x.min().float(), x.max().float()
            y0, y1 = y.min().float(), y.max().float()
            cx = (x0 + x1) / 2 / W
            cy = (y0 + y1) / 2 / H
            w = (x1 - x0 + 1) / W
            h = (y1 - y0 + 1) / H
            boxes[i] = torch.tensor([cx, cy, w, h])
    return boxes
