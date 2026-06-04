"""
CTC/COCO dataset loader for BSGM-CellTrack.

Uses COCO-format JSON annotations (created by scripts/create_coco_from_ctc.py).
Returns triplets of consecutive frames [t-1, t, t+1] for temporal Mamba fusion.

Target format returned per sample:
    images:       list of 3 tensors (C_in, H, W) — [prev, curr, next]
    targets:      dict for the current frame (t):
        'labels'  : (M,) long
        'boxes'   : (M, 4) float — normalised cxcywh
        'masks'   : (M, H, W) bool
        'track_ids': (M,) long
        'image_id': int
        'orig_size': (H, W)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


# ---------------------------------------------------------------------------
# COCO-format annotation reader
# ---------------------------------------------------------------------------

class CocoAnnotations:
    """Thin wrapper around a COCO JSON annotation file."""

    def __init__(self, ann_file: str):
        with open(ann_file) as f:
            data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations: Dict[int, List] = {}  # image_id → list of anns
        for ann in data.get("annotations", []):
            self.annotations.setdefault(ann["image_id"], []).append(ann)
        self.image_ids = sorted(self.images.keys())

    def get_anns(self, image_id: int) -> List[dict]:
        return self.annotations.get(image_id, [])

    def decode_seg(self, seg, h: int, w: int) -> np.ndarray:
        """Decode COCO segmentation (polygon or RLE) to binary mask (H, W)."""
        if HAS_COCO:
            if isinstance(seg, list):
                rle = coco_mask.frPyObjects(seg, h, w)
                m = coco_mask.decode(coco_mask.merge(rle))
            elif isinstance(seg, dict):
                if isinstance(seg.get("counts"), list):
                    rle = coco_mask.frPyObjects(seg, h, w)
                    m = coco_mask.decode(rle)
                else:
                    m = coco_mask.decode(seg)
            else:
                m = np.zeros((h, w), dtype=np.uint8)
        else:
            # Fallback: no pycocotools — fill bounding box
            m = np.zeros((h, w), dtype=np.uint8)
        return m.astype(bool)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CTCCocoDataset(Dataset):
    """
    COCO-format CTC dataset returning frame triplets for temporal fusion.

    Parameters
    ----------
    img_dir   : Root directory for images (e.g., data/ctchuh7/CTC/train/01/).
    ann_file  : Path to COCO JSON annotations.
    transforms: Callable applied to (image_list, target) and returning same.
    target_size: (H, W) to resize images to. None = keep original.
    in_channels: 1 or 3 (grayscale replicated to 3 channels if needed).
    temporal_window: number of frames per sample (default 3: prev, curr, next).
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transforms=None,
        target_size: Optional[Tuple[int, int]] = (512, 512),
        in_channels: int = 3,
        temporal_window: int = 3,
        augment: bool = True,
    ):
        self.img_dir = img_dir
        self.coco = CocoAnnotations(ann_file)
        self.transforms = transforms
        self.target_size = target_size
        self.in_channels = in_channels
        self.temporal_window = temporal_window
        self.augment = augment
        self.ids = self.coco.image_ids
        # Clip endpoints so we always have a full window
        half = temporal_window // 2
        self.ids = self.ids[half: len(self.ids) - half]

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_id: int) -> np.ndarray:
        img_info = self.coco.images[image_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = np.array(Image.open(img_path))
        if img.ndim == 2:
            img = img[:, :, None]  # (H, W, 1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def _load_target(self, image_id: int, H: int, W: int) -> Dict:
        anns = self.coco.get_anns(image_id)
        labels, boxes, masks, track_ids = [], [], [], []
        for ann in anns:
            labels.append(0)  # all cells are class 0
            # COCO bbox: [x, y, w, h] → cxcywh normalised
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            boxes.append([cx, cy, bw / W, bh / H])
            # Segmentation mask
            if "segmentation" in ann and ann["segmentation"]:
                m = self.coco.decode_seg(ann["segmentation"], H, W)
            else:
                m = np.zeros((H, W), dtype=bool)
                x0, y0 = int(x), int(y)
                x1, y1 = min(W, int(x + bw)), min(H, int(y + bh))
                m[y0:y1, x0:x1] = True
            masks.append(m)
            track_ids.append(ann.get("track_id", 0))

        return {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "masks": torch.from_numpy(np.stack(masks, 0)) if masks else torch.zeros(0, H, W, dtype=torch.bool),
            "track_ids": torch.tensor(track_ids, dtype=torch.long),
            "image_id": image_id,
            "orig_size": (H, W),
        }

    def _preprocess_image(self, img: np.ndarray) -> Tensor:
        """numpy (H, W, C) → float tensor (C_in, H_out, W_out)."""
        # Normalise to [0, 1]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / (img.max() + 1e-8)

        # To tensor (C, H, W)
        t = torch.from_numpy(img.transpose(2, 0, 1))

        # Replicate to `in_channels`
        if t.shape[0] == 1 and self.in_channels == 3:
            t = t.expand(3, -1, -1)
        elif t.shape[0] == 3 and self.in_channels == 1:
            t = t.mean(0, keepdim=True)

        # Resize
        if self.target_size is not None:
            t = F.interpolate(
                t.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
            ).squeeze(0)

        return t

    def _resize_target(self, target: Dict, new_hw: Tuple[int, int]) -> Dict:
        """Rescale masks and boxes to new spatial size."""
        H_new, W_new = new_hw
        H_orig, W_orig = target["orig_size"]
        if H_orig == H_new and W_orig == W_new:
            return target
        # Boxes: just update normalisation (already normalised, no change needed)
        # Masks: resize
        if target["masks"].shape[0] > 0:
            masks_float = target["masks"].float().unsqueeze(1)  # (M, 1, H, W)
            masks_resized = F.interpolate(
                masks_float, size=(H_new, W_new), mode="nearest"
            ).squeeze(1).bool()
            target = dict(target)
            target["masks"] = masks_resized
            target["orig_size"] = (H_new, W_new)
        return target

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], Dict]:
        curr_id = self.ids[idx]
        half = self.temporal_window // 2
        curr_pos = self.coco.image_ids.index(curr_id)

        # Gather window of frame ids (clamp at boundaries)
        frame_ids = []
        for offset in range(-half, half + 1):
            pos = max(0, min(len(self.coco.image_ids) - 1, curr_pos + offset))
            frame_ids.append(self.coco.image_ids[pos])

        # Load and preprocess each frame
        images = []
        for fid in frame_ids:
            raw = self._load_image(fid)
            img_t = self._preprocess_image(raw)
            images.append(img_t)

        # Load target for current (middle) frame
        curr_raw = self._load_image(curr_id)
        H_orig, W_orig = curr_raw.shape[:2]
        target = self._load_target(curr_id, H_orig, W_orig)

        if self.target_size is not None:
            target = self._resize_target(target, self.target_size)

        if self.transforms is not None:
            images, target = self.transforms(images, target)

        return images, target


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Collate a list of (images_list, target) into batched tensors.
    Pads masks to the maximum number of instances in the batch.
    """
    images_batch = []
    targets_batch = []
    T = len(batch[0][0])

    for t_idx in range(T):
        frame_tensors = torch.stack([sample[0][t_idx] for sample in batch], dim=0)
        images_batch.append(frame_tensors)

    for sample in batch:
        targets_batch.append(sample[1])

    return images_batch, targets_batch


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, split: str = "train") -> CTCCocoDataset:
    data_dir = cfg.get("data_dir", "data")
    dataset_name = cfg.get("dataset", "ctchuh7")
    ann_file = os.path.join(data_dir, dataset_name, "COCO", "annotations", f"instances_{split}.json")
    img_dir = os.path.join(data_dir, dataset_name, "CTC", split)

    target_size = cfg.get("target_size", (512, 512))
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    return CTCCocoDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        target_size=tuple(target_size) if target_size else None,
        in_channels=cfg.get("backbone_in_channels", 3),
        temporal_window=3,
        augment=(split == "train"),
    )
