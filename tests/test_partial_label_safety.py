import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ai_cstq.datasets.ctc_coco import CTCCocoDataset
from ai_cstq.models.criterion import SetCriterion


class _UnusedMatcher:
    pass


def _criterion():
    return SetCriterion(_UnusedMatcher(), {})


def test_temporal_windows_never_cross_sequences(tmp_path: Path):
    images = []
    for seq in ("01", "02"):
        for frame in range(2):
            name = f"CTC_{seq}_frame_{frame:03d}.tif"
            pixels = np.full((4, 4), int(seq), np.uint8)
            pixels[0, 0] = 255
            Image.fromarray(pixels).save(tmp_path / name)
            images.append({
                "id": len(images), "file_name": name, "width": 4, "height": 4,
                "man_track_id": seq, "frame_id": 0,
            })
    ann = tmp_path / "anno.json"
    ann.write_text(json.dumps({"images": images, "annotations": []}))
    dataset = CTCCocoDataset(str(tmp_path), str(ann), target_size=None, in_channels=1)

    triplet, target = dataset[1]
    assert target["sequence_id"] == "01"
    assert all(torch.isclose(frame[0, 0, 1], torch.tensor(1 / 255)) for frame in triplet)
    triplet, target = dataset[2]
    assert target["sequence_id"] == "02"
    assert all(torch.isclose(frame[0, 0, 1], torch.tensor(2 / 255)) for frame in triplet)


def test_mask_losses_are_invariant_inside_ignore_region():
    criterion = _criterion()
    target = {
        "masks": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool),
        "ignore_mask": torch.tensor([[False, True], [False, False]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    base = torch.zeros(1, 1, 2, 2)
    changed_inside = base.clone()
    changed_inside[0, 0, 0, 1] = 20
    changed_outside = base.clone()
    changed_outside[0, 0, 1, 1] = 20

    a = criterion.loss_masks({"pred_masks": base}, [target], indices, 1)
    b = criterion.loss_masks({"pred_masks": changed_inside}, [target], indices, 1)
    c = criterion.loss_masks({"pred_masks": changed_outside}, [target], indices, 1)
    assert torch.allclose(a["loss_mask_focal"], b["loss_mask_focal"])
    assert torch.allclose(a["loss_mask_dice"], b["loss_mask_dice"])
    assert not torch.allclose(a["loss_mask_focal"], c["loss_mask_focal"])


def test_unmatched_classification_is_suppressed_inside_ignore_region():
    criterion = _criterion()
    outputs = {
        "pred_logits": torch.tensor([[[0.0], [10.0]]]),
        "pred_boxes": torch.tensor([[[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]]]),
    }
    target = {"ignore_mask": torch.tensor([[False, False], [False, True]])}
    empty = [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]
    ignored = criterion.loss_labels(outputs, [target], empty, 1)["loss_cls"]
    outputs["pred_boxes"][0, 1, :2] = torch.tensor([0.25, 0.25])
    supervised = criterion.loss_labels(outputs, [target], empty, 1)["loss_cls"]
    assert supervised > ignored
