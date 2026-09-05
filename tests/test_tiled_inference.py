import numpy as np
import torch

from ai_cstq.util.tiled_inference import (
    compute_tile_origins,
    center_weight_map,
    assign_track_ids,
)


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------

def test_tile_origins_cover_frame_and_align_to_edge():
    H, W, ts, stride = 1024, 1024, 256, 192
    origins = compute_tile_origins(H, W, ts, stride)
    ys = sorted({y for y, _ in origins})
    xs = sorted({x for _, x in origins})
    assert ys[0] == 0 and ys[-1] == H - ts
    assert xs[0] == 0 and xs[-1] == W - ts
    # every tile stays inside the frame
    for y, x in origins:
        assert 0 <= y <= H - ts and 0 <= x <= W - ts
    # the union of tiles covers every pixel (adjacent origins overlap by
    # at least `ts - stride`, so there can be no gap)
    covered = np.zeros((H, W), dtype=bool)
    for y, x in origins:
        covered[y:y + ts, x:x + ts] = True
    assert covered.all()


def test_tile_origins_single_tile_when_frame_smaller():
    origins = compute_tile_origins(200, 200, 256, 192)
    assert origins == [(0, 0)]


def test_center_weight_map_shape_and_range():
    w = center_weight_map(256, 64)
    assert w.shape == (256, 256)
    assert w.max() == 1.0
    assert w.min() >= 0.05 * 0.05 - 1e-6
    # centre is fully weighted, corners are the most attenuated
    centre = w[128, 128]
    corner = w[0, 0]
    assert centre > corner


# ---------------------------------------------------------------------------
# Cross-frame identity
# ---------------------------------------------------------------------------

def _mask(y0, x0, size=20, canvas=64):
    m = torch.zeros(canvas, canvas, dtype=torch.bool)
    m[y0:y0 + size, x0:x0 + size] = True
    return m


def test_track_ids_stable_across_frames_for_same_cell():
    frame0 = [{"mask_bool": _mask(10, 10)}]
    frame1 = [{"mask_bool": _mask(11, 11)}]   # same cell, moved 1px
    ids = assign_track_ids([frame0, frame1], iou_thresh=0.3)
    assert ids[0] == ids[1]
    assert ids[0][0] >= 1


def test_track_ids_new_id_for_new_cell():
    frame0 = [{"mask_bool": _mask(10, 10)}]
    frame1 = [{"mask_bool": _mask(10, 10)}, {"mask_bool": _mask(40, 40)}]
    ids = assign_track_ids([frame0, frame1], iou_thresh=0.3)
    assert ids[1][0] == ids[0][0]
    assert ids[1][1] != ids[0][0]


def test_track_ids_retire_without_max_age():
    frame0 = [{"mask_bool": _mask(10, 10)}]
    frame1: list = []                          # cell missed this frame
    frame2 = [{"mask_bool": _mask(10, 10)}]     # reappears in the same spot
    ids = assign_track_ids([frame0, frame1, frame2], iou_thresh=0.3, max_age=0)
    assert ids[2][0] != ids[0][0], "max_age=0 must not bridge a missed frame"


def test_track_ids_bridge_gap_with_max_age():
    frame0 = [{"mask_bool": _mask(10, 10)}]
    frame1: list = []
    frame2 = [{"mask_bool": _mask(10, 10)}]
    ids = assign_track_ids([frame0, frame1, frame2], iou_thresh=0.3, max_age=1)
    assert ids[2][0] == ids[0][0], "max_age=1 should bridge a single missed frame"
