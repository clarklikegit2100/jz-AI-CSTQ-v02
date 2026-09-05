import torch

from ai_cstq.models.ms_deform_attn import (
    MSDeformAttn,
    MSDeformEncoder,
    ms_deform_attn_core_pytorch,
    get_encoder_reference_points,
    get_valid_ratios,
)


def _shapes():
    return torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]], dtype=torch.long)


def _level_start(shapes):
    return torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])


def test_core_sampling_shapes_and_finiteness():
    B, n_heads, head_dim, n_points = 2, 8, 32, 4
    shapes = _shapes()
    S = int(shapes.prod(1).sum())
    Lq = 50
    value = torch.randn(B, S, n_heads, head_dim)
    loc = torch.rand(B, Lq, n_heads, len(shapes), n_points, 2)
    w = torch.rand(B, Lq, n_heads, len(shapes), n_points)
    out = ms_deform_attn_core_pytorch(value, shapes, loc, w)
    assert out.shape == (B, Lq, n_heads * head_dim)
    assert torch.isfinite(out).all()


def test_module_forward_backward_2d_and_4d_refs():
    B, d_model = 2, 256
    shapes = _shapes()
    S = int(shapes.prod(1).sum())
    lsi = _level_start(shapes)
    attn = MSDeformAttn(d_model, n_levels=len(shapes), n_heads=8, n_points=4)
    mem = torch.randn(B, S, d_model, requires_grad=True)
    q = torch.randn(B, 40, d_model, requires_grad=True)

    for ref_dim in (2, 4):
        ref = torch.rand(B, 40, len(shapes), ref_dim)
        out = attn(q, ref, mem, shapes, lsi)
        assert out.shape == (B, 40, d_model)
        assert torch.isfinite(out).all()
        out.sum().backward(retain_graph=True)
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert mem.grad is not None and torch.isfinite(mem.grad).all()
        q.grad = None
        mem.grad = None


def test_encoder_forward_backward_and_token_count_preserved():
    B, d_model = 2, 256
    shapes = _shapes()
    S = int(shapes.prod(1).sum())
    lsi = _level_start(shapes)
    enc = MSDeformEncoder(num_layers=2, d_model=d_model, n_levels=len(shapes))
    src = torch.randn(B, S, d_model, requires_grad=True)
    pos = torch.randn(B, S, d_model)
    out = enc(src, pos, shapes, lsi)
    assert out.shape == (B, S, d_model)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(src.grad).all()


def test_reference_points_in_unit_range_and_padding_ratios():
    B = 3
    shapes = _shapes()
    S = int(shapes.prod(1).sum())
    mask = torch.zeros(B, S, dtype=torch.bool)
    vr = get_valid_ratios(mask, shapes)
    assert vr.shape == (B, len(shapes), 2)
    assert torch.allclose(vr, torch.ones_like(vr))
    ref = get_encoder_reference_points(shapes, vr, torch.device("cpu"))
    assert ref.shape == (B, S, len(shapes), 2)
    assert (ref >= 0).all() and (ref <= 1).all()


def test_amp_autocast_runs_without_nan():
    if not torch.cuda.is_available():
        return
    B, d_model = 2, 256
    shapes = _shapes().cuda()
    S = int(shapes.prod(1).sum())
    lsi = _level_start(shapes).cuda()
    enc = MSDeformEncoder(num_layers=2, d_model=d_model, n_levels=len(shapes)).cuda()
    src = torch.randn(B, S, d_model, device="cuda", requires_grad=True)
    pos = torch.randn(B, S, d_model, device="cuda")
    with torch.autocast("cuda"):
        out = enc(src, pos, shapes, lsi)
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    assert torch.isfinite(src.grad).all()
