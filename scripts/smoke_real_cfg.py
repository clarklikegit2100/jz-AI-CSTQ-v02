"""GPU-fit smoke test for the REAL Huh7 BSGM config (vs the dry-run toy config).

Runs one forward+backward at 512x512 / num_queries=300 and reports peak GPU mem,
so we know whether a full retrain fits on the 6GB RTX 3050 before committing hours.
Isolates the variables under test; keeps with_div=False for eval compatibility.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
from ai_cstq.models import build_model
from ai_cstq.models.criterion import build_criterion
from ai_cstq.datasets import build_dataset
from ai_cstq.datasets.ctc_coco import collate_fn
import torch.nn.functional as F


def load_cfg():
    p = os.path.join(ROOT, "cfgs", "ctchuh7_bsgm.yaml")
    cfg = yaml.safe_load(open(p))
    base = yaml.safe_load(open(os.path.join(ROOT, "cfgs", cfg.pop("base_config"))))
    base.update(cfg)
    cfg = base
    cfg["dataset"] = "ctc-huh7"          # local data uses hyphen
    cfg["data_dir"] = os.path.join(ROOT, "data")
    cfg["with_div"] = False              # isolate num_queries fix; stay eval-compatible
    # --- Memory-safe controlled config for 6GB GPU ---
    # Keep toy resolution/depth/mamba (fits 6GB); change ONLY the variable that
    # caused the classification collapse: num_queries 20 -> 300.
    cfg["target_size"] = [256, 256]
    cfg["enc_layers"] = 1
    cfg["dec_layers"] = 2
    cfg["dim_feedforward"] = 256
    cfg["num_feature_levels"] = 4
    cfg["dec_n_points"] = 2
    cfg["mamba_d_state"] = 4
    cfg["mamba_d_conv"] = 4
    cfg["graph_topk"] = 4
    cfg["graph_heads"] = 2
    cfg["bayesian_dropout"] = 0.0
    cfg["num_queries"] = 300             # THE FIX (was 20)
    cfg["use_amp"] = False               # toy run trained without AMP; AMP -> NaN in matcher
    return cfg


def main():
    dev = torch.device("cuda")
    cfg = load_cfg()
    print(f"REAL cfg: img={cfg['target_size']} num_queries={cfg['num_queries']} "
          f"enc={cfg['enc_layers']} dec={cfg['dec_layers']} dim_ff={cfg['dim_feedforward']} "
          f"with_div={cfg['with_div']} amp={cfg.get('use_amp')}")

    model = build_model(cfg).to(dev)
    crit = build_criterion(cfg).to(dev)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n/1e6:.1f}M")

    ds = build_dataset(cfg, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                         num_workers=0, collate_fn=collate_fn)
    frames_list, targets = next(iter(loader))
    frames = [f.to(dev) for f in frames_list]
    targets = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    print(f"frame shape: {frames[0].shape}  gt cells: {targets[0]['labels'].shape[0]}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_amp = cfg.get("use_amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    torch.cuda.reset_peak_memory_stats()

    model.train()
    # probe mask size
    model.eval()
    with torch.no_grad():
        probe = model(frames)
    pm = probe.get("pred_masks")
    mh, mw = (pm.shape[-2], pm.shape[-1]) if pm is not None else (cfg["target_size"][0]//4,)*2
    model.train()
    tr = []
    for t in targets:
        m = t["masks"].float()
        if m.shape[0] > 0:
            m = F.interpolate(m.unsqueeze(0), size=(mh, mw), mode="nearest").squeeze(0).bool()
        tr.append({**t, "masks": m})

    t0 = time.perf_counter()
    opt.zero_grad()
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(frames, targets=tr)
        loss = crit(out, tr)["loss_total"]
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nOK  loss={loss.item():.3f}  step={dt*1000:.0f}ms  "
          f"peak_mem={peak:.2f}GB / {total:.1f}GB")
    print(f"pred_logits {tuple(out['pred_logits'].shape)}  "
          f"pred_masks {tuple(out['pred_masks'].shape) if out.get('pred_masks') is not None else None}")


if __name__ == "__main__":
    main()
