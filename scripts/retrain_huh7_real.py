"""Real-config Huh7 training (for a >=16GB cloud GPU, e.g. RunPod RTX 4090/A40).

Trains the FULL v3.2 BSGM architecture that does NOT fit the local 6GB card:
  target_size 512, enc_layers 4, dec_layers 6, dim_feedforward 1024,
  num_queries 300, num_feature_levels 4   (from cfgs/ctchuh7_bsgm.yaml)
plus the two correctness fixes validated locally:
  * criterion.loss_labels already fixed (only matched queries -> cell target)
  * focal_alpha 0.25 -> 0.5  (up-weight the rare cell class; avoids under-confidence)

Reuses run_full_epoch's proven full-Huh7 data pipeline (reads from $DEEP_CSTQ_DATA).
AMP stays OFF: the matcher NaNs under autocast; a 24GB GPU fits fp32 comfortably.

Checkpoints -> results/ctc-huh7-real/checkpoint_epoch{N}.pth

Usage (on the pod, env jz-AI-CSTQ-v02 active, data uploaded):
    export DEEP_CSTQ_DATA=/workspace/data/Deep_CSTQ_Datasets/src/output
    python scripts/retrain_huh7_real.py --epochs 24 --lr_drop 20
"""
import os, sys, argparse
from pathlib import Path
import yaml
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_full_epoch as rfe
from ai_cstq.models import build_model
from ai_cstq.models.criterion import build_criterion
from ai_cstq.engine import build_optimizer


def load_real_cfg():
    p = ROOT / "cfgs" / "ctchuh7_bsgm.yaml"
    cfg = yaml.safe_load(open(p))
    base = yaml.safe_load(open(ROOT / "cfgs" / cfg.pop("base_config")))
    base.update(cfg)
    cfg = base
    cfg["with_div"] = False        # keep eval-compatible with evaluate_ctc.py
    cfg["focal_alpha"] = 0.5       # calibration fix (validated direction)
    cfg["set_cost_class"] = 4.0    # matcher: balance class vs box cost (was 1.0)
    cfg["cls_loss_coef"] = 8.0     # stronger cls gradient (was 4.0)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr_drop", type=int, default=20)
    args = ap.parse_args()

    cfg = load_real_cfg()
    img_size = int(cfg["target_size"][0])

    rfe._patch_coco_script()
    device = torch.device("cuda")
    print(f"REAL config | img={img_size} nq={cfg['num_queries']} "
          f"enc={cfg['enc_layers']} dec={cfg['dec_layers']} ff={cfg['dim_feedforward']} "
          f"focal_alpha={cfg['focal_alpha']} | epochs={args.epochs} lr_drop={args.lr_drop}")
    print(f"DEEP_CSTQ_DATA = {rfe.DEEP_CSTQ_SRC}")

    src_tag, dst_tag, name, max_seqs, src_type = rfe.DATASETS["huh7"]
    seqs = rfe.get_train_seqs(src_tag, src_type, max_seqs)
    ann_file = rfe.ensure_coco(src_tag, dst_tag, src_type, seqs, regen=False)

    ckpt_dir = ROOT / "results" / "ctc-huh7-real"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    optimizer = build_optimizer(model, {
        "lr": cfg.get("lr", 2e-4),
        "lr_backbone": cfg.get("lr_backbone", 2e-5),
        "weight_decay": cfg.get("weight_decay", 1e-4),
    })
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_drop], gamma=0.1)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n/1e6:.1f}M")

    # resume
    existing = sorted(ckpt_dir.glob("checkpoint_epoch*.pth"),
                      key=lambda p: int(p.stem.replace("checkpoint_epoch", "")))
    start = 1
    if existing:
        ep = int(existing[-1].stem.replace("checkpoint_epoch", ""))
        c = torch.load(existing[-1], map_location="cpu")
        model.load_state_dict(c["model_state"])
        if "optimizer_state" in c:
            optimizer.load_state_dict(c["optimizer_state"])
        for _ in range(ep):
            scheduler.step()
        start = ep + 1
        print(f"resumed from epoch {ep}")

    for epoch in range(start, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch}/{args.epochs}  (lr={lr_now:.2e}) ---")
        r = rfe.run_one_epoch(
            src_tag=src_tag, dst_tag=dst_tag, src_type=src_type, name=name,
            seqs=seqs, ann_file=ann_file, device=device, img_size=img_size,
            print_every=100, model=model, optimizer=optimizer, criterion=criterion,
        )
        model, optimizer, criterion = r["model"], r["optimizer"], r["criterion"]
        scheduler.step()
        print(f"[huh7-real] Epoch {epoch} done  loss={r['avg_loss']:.3f}  "
              f"{r['avg_ms']:.0f}ms/batch  {r['total_min']:.1f}min  err={r['n_err']}")
        torch.save({
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg, "epoch": epoch, "avg_loss": r["avg_loss"], "dataset": name,
        }, ckpt_dir / f"checkpoint_epoch{epoch}.pth")
        print(f"CHECKPOINT_READY epoch{epoch}")

    print("\nDONE")


if __name__ == "__main__":
    main()
