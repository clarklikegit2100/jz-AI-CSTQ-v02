"""Controlled retrain of Huh7 with the num_queries fix.

Reuses run_full_epoch's proven full-Huh7 data pipeline. Changes ONLY the two
hyperparameters responsible for the failure, everything else identical to the
dry-run toy config (so the comparison is clean and it fits the 6GB GPU):
  * num_queries  20 -> 300   (root cause: restores no-object supervision)
  * LR drop at epoch 8       (fixes the flat-LR overshoot)

Checkpoints -> results/ctc-huh7-fixed/checkpoint_epoch{N}.pth
"""
import os, sys, time, argparse
from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_full_epoch as rfe
from ai_cstq.models import build_model
from ai_cstq.models.criterion import build_criterion
from ai_cstq.engine import build_optimizer

# Toy config + the fix (num_queries 20 -> 300)
FIXED_CFG = dict(rfe.BSGM_CFG)
FIXED_CFG["num_queries"] = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr_drop", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    rfe._patch_coco_script()
    device = torch.device("cuda")
    print(f"device={device}  num_queries={FIXED_CFG['num_queries']} (was 20)  "
          f"img={args.img_size}  epochs={args.epochs}  lr_drop={args.lr_drop}")

    src_tag, dst_tag, name, max_seqs, src_type = rfe.DATASETS["huh7"]
    seqs = rfe.get_train_seqs(src_tag, src_type, max_seqs)
    ann_file = rfe.ensure_coco(src_tag, dst_tag, src_type, seqs, regen=False)

    ckpt_dir = ROOT / "results" / "ctc-huh7-fixed"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(FIXED_CFG).to(device)
    criterion = build_criterion(rfe.LOSS_CFG).to(device)
    optimizer = build_optimizer(model, {"lr": 2e-4, "lr_backbone": 2e-5, "weight_decay": 1e-4})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_drop], gamma=0.1)

    # resume if checkpoints exist
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
            seqs=seqs, ann_file=ann_file, device=device, img_size=args.img_size,
            print_every=100, model=model, optimizer=optimizer, criterion=criterion,
        )
        model, optimizer, criterion = r["model"], r["optimizer"], r["criterion"]
        scheduler.step()
        print(f"[huh7-fixed] Epoch {epoch} done  loss={r['avg_loss']:.3f}  "
              f"{r['avg_ms']:.0f}ms/batch  {r['total_min']:.1f}min  err={r['n_err']}")
        torch.save({
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "cfg": FIXED_CFG, "epoch": epoch, "avg_loss": r["avg_loss"], "dataset": name,
        }, ckpt_dir / f"checkpoint_epoch{epoch}.pth")
        print(f"CHECKPOINT_READY epoch{epoch}")

    print("\nDONE")


if __name__ == "__main__":
    main()
