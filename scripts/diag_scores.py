"""Diagnostic: dump detection-score distribution for a trained checkpoint.

Answers: why does every query pass conf_threshold=0.3?
Compares keep rules:  sigmoid(ch0)>0.3   vs   argmax(ch0 vs ch1)=cell   vs   top-k.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ai_cstq.models import build_model

DEEP = Path("F:/GitHub/Deep_CSTQ_Datasets/src/output")
IMG_SIZE = 256


def load_frame(path):
    from PIL import Image
    img = np.array(Image.open(str(path)).convert("L")).astype(np.float32)
    img = img / (img.max() + 1e-6)
    t = torch.from_numpy(img)[None, None]
    t = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t.repeat(1, 3, 1, 1)[0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ctc-huh7/checkpoint_epoch19.pth")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / a.ckpt, map_location="cpu")
    model = build_model(ckpt["cfg"]).to(dev).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {a.ckpt}  (num_queries={ckpt['cfg'].get('num_queries')}, train loss {ckpt['avg_loss']:.3f})")

    seq = DEEP / "Fluo-C2DL-Huh7" / "test" / "37"
    frames = sorted(seq.glob("t*.tif"))[:3]
    fs = [load_frame(f).to(dev) for f in frames]

    # count GT cells in frame 1
    gt_dir = seq.parent / "37_GT" / "TRA"
    gts = sorted(gt_dir.glob("man_track*.tif"))
    if gts:
        from PIL import Image
        m = np.array(Image.open(str(gts[1])))
        print(f"GT cells in frame 1: {len(np.unique(m)) - 1}")

    with torch.no_grad():
        out = model([fs[0][None], fs[1][None], fs[2][None]])
    logits = out["pred_logits"][0]          # (N, 2)
    print(f"\npred_logits shape: {tuple(out['pred_logits'].shape)}")
    ch0 = logits[:, 0].sigmoid()            # cell
    ch1 = logits[:, 1].sigmoid()            # background

    def stats(x):
        return f"min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} median={x.median():.3f}"
    print(f"\nch0 (cell)       sigmoid: {stats(ch0)}")
    print(f"ch1 (background) sigmoid: {stats(ch1)}")
    print(f"\nraw logits ch0: min={logits[:,0].min():.2f} max={logits[:,0].max():.2f} mean={logits[:,0].mean():.2f}")
    print(f"raw logits ch1: min={logits[:,1].min():.2f} max={logits[:,1].max():.2f} mean={logits[:,1].mean():.2f}")

    N = logits.shape[0]
    print(f"\nkeep rules (N={N} queries):")
    for thr in (0.3, 0.5, 0.7, 0.9):
        print(f"  sigmoid(ch0) > {thr}:  {(ch0 > thr).sum().item()} kept")
    print(f"  argmax(ch0 vs ch1)=cell:  {(ch0 > ch1).sum().item()} kept")

    order = ch0.argsort(descending=True)
    print(f"\ntop-8 ch0 scores: {[round(ch0[i].item(),3) for i in order[:8]]}")
    print(f"their ch1 scores: {[round(ch1[i].item(),3) for i in order[:8]]}")


if __name__ == "__main__":
    main()
