"""
Deep-CSTQ Family: Unified Evaluation Report
============================================
Compares all four systems on the same CTC test sequences:

  v1.0  Deep-CSTQ-GR    (GRU + GNN,  tracking-by-detection, needs pre-segmentation)
  v2.0  Deep-CSTQ-MG    (Mamba + GNN, tracking-by-detection, needs pre-segmentation)
  v3.0  Cell-TRACTR      (ResNet50 + Deformable DETR, end-to-end)
  v3.2  BSGM-CellTrack   (Swin-T + Mamba + GATv2 + Bayes, end-to-end)

Usage
-----
  # Step A — run each system's inference first (see --help for skip flags)
  python scripts/generate_eval_report.py --run-all

  # Step B — collect scores only (inference already done)
  python scripts/generate_eval_report.py --collect-only

  # Single system + single dataset
  python scripts/generate_eval_report.py --systems bsgm celltractr --datasets huh7 gowt1

  # Show architecture comparison only (no inference)
  python scripts/generate_eval_report.py --arch-only

Fair evaluation notes
---------------------
- Deep_CSTQ (v1.0/v2.0) requires pre-computed segmentation masks as input.
  Two evaluation modes are reported:
    (a) With Silver Truth (ST) masks  → realistic inference condition
    (b) With GT masks                 → upper-bound / oracle condition
  This distinction is critical: ST/GT masks directly determine TRA.
  End-to-end systems (v3.0/v3.2) produce their own segmentation.

- All systems evaluated on the same test sequences, same CTC binaries:
    TRAMeasure.exe / SEGMeasure.exe / DETMeasure.exe

- BSGM: real config (512², nq=300, enc=4, dec=6, focal_alpha=0.5, AMP off).
  Criterion bug fix applied (2026-06-11). Best checkpoint per dataset selected
  by lowest validation loss, not necessarily epoch 24.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Repo roots ────────────────────────────────────────────────────────────────
ROOT_BSGM    = Path(__file__).parent.parent          # jz-AI-CSTQ-v02
ROOT_CTC     = Path("F:/GitHub/99-CellTracktor/code-win11")
ROOT_DCSTQ   = Path("F:/GitHub/Deep_CSTQ")
EVAL_BINS    = Path(os.environ.get(
    "CTC_EVAL_BINS",
    "F:/GitHub/99-CellTracktor/EvaluationSoftware/Win"
))

# ── Conda environments ────────────────────────────────────────────────────────
ENV_BSGM   = "jz-AI-CSTQ-v02"
ENV_CTC    = "99-celltractr"
ENV_DCSTQ  = "deepcstq"

# ── Dataset registry ──────────────────────────────────────────────────────────
# key → (ctc_full_name, bsgm_dst_tag, ctc_seqs_for_99ctc, bsgm_ckpt_epoch, bsgm_ckpt_dir)
DATASETS = {
    "huh7":  ("Fluo-C2DL-Huh7",  "ctc-huh7",  ["37"],       24, "ctc-huh7-real"),
    "gowt1": ("Fluo-N2DH-GOWT1", "ctc-gowt1", ["03", "04"], 24, "ctc-gowt1"),
    "u373":  ("PhC-C2DH-U373",   "ctc-u373",  ["03", "04"], 24, "ctc-u373"),
    "psc":   ("PhC-C2DL-PSC",    "ctc-psc",   ["05", "06"], 24, "ctc-psc"),
    "dhela": ("DIC-C2DH-HeLa",   "ctc-dhela", ["03", "04"], 24, "ctc-dhela"),
    "sim":   ("Fluo-N2DH-SIM+",  "ctc-sim",   ["19", "20"], 24, "ctc-sim"),
}

ALL_DATASETS = list(DATASETS.keys())
ALL_SYSTEMS  = ["dcstq_gr_st", "dcstq_mg_st", "celltractr", "bsgm"]

# Display names for the report
SYSTEM_LABELS = {
    "dcstq_gr_st":  "v1.0 Deep-CSTQ-GR (ST mask)",
    "dcstq_mg_st":  "v2.0 Deep-CSTQ-MG (ST mask)",
    "dcstq_gr_gt":  "v1.0 Deep-CSTQ-GR (GT mask)",
    "dcstq_mg_gt":  "v2.0 Deep-CSTQ-MG (GT mask)",
    "celltractr":   "v3.0 Cell-TRACTR (end-to-end)",
    "bsgm":         "v3.2 BSGM-CellTrack (end-to-end)",
}


# ─────────────────────────────────────────────────────────────────────────────
# CTC evaluation binary runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ctc_eval(parent_dir: Path, seq: str, digits: int = 3) -> dict[str, float]:
    """Call TRAMeasure / SEGMeasure / DETMeasure and parse scores."""
    scores = {}
    for metric in ("TRA", "SEG", "DET"):
        exe = EVAL_BINS / f"{metric}Measure.exe"
        if not exe.exists():
            print(f"    [WARN] Binary not found: {exe}")
            scores[metric] = float("nan")
            continue
        cmd = [str(exe), str(parent_dir), seq, str(digits)]
        if metric == "DET":
            cmd.append("1")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out = result.stdout + result.stderr
            # parse "TRA measure: 0.9174" or "DET: 0.9174" or bare float
            m = re.search(r"(\d\.\d+)", out)
            scores[metric] = float(m.group(1)) if m else float("nan")
        except Exception as e:
            print(f"    [WARN] {metric}Measure failed: {e}")
            scores[metric] = float("nan")
    return scores


def avg_scores(seq_scores: list[dict]) -> dict[str, float]:
    """Average scores across multiple sequences."""
    out: dict[str, float] = {}
    for metric in ("TRA", "SEG", "DET"):
        vals = [s[metric] for s in seq_scores if not _isnan(s.get(metric, float("nan")))]
        out[metric] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _isnan(v: float) -> bool:
    return v != v


# ─────────────────────────────────────────────────────────────────────────────
# v3.0 Cell-TRACTR runner
# ─────────────────────────────────────────────────────────────────────────────

def run_celltractr(dataset_key: str, skip_infer: bool = False) -> dict[str, float]:
    """
    Run 99-CellTracktor inference + eval for one dataset.
    Uses run_infer_eval.py which handles the full pipeline.
    Returns averaged TRA/SEG/DET across test sequences.
    """
    full_name, _, seqs, _, _ = DATASETS[dataset_key]
    ctc_dataset_key = f"ctc{dataset_key}"   # e.g. ctchuh7

    print(f"\n  [Cell-TRACTR] {full_name}")

    # Map dataset key to CellTracktor dataset name
    ctc_name_map = {
        "huh7":  "ctchuh7",
        "gowt1": "ctcgowt1",
        "u373":  "ctcu373",
        "psc":   "ctcpsc",
        "dhela": "ctcdhela",  # may not exist — check cfgs/
        "sim":   "ctcsim",
    }
    ctc_ds = ctc_name_map.get(dataset_key)
    if ctc_ds is None:
        print(f"    [SKIP] No CellTracktor config for {dataset_key}")
        return {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}

    cfg_path = ROOT_CTC / "cfgs" / f"train_{ctc_ds}.yaml"
    if not cfg_path.exists():
        print(f"    [SKIP] Config not found: {cfg_path}")
        return {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}

    if not skip_infer:
        print(f"    Running inference via run_infer_eval.py (env={ENV_CTC}) ...")
        seqs_str = " ".join(seqs)
        cmd = [
            "conda", "run", "-n", ENV_CTC, "--no-capture-output",
            "python", "run_infer_eval.py",
            "--dataset", ctc_ds,
            "--seqs", *seqs,
            "--skip-eval",   # we'll run eval ourselves for uniform parsing
        ]
        result = subprocess.run(cmd, cwd=ROOT_CTC)
        if result.returncode != 0:
            print(f"    [ERROR] Inference failed (rc={result.returncode})")
            return {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}

    # Evaluate: RES dirs at ROOT_CTC/data/{ctc_ds}/CTC/test/{seq}_RES
    test_dir = ROOT_CTC / "data" / ctc_ds / "CTC" / "test"
    all_seq_scores = []
    for seq in seqs:
        res_dir = test_dir / f"{seq}_RES"
        if not res_dir.exists():
            print(f"    [WARN] RES dir missing: {res_dir}")
            all_seq_scores.append({"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")})
            continue
        s = run_ctc_eval(test_dir, seq, digits=3)
        print(f"    seq {seq}  TRA={s['TRA']:.4f}  SEG={s['SEG']:.4f}  DET={s['DET']:.4f}")
        all_seq_scores.append(s)

    return avg_scores(all_seq_scores)


# ─────────────────────────────────────────────────────────────────────────────
# v3.2 BSGM runner
# ─────────────────────────────────────────────────────────────────────────────

def run_bsgm(dataset_key: str, skip_infer: bool = False,
             conf_threshold: float = 0.5) -> dict[str, float]:
    """
    Run BSGM-CellTrack inference + eval for one dataset via evaluate_ctc.py.
    Returns averaged TRA/SEG/DET.
    """
    full_name, dst_tag, seqs, ckpt_epoch, ckpt_dir = DATASETS[dataset_key]
    print(f"\n  [BSGM v3.2] {full_name}")

    if not skip_infer:
        print(f"    Running evaluate_ctc.py (env={ENV_BSGM}, epoch={ckpt_epoch}) ...")
        cmd = [
            "conda", "run", "-n", ENV_BSGM, "--no-capture-output",
            "python", str(ROOT_BSGM / "scripts" / "evaluate_ctc.py"),
            "--datasets", dataset_key,
            "--ckpt_epoch", str(ckpt_epoch),
            "--ckpt_dir", ckpt_dir,
            "--conf_threshold", str(conf_threshold),
            "--max_track_queries", "300",
        ]
        result = subprocess.run(cmd, cwd=ROOT_BSGM,
                                env={**os.environ,
                                     "PYTHONUTF8": "1",
                                     "PYTHONIOENCODING": "utf-8"})
        if result.returncode != 0:
            print(f"    [ERROR] evaluate_ctc.py failed (rc={result.returncode})")
            return {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}

    # BSGM writes RES to Deep_CSTQ_Datasets test dir (for 'deep' sources)
    # or ROOT_BSGM/data/{dst_tag}/CTC/test/ (for local sources)
    deep_cstq_data = Path(os.environ.get(
        "DEEP_CSTQ_DATA", "F:/GitHub/Deep_CSTQ_Datasets/src/output"))

    deep_datasets = {"huh7": "Fluo-C2DL-Huh7",
                     "gowt1": "Fluo-N2DH-GOWT1",
                     "u373": "PhC-C2DH-U373"}
    if dataset_key in deep_datasets:
        test_parent = deep_cstq_data / deep_datasets[dataset_key] / "test"
        # pick the first seq dir (e.g. "37" for Huh7)
        if test_parent.exists():
            raw_seqs = sorted(d.name for d in test_parent.iterdir()
                              if d.is_dir() and not d.name.endswith(("_GT", "_RES")))
            eval_seqs = raw_seqs[:1]
        else:
            eval_seqs = seqs[:1]
    else:
        test_parent = ROOT_BSGM / "data" / dst_tag / "CTC" / "test"
        eval_seqs = ["01"]

    all_seq_scores = []
    for seq in eval_seqs:
        s = run_ctc_eval(test_parent, seq, digits=3)
        print(f"    seq {seq}  TRA={s['TRA']:.4f}  SEG={s['SEG']:.4f}  DET={s['DET']:.4f}")
        all_seq_scores.append(s)

    return avg_scores(all_seq_scores)


# ─────────────────────────────────────────────────────────────────────────────
# v1.0 / v2.0 Deep_CSTQ GNN runner  (placeholder — fill in when interface known)
# ─────────────────────────────────────────────────────────────────────────────

def run_deep_cstq(dataset_key: str, temporal: str = "gru",
                  seg_mode: str = "ST", skip_infer: bool = False) -> dict[str, float]:
    """
    Run Deep_CSTQ (GNN-based) inference + CTC eval.

    Parameters
    ----------
    temporal : "gru"   → v1.0 Deep-CSTQ-GR
               "mamba" → v2.0 Deep-CSTQ-MG
    seg_mode : "ST"    → Silver Truth masks (realistic)
               "GT"    → Ground Truth masks (oracle upper bound)

    NOTE: Fill in the actual CLI call once Deep_CSTQ's evaluate.py interface
    is confirmed.  The RES output path must match what run_ctc_eval() expects.
    """
    full_name, _, _, _, _ = DATASETS[dataset_key]
    label = f"Deep-CSTQ-{'GR' if temporal == 'gru' else 'MG'} ({seg_mode})"
    print(f"\n  [{label}] {full_name}")

    if not skip_infer:
        # TODO: replace with the real Deep_CSTQ evaluate CLI call.
        # Example (adjust to actual script name and args):
        #   cmd = [
        #       "conda", "run", "-n", ENV_DCSTQ, "--no-capture-output",
        #       "python", str(ROOT_DCSTQ / "evaluate.py"),
        #       "--dataset", dataset_key,
        #       "--use_temporal", temporal,
        #       "--seg_subdir_inf", "02_ST/SEG" if seg_mode == "ST" else "01_GT/SEG",
        #   ]
        #   subprocess.run(cmd, cwd=ROOT_DCSTQ)
        print(f"    [TODO] Deep_CSTQ inference not yet wired up. "
              f"Run manually in env '{ENV_DCSTQ}', then re-run with --collect-only.")
        return {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}

    # After inference, Deep_CSTQ should write RES to:
    #   ROOT_DCSTQ / "data" / {dataset_key} / "CTC" / "test" / "{seq}_RES"
    # Adjust the path below to match the actual output location.
    test_dir = ROOT_DCSTQ / "data" / dataset_key / "CTC" / "test"
    _, _, seqs, _, _ = DATASETS[dataset_key]
    all_seq_scores = []
    for seq in seqs:
        if not (test_dir / f"{seq}_RES").exists():
            print(f"    [WARN] RES dir missing: {test_dir / (seq + '_RES')}")
            all_seq_scores.append({"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")})
            continue
        s = run_ctc_eval(test_dir, seq, digits=3)
        print(f"    seq {seq}  TRA={s['TRA']:.4f}  SEG={s['SEG']:.4f}  DET={s['DET']:.4f}")
        all_seq_scores.append(s)

    return avg_scores(all_seq_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture comparison table (static, printed without running inference)
# ─────────────────────────────────────────────────────────────────────────────

ARCH_TABLE = {
    "Paradigm":        ["Tracking-by-Detection", "Tracking-by-Detection", "End-to-End",       "End-to-End"],
    "Input":           ["img + seg mask",         "img + seg mask",         "img only",          "img only"],
    "Backbone":        ["ResNet-50",              "ResNet-50",              "ResNet-50",          "Swin-T"],
    "Temporal":        ["GRU",                    "Mamba SSM",              "3-frame Conv",       "Mamba SSM"],
    "Spatial model":   ["GNN edge-classif.",      "GNN edge-classif.",      "Deformable DETR",    "GATv2 + Deformable DETR"],
    "Uncertainty":     ["None",                   "None",                   "None",               "BayesianDropout"],
    "Hidden dim":      ["—",                      "—",                      "144",                "256"],
    "Enc / Dec layers":["—",                      "—",                      "4 / 4",              "4 / 6"],
    "Num queries":     ["—",                      "—",                      "400",                "300–400"],
    "Train img size":  ["—",                      "—",                      "512² (Huh7)·1024² (GOWT1)", "512²"],
    "Seg output":      ["Uses input mask",        "Uses input mask",        "Mask head",          "Mask head"],
    "Division detect": ["Yes (edge label)",       "Yes (edge label)",       "Yes (8D box)",       "Yes (8D box)"],
    "CUDA extension":  ["No",                     "No",                     "Yes (ms_deform_attn)","No (pure PyTorch)"],
    "Params (M)":      ["~5–10",                  "~5–10",                  "~34",                "~50"],
}

SYSTEMS_HEADER = [
    "v1.0 Deep-CSTQ-GR",
    "v2.0 Deep-CSTQ-MG",
    "v3.0 Cell-TRACTR",
    "v3.2 BSGM (ours)",
]


def print_arch_table():
    col_w = [28] + [24] * 4
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def row(cells):
        parts = []
        for c, w in zip(cells, col_w):
            parts.append(f" {str(c):<{w-2}} ")
        return "|" + "|".join(parts) + "|"

    print("\n" + "=" * 130)
    print("  ARCHITECTURE COMPARISON")
    print("=" * 130)
    print(sep)
    print(row(["Property"] + SYSTEMS_HEADER))
    print(sep)
    for prop, vals in ARCH_TABLE.items():
        print(row([prop] + vals))
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = {"TRA": 0.94, "SEG": 0.88, "DET": 0.92}

PARADIGM_NOTE = """
FAIRNESS NOTE
─────────────
v1.0 / v2.0 (GNN) receive pre-computed segmentation masks as input.
  • "ST mask"  = Silver Truth (algorithm-generated) — realistic deployment condition.
  • "GT mask"  = Ground Truth (human annotation)    — oracle upper bound only.
v3.0 / v3.2 produce segmentation from raw images with no external input.
Direct TRA comparison across paradigms must account for this asymmetry.
"""


def fmt(v: float) -> str:
    if _isnan(v):
        return "  —   "
    mark = " ✓" if v >= 0.90 else "  "
    return f"{v:.4f}{mark}"


def generate_report(results: dict, out_path: Path | None = None):
    """Print and optionally save the final comparison report."""
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p()
    p("=" * 100)
    p("  Deep-CSTQ Family — CTC Evaluation Report")
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p("=" * 100)
    p(PARADIGM_NOTE)

    for ds_key, ds_results in results.items():
        if not ds_results:
            continue
        full_name = DATASETS[ds_key][0]
        p(f"\n{'─'*100}")
        p(f"  Dataset: {full_name}  ({ds_key})")
        p(f"{'─'*100}")
        hdr = f"  {'System':<38}  {'TRA':>8}  {'SEG':>8}  {'DET':>8}"
        p(hdr)
        p(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*8}")

        # Group: tracking-by-detection
        p("  ── Tracking-by-Detection (requires external segmentation) ──")
        for sys_key in ["dcstq_gr_st", "dcstq_mg_st", "dcstq_gr_gt", "dcstq_mg_gt"]:
            if sys_key not in ds_results:
                continue
            s = ds_results[sys_key]
            label = SYSTEM_LABELS[sys_key]
            p(f"  {label:<38}  {fmt(s['TRA']):>8}  {fmt(s['SEG']):>8}  {fmt(s['DET']):>8}")

        # Group: end-to-end
        p("  ── End-to-End (raw image → segmentation + tracking) ────────")
        for sys_key in ["celltractr", "bsgm"]:
            if sys_key not in ds_results:
                continue
            s = ds_results[sys_key]
            label = SYSTEM_LABELS[sys_key]
            p(f"  {label:<38}  {fmt(s['TRA']):>8}  {fmt(s['SEG']):>8}  {fmt(s['DET']):>8}")

        p(f"  {'CTC target (≥)':.<38}  {'≥0.94':>8}  {'≥0.88':>8}  {'≥0.92':>8}")

    # Summary heatmap across datasets
    p()
    p("=" * 100)
    p("  SUMMARY — Average TRA across all datasets (end-to-end systems only)")
    p("=" * 100)
    for sys_key in ["celltractr", "bsgm"]:
        all_tra = [results[ds][sys_key]["TRA"]
                   for ds in results if sys_key in results.get(ds, {})]
        valid = [v for v in all_tra if not _isnan(v)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        label = SYSTEM_LABELS[sys_key]
        p(f"  {label:<38}  avg TRA = {fmt(avg)}")

    p()
    p("  ✓ = meets TRA ≥ 0.90")
    p()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Report saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Deep-CSTQ family unified evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--systems", nargs="+", default=ALL_SYSTEMS,
                   choices=["dcstq_gr_st", "dcstq_mg_st",
                             "dcstq_gr_gt", "dcstq_mg_gt",
                             "celltractr", "bsgm"],
                   help="Systems to evaluate")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS)
    p.add_argument("--run-all",      action="store_true",
                   help="Run inference + eval for all selected systems")
    p.add_argument("--collect-only", action="store_true",
                   help="Skip inference; collect scores from existing RES dirs")
    p.add_argument("--arch-only",    action="store_true",
                   help="Print architecture table only, no evaluation")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="Confidence threshold for BSGM (default 0.5)")
    p.add_argument("--bsgm-epoch", type=int, default=None,
                   help="Override BSGM checkpoint epoch (default: per-dataset config)")
    p.add_argument("--results-json", type=str, default=None,
                   help="Load previously saved JSON results instead of running")
    p.add_argument("--out-dir", type=str,
                   default=str(ROOT_BSGM / "results" / "eval_report"),
                   help="Output directory for report and JSON")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    print_arch_table()

    if args.arch_only:
        return

    # ── Load previously saved results ──────────────────────────────────────
    if args.results_json:
        with open(args.results_json) as f:
            results = json.load(f)
        print(f"Loaded results from {args.results_json}")
        generate_report(results, out_dir / "report.txt")
        return

    skip_infer = args.collect_only and not args.run_all

    # ── Run evaluations ────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    t0 = time.perf_counter()

    for ds_key in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {DATASETS[ds_key][0]}  ({ds_key})")
        print(f"{'='*60}")
        results[ds_key] = {}

        # Override BSGM epoch if requested
        if args.bsgm_epoch is not None:
            DATASETS[ds_key] = (
                DATASETS[ds_key][0],
                DATASETS[ds_key][1],
                DATASETS[ds_key][2],
                args.bsgm_epoch,
                DATASETS[ds_key][4],
            )

        for sys_key in args.systems:
            try:
                if sys_key == "bsgm":
                    scores = run_bsgm(ds_key, skip_infer, args.conf_threshold)
                elif sys_key == "celltractr":
                    scores = run_celltractr(ds_key, skip_infer)
                elif sys_key == "dcstq_gr_st":
                    scores = run_deep_cstq(ds_key, "gru", "ST", skip_infer)
                elif sys_key == "dcstq_mg_st":
                    scores = run_deep_cstq(ds_key, "mamba", "ST", skip_infer)
                elif sys_key == "dcstq_gr_gt":
                    scores = run_deep_cstq(ds_key, "gru", "GT", skip_infer)
                elif sys_key == "dcstq_mg_gt":
                    scores = run_deep_cstq(ds_key, "mamba", "GT", skip_infer)
                else:
                    scores = {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}
                results[ds_key][sys_key] = scores
            except Exception as e:
                print(f"  [ERROR] {sys_key} on {ds_key}: {e}")
                results[ds_key][sys_key] = {"TRA": float("nan"), "SEG": float("nan"), "DET": float("nan")}

    elapsed = time.perf_counter() - t0
    print(f"\n  Total evaluation time: {elapsed/60:.1f} min")

    # ── Save raw scores ────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = out_dir / f"scores_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Scores JSON saved: {json_path}")

    # ── Generate report ────────────────────────────────────────────────────
    generate_report(results, out_dir / f"report_{ts}.txt")


if __name__ == "__main__":
    main()
