"""Phase 5, todo 9b: gamma interventions on the frozen E4 fusion.

For each category, deterministically REBUILD the exact synthetic inputs used
by scripts/generate_attribution_data.py (seed rule: md5(source|kind|subtype|
severity); sources = first N TRAIN-NORMAL records) and re-run E4.predict under
four fusion gate settings, without retraining:

- full:          gamma_rgb, gamma_depth at their E4-trained values
- gamma_rgb=0:   RGB residual suppressed  (f_rgb = Phase 2 identity)
- gamma_depth=0: depth residual suppressed
- both=0:        cross-modal residual fully removed (identity / Phase 2)

Scores are reported as the image-max of the CALIBRATED fused mean map
(valid-depth masking, Phase 3A 50/50). "Delta" = full_score - gate_score, so
a positive Delta means the gated residual contributes to the decision.
Reported per injected kind x severity, per category, and pooled.

Interpretation: a larger Delta on RGB_ONLY injections when zeroing gamma_rgb
(and on DEPTH_ONLY when zeroing gamma_depth) shows the residual fusion is
modality-directional. Evidence remains RELATIVE anomaly evidence.

Writes experiments/tables/gamma_intervention_{summary,detail,pooled}.csv and
gamma_intervention_metrics.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from xmvad.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load shared synthetic-sample helpers deterministically from the generator.
_GEN = importlib.util.spec_from_file_location(
    "generate_attribution_data",
    REPO_ROOT / "scripts" / "generate_attribution_data.py")
gen = importlib.util.module_from_spec(_GEN)
sys.modules["generate_attribution_data"] = gen
_GEN.loader.exec_module(gen)

PLAN, det_seed, apply_variant, tensorize, build_e4 = (
    gen.PLAN, gen.det_seed, gen.apply_variant, gen.tensorize, gen.build_e4)

GATES = ("full", "gamma_rgb_zero", "gamma_depth_zero", "both_zero")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--config", default="configs/concat_fusion.yaml")
    ap.add_argument("--rgb-checkpoint", default="checkpoints/E1_baseline_rgb")
    ap.add_argument("--depth-checkpoint", default="checkpoints/E2_baseline_depth")
    ap.add_argument("--fusion-checkpoint", default="checkpoints/E4_concat_fusion")
    ap.add_argument("--calibration", default="experiments/tables/E4_concat_calibration.json")
    ap.add_argument("--sources", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from xmvad.data.depth import extract_depth, load_xyz, valid_mask_from_xyz
    from xmvad.data.mvtec3d import discover_split, load_rgb
    from xmvad.data.stats import fit_train_depth_stats
    from xmvad.data.transforms import SynchronizedTransform
    from xmvad.fusion.score_fusion import calibrate
    from xmvad.metrics.scores import variant_scores

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/raw/mvtec3d"))
    input_size = tuple(data_cfg.get("input_size", [256, 256]))
    H, W = input_size
    rgb_mean = tuple(data_cfg.get("rgb_mean", [0.485, 0.456, 0.406]))
    rgb_std = tuple(data_cfg.get("rgb_std", [0.229, 0.224, 0.225]))
    clip = tuple(data_cfg["depth_clip"]) if data_cfg.get("depth_clip") else None
    with open(REPO_ROOT / args.calibration, encoding="utf-8") as f:
        calib = json.load(f)["categories"]
    device = torch.device(args.device)

    summary_rows, detail_rows = [], []

    for cat in args.categories:
        pt = str(sorted(Path(args.fusion_checkpoint).glob(f"{cat}_best.pt"))[0])
        sd = torch.load(pt, map_location="cpu", weights_only=False)["fusion_state"]
        gv = {"gamma_rgb": float(sd["gamma_rgb"]), "gamma_depth": float(sd["gamma_depth"])}
        print(f"== {cat} gamma={gv} ==", flush=True)
        depth_stats = fit_train_depth_stats(root, cat, data_cfg.get("depth_norm", "robust_median"),
                                            verbose=False)
        tf = SynchronizedTransform(input_size=input_size, rgb_mean=rgb_mean, rgb_std=rgb_std,
                                   depth_stats=None, depth_clip=clip, train=False)
        e4 = build_e4(cfg, cat, args.rgb_checkpoint, args.depth_checkpoint,
                      args.fusion_checkpoint, device)
        cm = calib[cat]
        med_r, sc_r = cm["rgb"]["median"], cm["rgb"]["scale"]
        med_d, sc_d = cm["depth"]["median"], cm["depth"]["scale"]
        train_recs = sorted((r for r in discover_split(root, cat, "train") if r.label == 0),
                            key=lambda r: r.xyz_path.stem)[: args.sources]

        buckets = defaultdict(list)          # (kind, severity) -> list of delta dicts
        for rec in tqdm(train_recs, desc=f"interv {cat}", leave=False):
            sid = f"{cat}/train/good/{rec.xyz_path.stem}"
            rgb0 = np.array(load_rgb(rec.rgb_path))
            xyz0 = load_xyz(rec.xyz_path)
            depth0, valid0 = extract_depth(xyz0), valid_mask_from_xyz(xyz0)
            for kind, subtype, sev, kw in PLAN:
                seed = det_seed(sid, kind, subtype, sev)
                rgb_p, depth_p, valid_p, mask = apply_variant(rgb0, depth0, valid0,
                                                              kind, subtype, seed, kw)
                rgb_t, depth_t, valid_t = tensorize(Image.fromarray(rgb_p), depth_p,
                                                    valid_p, tf, depth_stats, clip, H, W)
                v = valid_t[0, 0].numpy().astype(bool)
                scores = {}
                for gate in GATES:
                    with torch.no_grad():
                        e4.fusion.gamma_rgb.data = torch.tensor(gv["gamma_rgb"] if gate in
                                                                ("full", "gamma_depth_zero") else 0.0)
                        e4.fusion.gamma_depth.data = torch.tensor(gv["gamma_depth"] if gate in
                                                                 ("full", "gamma_rgb_zero") else 0.0)
                        out = e4.predict(rgb_t.to(device), depth_t.to(device), valid_t.to(device))
                    zr = calibrate(out["rgb_map"][0, 0].cpu().numpy().astype(np.float64),
                                   med_r, sc_r)
                    zd = calibrate(out["depth_map"][0, 0].cpu().numpy().astype(np.float64),
                                   med_d, sc_d)
                    fused = np.where(v, 0.5 * zr + 0.5 * zd, zr)
                    scores[gate] = variant_scores(fused, None)["max"]
                if scores["full"] > 0.5:
                    buckets[(kind, sev)].append({
                        "full": scores["full"],
                        "drgb": scores["full"] - scores["gamma_rgb_zero"],
                        "ddepth": scores["full"] - scores["gamma_depth_zero"],
                        "dboth": scores["full"] - scores["both_zero"]})

        summary_rows.append({"category": cat, "gamma_rgb": gv["gamma_rgb"],
                             "gamma_depth": gv["gamma_depth"],
                             "n_samples": sum(len(x) for x in buckets.values())})
        for (kind, sev), lst in sorted(buckets.items()):
            detail_rows.append({"category": cat, "kind": kind, "severity": sev, "n": len(lst),
                                "s_full": float(np.mean([x["full"] for x in lst])),
                                "delta_rgb_resid": float(np.mean([x["drgb"] for x in lst])),
                                "delta_depth_resid": float(np.mean([x["ddepth"] for x in lst])),
                                "delta_no_resid": float(np.mean([x["dboth"] for x in lst]))})

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    pooled = detail.groupby(["kind", "severity"], as_index=False).agg(
        n=("n", "sum"), s_full=("s_full", "mean"),
        delta_rgb_resid=("delta_rgb_resid", "mean"),
        delta_depth_resid=("delta_depth_resid", "mean"),
        delta_no_resid=("delta_no_resid", "mean"))
    tbl_dir = REPO_ROOT / "experiments" / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tbl_dir / "gamma_intervention_summary.csv", index=False)
    detail.to_csv(tbl_dir / "gamma_intervention_detail.csv", index=False)
    pooled.to_csv(tbl_dir / "gamma_intervention_pooled.csv", index=False)

    md = ["# Gamma interventions on frozen E4 fusion (Phase 5)",
          "",
          "Gate zeroed tests (no retraining). Delta = score(full) - score(gate);",
          "a positive Delta means that residual gate contributes to the decision.",
          "Scores are image-max of the calibrated fused mean map (valid-depth",
          "masking). Evidence is relative anomaly evidence, not a causal claim.", ""]
    md.append("## Pooled by injected kind x severity (mean Delta)")
    md.append("")
    md.append(pooled.round(4).to_markdown(index=False))
    md.append("")
    md.append("## Per-category gates and Delta summary")
    md.append("")
    md.append(summary.round(6).to_markdown(index=False))
    md.append("")
    md.append("## Per-category x kind x severity")
    md.append("")
    md.append(detail.round(4).to_markdown(index=False))
    md += ["",
           "Note: per-sample Delta uses the image-max of the calibrated fused map;",
           "samples with full score <= 0.5 (no anomaly signal) are dropped from the",
           "mean to keep signal-to-noise interpretable.",
           "gate legend: full / gamma_rgb=0 / gamma_depth=0 / both=0."]
    (tbl_dir / "gamma_intervention_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print(pooled.round(4).to_markdown(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())