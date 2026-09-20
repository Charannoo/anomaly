"""Phase 5, todo 10: real-defect evidence analysis + failure-case discovery.

Computes modality evidence on REAL test defects using stored E4 predictions
(experiments/predictions/E4_concat_fusion/<cat>/<sample_id>.npz, maps at 256).
Evidence is evaluated inside the real GT region R (resized GT in the npz),
on CALIBRATED branch maps.

Outputs (deterministic):
- experiments/tables/real_evidence_metrics.csv  (one row per defect image)
- experiments/tables/real_evidence_summary.csv  (per-category + pooled)
- experiments/figures/explainability_failures/candidates.csv
  (top hardest samples per category: strong GT signal, weak E4 score)

Evidence = relative anomaly evidence, not a causal claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from xmvad.fusion.score_fusion import calibrate
from xmvad.metrics.attribution import (CLASS_DEPTH_DOMINANT,
                                       CLASS_JOINT, CLASS_RGB_DOMINANT,
                                       CLASS_UNCERTAIN, evidence_class,
                                       region_evidence, relative_evidence)
from xmvad.metrics.scores import variant_scores

PRED_DIR = REPO_ROOT / "experiments" / "predictions" / "E4_concat_fusion"
TBL_DIR = REPO_ROOT / "experiments" / "tables"
FIG_DIR = REPO_ROOT / "experiments" / "figures" / "explainability_failures"
CALIB = json.load(open(REPO_ROOT / "experiments" / "tables" / "E4_concat_calibration.json",
                       encoding="utf-8"))["categories"]


def load_all() -> list[tuple[str, str, dict[str, np.ndarray]]]:
    out = []
    for cat_dir in sorted(PRED_DIR.glob("*")):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        for npz in sorted(cat_dir.glob("*.npz")):
            with np.load(npz, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            sid = str(d["sample_id"])
            out.append((cat, sid, d))
    return out


def main() -> int:
    rows = []
    for cat, sid, d in load_all():
        defect = str(d["defect"])
        cm = CALIB[cat]
        med_r, sc_r = cm["rgb"]["median"], cm["rgb"]["scale"]
        med_d, sc_d = cm["depth"]["median"], cm["depth"]["scale"]
        zr = calibrate(np.asarray(d["rgb_map"], dtype=np.float64), med_r, sc_r)
        zd = calibrate(np.asarray(d["depth_map"], dtype=np.float64), med_d, sc_d)
        gt = np.asarray(d["gt"], dtype=bool)
        e_r_topk = region_evidence(zr, gt, mode="topk", frac=0.01)
        e_d_topk = region_evidence(zd, gt, mode="topk", frac=0.01)
        e_r_mean = region_evidence(zr, gt, mode="mean")
        e_d_mean = region_evidence(zd, gt, mode="mean")
        p_rgb, _ = relative_evidence(e_r_topk, e_d_topk)
        cls = evidence_class(e_r_topk, e_d_topk)
        s_full = variant_scores(np.asarray(d["map_mean"], dtype=np.float64), None)["max"]
        rows.append({"category": cat, "defect": defect, "sample_id": sid,
                     "gt_area": float(gt.mean()) if gt.size else 0.0, "label": int(d["label"]),
                     "s_full": s_full, "p_rgb": p_rgb, "evidence_class": cls,
                     "e_rgb_topk": e_r_topk, "e_depth_topk": e_d_topk,
                     "e_rgb_mean": e_r_mean, "e_depth_mean": e_d_mean})
    frame = pd.DataFrame(rows).sort_values(["category", "sample_id"]).reset_index(drop=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TBL_DIR / "real_evidence_metrics.csv", index=False)

    # Per-category summary: class mix + separation quality.
    summ = frame.groupby("category").apply(lambda g: pd.Series({
        "n": len(g),
        "mean_p_rgb": float(np.mean(g["p_rgb"])),
        "frac_rgb_dom": float(np.mean(g["evidence_class"] == CLASS_RGB_DOMINANT)),
        "frac_depth_dom": float(np.mean(g["evidence_class"] == CLASS_DEPTH_DOMINANT)),
        "frac_joint": float(np.mean(g["evidence_class"] == CLASS_JOINT)),
        "frac_uncertain": float(np.mean(g["evidence_class"] == CLASS_UNCERTAIN)),
        "s_full_mean": float(np.mean(g["s_full"])),
        "s_full_min": float(np.min(g["s_full"])),
        "large_gt_frac": float(np.mean(g["gt_area"] > 0.01)),
    }), include_groups=False).reset_index()
    summ.to_csv(TBL_DIR / "real_evidence_summary.csv", index=False)

    # Failure/discovery candidates: ranks by low score among large-GT defects.
    big = frame[frame["gt_area"] > 0.01].copy()
    big["rank_score"] = big.groupby("category")["s_full"].rank(ascending=True)
    cand = big[big["rank_score"] <= 3].sort_values(["category", "rank_score"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cand.to_csv(FIG_DIR / "candidates.csv", index=False)

    # Also strongest modality-dominant real examples for the operator figure set.
    rgb_ex = frame.loc[frame[frame["gt_area"] > 0.005]
                       .groupby("category")["p_rgb"].idxmax()].drop_duplicates("category")
    depth_ex = frame.loc[frame[frame["gt_area"] > 0.005]
                         .groupby("category")["p_rgb"].idxmin()].drop_duplicates("category")
    rgb_ex.to_csv(FIG_DIR / "rgb_dominant_candidates.csv", index=False)
    depth_ex.to_csv(FIG_DIR / "depth_dominant_candidates.csv", index=False)

    md = ["# Real-defect evidence analysis (Phase 5, todo 10)",
          "",
          "Evidence on CALIBRATED branch maps inside the real GT region (256-res GT",
          "stored with E4 predictions). Relative anomaly evidence, not causal.",
          ""]
    md.append(f"Images analysed: {len(frame):d} across {frame['category'].nunique()} categories.")
    md += ["", "## Per-category summary", "",
           summ.round(4).to_markdown(index=False)]
    md += ["", "## Pooled", ""]
    pooled = {"n": len(frame),
              "mean_p_rgb": float(frame["p_rgb"].mean()),
              "frac_rgb_dom": float(np.mean(frame["evidence_class"] == CLASS_RGB_DOMINANT)),
              "frac_depth_dom": float(np.mean(frame["evidence_class"] == CLASS_DEPTH_DOMINANT)),
              "frac_joint": float(np.mean(frame["evidence_class"] == CLASS_JOINT)),
              "frac_uncertain": float(np.mean(frame["evidence_class"] == CLASS_UNCERTAIN)),
              "s_full_mean": float(frame["s_full"].mean())}
    md.append(pd.DataFrame([pooled]).round(4).to_markdown(index=False))
    md += ["", "## Failure/discovery candidates (low E4 score, large GT)",
           "", cand.round(4).to_markdown(index=False)]
    md += ["", "## Strongest real RGB-dominant / depth-dominant examples (per category)",
           "", "### RGB-dominant",
           "", rgb_ex.round(4).to_markdown(index=False),
           "", "### Depth-dominant",
           "", depth_ex.round(4).to_markdown(index=False)]
    (TBL_DIR / "real_evidence.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())