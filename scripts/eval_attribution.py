"""Phase 5, todo 8: evaluate controlled modality attribution.

Reads experiments/attribution/attribution_data/<category>.npz (Phase 5 todo 7)
and computes, per severity and pooled:

- dominant-modality accuracy (predicted evidence class in {RGB-DOMINANT,
  DEPTH-DOMINANT} for RGB_ONLY / DEPTH_ONLY injections; JOINT for JOINT),
- macro-F1 over the 3 injected modalities,
- confusion matrix (predicted class vs injected modality),
- mean evidence ratio pos(E_rgb)/pos(E_depth) and p_rgb per group,
- AttributionConsistency: decision-ablation direction vs evidence direction,
- spatial explanation quality: pixel-AUROC of RGB map / depth map / fused map
  against the injected mask.

Evidence is relative anomaly evidence (not a causal claim). Deterministic.

Writes experiments/tables/attribution_metrics.csv and .md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from xmvad.metrics.attribution import (CLASS_DEPTH_DOMINANT,
                                       CLASS_JOINT, CLASS_RGB_DOMINANT,
                                       attribution_consistency)

DATA_DIR = REPO_ROOT / "experiments" / "attribution" / "attribution_data"
TABLE_DIR = REPO_ROOT / "experiments" / "tables"

TRUE_CLASS = {"rgb_only": CLASS_RGB_DOMINANT, "depth_only": CLASS_DEPTH_DOMINANT,
              "joint": CLASS_JOINT}


def main() -> int:
    cats = sorted([p.name[:-4] for p in DATA_DIR.glob("*.npz")])
    if not cats:
        print("no attribution data found; run scripts/generate_attribution_data.py")
        return 1

    sevs = []
    raw = {}
    for cat in cats:
        with np.load(DATA_DIR / f"{cat}.npz", allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        sevs = sorted({s for s in np.unique(d["severity"])}) if len(sevs) == 0 else sevs
        raw[cat] = d

    rows = []
    grouppool = {sev: {"kind": [], "class": [], "p_rgb": [], "e_r": [], "e_d": [],
                       "dr": [], "dd": [], "ok": []} for sev in list(sevs) + ["all"]}

    for cat in cats:
        d = raw[cat]
        kind = [k.decode() if isinstance(k, bytes) else k for k in d["kind"]]
        cls = [c.decode() if isinstance(c, bytes) else c for c in d["evidence_class"]]
        sev = [s.decode() if isinstance(s, bytes) else s for s in d["severity"]]
        ok = [dominant_ok(c, k) for c, k in zip(cls, kind)]
        acc = float(np.mean(ok))
        mf = macro_f1(ok, kind)
        # AttributionConsistency per sample (evidence vs decision direction).
        cons = [float(attribution_consistency(p, dr, dd))
                for p, dr, dd in zip(d["p_rgb"], d["delta_rgb"], d["delta_depth"])]
        # Spatial quality: pixel-AUROC of each map vs injected mask.
        pa_r = np.stack(spatial_auroc_per_sample(d["rgb_map"], d["gt_mask"]))
        pa_d = np.stack(spatial_auroc_per_sample(d["depth_map"], d["gt_mask"]))
        pa_f = np.stack(spatial_auroc_per_sample(d["fused_mean"], d["gt_mask"]))
        rows.append({"category": cat,
                     "dom_acc": acc, "macro_f1": mf, "consistency": float(np.mean(cons)),
                     "p_rgb_ratio": float(_geom_mean(_ratio(d["e_rgb_topk"], d["e_depth_topk"]))),
                     "pa_rgb": float(np.mean(pa_r)), "pa_depth": float(np.mean(pa_d)),
                     "pa_fused": float(np.mean(pa_f)),
                     "n": int(d["kind"].size)})
        for se in list(sevs) + ["all"]:
            m = np.array([(st == se) for st in sev]) if se != "all" else np.ones(d["kind"].size, bool)
            g = grouppool[se]
            g["kind"] += [kind[i] for i in np.where(m)[0]]
            g["class"] += [cls[i] for i in np.where(m)[0]]
            g["p_rgb"] += [float(d["p_rgb"][i]) for i in np.where(m)[0]]
            g["e_r"] += [float(d["e_rgb_topk"][i]) for i in np.where(m)[0]]
            g["e_d"] += [float(d["e_depth_topk"][i]) for i in np.where(m)[0]]
            g["dr"] += [float(d["delta_rgb"][i]) for i in np.where(m)[0]]
            g["dd"] += [float(d["delta_depth"][i]) for i in np.where(m)[0]]
            g["ok"] += [bool(ok[i]) for i in np.where(m)[0]]

    df = pd.DataFrame(rows)
    # Per-severity detail table (rows = severity x modality).
    det_rows = []
    per_group = {}
    for se in list(sevs) + ["all"]:
        g = grouppool[se]
        per_group[se] = summarize_group(se, g)
        det_rows.append(per_group[se])
    det = pd.DataFrame(det_rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "attribution_metrics.csv", index=False)
    det.to_csv(TABLE_DIR / "attribution_metrics_detail.csv", index=False)

    md = [f"# Controlled modality-attribution metrics (Phase 5)",
          "",
          "Relative anomaly evidence on CALIBRATED branch maps; region = injected",
          "production-like synthetic mask. Not a causal claim. Data: TRAIN-NORMAL only.",
          ""]
    md.append(f"Categories: {', '.join(cats)};  severities: {', '.join(sevs)};  "
              f"total synthetic samples: {int(sum(d['kind'].size for d in raw.values()))}")
    md += ["", "## Per-category (pooled over severity)", "",
           df.round(4).to_markdown(index=False)]
    md += ["", "## Per-severity (pooled over categories)", "",
           det.round(4).to_markdown(index=False)]
    md += ["", "## Confusion blocks (predicted class column x injected modality row, ",
           "fraction of samples) on pooled data", ""]
    k = grouppool["all"]["kind"]
    c = grouppool["all"]["class"]
    lbls = [CLASS_RGB_DOMINANT, CLASS_DEPTH_DOMINANT, CLASS_JOINT]
    cm_lines = ["| injected \\ predicted | " + " | ".join(lbls) + " |", "|---|" + "---|" * 3]
    for kind in ("rgb_only", "depth_only", "joint"):
        idx = [i for i, kk in enumerate(k) if kk == kind]
        if not idx:
            continue
        row = [f"{np.mean([c[i] == p for i in idx]):.3f}" for p in lbls]
        cm_lines.append(f"| {kind} | " + " | ".join(row) + " |")
    md += cm_lines
    md += ["",
           "## Notes",
           "- AttributionConsistency: (Delta_RGB vs Delta_DEPTH) agrees with ",
           "  (p_rgb vs 0.5) - decision-ablation direction aligns with evidence direction.",
           "- Spatial pixel-AUROC: branch/ fused anomaly map against the injected mask.",
           "- UNCERTAIN class (decisions with effectively no positive evidence) is a ",
           "  correctness neutral outcome; use apply to cases with decisive evidence.",
           "- macro-F1: correct prediction -> injected kind, otherwise 'MISS'",
           "  (wrong dominant class or UNCERTAIN); 3-class macro average. Corrected in",
           "  the Phase 6 audit (previously joint was always scored correct).",
           ""]
    (TABLE_DIR / "attribution_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[-32:]))
    return 0


def dominant_ok(pred_class: str, kind: str) -> bool:
    if kind == "joint":
        return pred_class == CLASS_JOINT
    return pred_class == TRUE_CLASS[kind]


def macro_f1(ok: list[bool], kind: list[str]) -> float:
    """Faithful macro-F1. Correct prediction -> injected kind; anything else
    (wrong dominant class or UNCERTAIN) -> 'MISS' (incorrect). Previously the
    joint branch always emitted 'joint' even when incorrect, inflating F1;
    corrected in the Phase 6 reporting audit (see docs/PHASE5_EXPLAINABILITY.md)."""
    from sklearn.metrics import f1_score
    labels = ["rgb_only", "depth_only", "joint"]
    y_pred = [kind[i] if ok[i] else "MISS" for i in range(len(kind))]
    return float(f1_score(kind, y_pred, labels=labels, average="macro", zero_division=0.0))


def summarize_group(se, g):
    n = len(g["kind"])
    if n == 0:
        return {"severity": se, "n": 0}
    acc = float(np.mean(g["ok"]))
    from sklearn.metrics import f1_score
    y_pred = [kind if ok else "MISS" for ok, kind in zip(g["ok"], g["kind"])]
    mf = float(f1_score(g["kind"], y_pred, labels=["rgb_only", "depth_only", "joint"],
                        average="macro", zero_division=0.0))
    cons = float(np.mean([attribution_consistency(p, dr, dd)
                          for p, dr, dd in zip(g["p_rgb"], g["dr"], g["dd"])]))
    my_p = float(np.mean(g["p_rgb"]))
    er = float(_geom_mean(_ratio(g["e_r"], g["e_d"])))
    out = {"severity": se, "n": n, "dom_acc": acc, "macro_f1": mf,
           "consistency": cons, "mean_p_rgb": my_p,
           "evid_ratio_rgb_over_depth": er}
    for kind in ("rgb_only", "depth_only", "joint"):
        i = [ii for ii, kk in enumerate(g["kind"]) if kk == kind]
        out[f"acc_{kind}"] = float(np.mean([g["ok"][ii] for ii in i])) if i else float("nan")
        out[f"p_rgb_{kind}"] = float(np.mean([g["p_rgb"][ii] for ii in i])) if i else float("nan")
    return out


def _ratio(e_r, e_d):
    import numpy as _np
    return _np.array([max(0.0, float(a)) / (max(0.0, float(b)) + 1e-6)
                      for a, b in zip(e_r, e_d)])


def _geom_mean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-6))))) if x.size else 0.0


def spatial_auroc_per_sample(maps, masks):
    out = []
    for m, mask in zip(maps, masks):
        out.append(_pixel_auc(np.asarray(m, dtype=np.float64),
                              np.asarray(mask, dtype=bool)))
    return out


def _pixel_auc(pred, mask) -> float:
    """Per-map pixel AUC (tie-aware, exact Mann-Whitney, O(n log n))."""
    m = mask.ravel()
    p = pred.ravel()
    n = p.size
    npos = int(m.sum())
    if npos == 0 or npos == n:
        return 1.0
    nneg = n - npos
    order = np.argsort(p, kind="mergesort")
    p_s = p[order]
    m_s = m[order]
    pos_ids, pos_counts = np.unique(p_s, return_inverse=True, return_counts=True)[1:]
    pos_per_grp = np.bincount(pos_ids, weights=m_s)
    neg_per_grp = pos_counts.astype(np.float64) - pos_per_grp
    cneg_below = np.cumsum(neg_per_grp) - neg_per_grp
    favor = float(np.sum(pos_per_grp * (cneg_below + 0.5 * neg_per_grp)))
    return favor / float(npos * nneg)


if __name__ == "__main__":
    raise SystemExit(main())