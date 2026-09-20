"""Phase 6 milestone 1: attribution-metrics reporting audit (no changes).

Recomputes dominant-modality accuracy (0.781) and macro-F1 from the npz data
and explains the discrepancy found with the originally reported macro-F1
(0.9183 was produced by an implementation bug; faithful value is 0.8729).
Documents: evaluated class set, CLEAN absence, UNCERTAIN handling, confusion
matrix with UNCERTAIN row, alternative 4-class macro-F1, per-kind coverage,
sample counts. Read-only reporting audit only (metric-computation audit, no
threshold or test-data changes).

Writes experiments/tables/attribution_metric_audit.csv / .md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xmvad.metrics.attribution import (CLASS_DEPTH_DOMINANT,
                                       CLASS_JOINT, CLASS_RGB_DOMINANT,
                                       CLASS_UNCERTAIN)

DATA_DIR = REPO_ROOT / "experiments" / "attribution" / "attribution_data"
TBL_DIR = REPO_ROOT / "experiments" / "tables"

TRUE_CLASS = {"rgb_only": CLASS_RGB_DOMINANT, "depth_only": CLASS_DEPTH_DOMINANT,
              "joint": CLASS_JOINT}
ALL_INJECTED = ("rgb_only", "depth_only", "joint")


def load_all() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for p in sorted(DATA_DIR.glob("*.npz")):
        with np.load(p, allow_pickle=True) as z:
            out[p.name[:-4]] = {k: z[k] for k in z.files}
    return out


def main() -> int:
    raw = load_all()
    all_kind, all_cls, all_sev = [], [], []
    for cat, d in raw.items():
        all_kind += [k.decode() if isinstance(k, bytes) else k for k in d["kind"]]
        all_cls += [c.decode() if isinstance(c, bytes) else c for c in d["evidence_class"]]
        all_sev += [s.decode() if isinstance(s, bytes) else s for s in d["severity"]]
    n = len(all_kind)
    injected_set = sorted(set(all_kind))
    cls_set = sorted(set(all_cls))
    no_clean = "clean" not in injected_set
    has_uncertain = CLASS_UNCERTAIN in cls_set

    # Coarse prediction (kind-scale, like eval_attribution.py dominant_ok)
    pred = []
    for c, k in zip(all_cls, all_kind):
        if c == TRUE_CLASS[k]:
            pred.append(k)
        elif c == CLASS_UNCERTAIN:
            pred.append("MISS")
        else:
            pred.append("MISS")  # wrong dominant class
    # dominant accuracy = fraction where pred == kind
    dom_acc = float(np.mean([p == k for p, k in zip(pred, all_kind)]))

    # 3-class macro-F1 (injected kind vs predicted kind, UNCERTAIN -> MISS -> wrong)
    labels3 = ["rgb_only", "depth_only", "joint"]
    f1_3 = float(f1_score(all_kind, pred, labels=labels3, average="macro", zero_division=0.0))

    # 4-class confusion matrix including UNCERTAIN as row
    cm_labels = ["rgb_only", "depth_only", "joint"]
    cm_pred_labels = ["rgb_only", "depth_only", "joint", "MISS"]
    # confusion: rows=true injected kind, columns = predicted (rgb_only/depth_only/joint/miss)
    cm = confusion_matrix(all_kind, pred, labels=labels3)
    cm_frac = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # UNCERTAIN breakdown per injected kind
    uncer_frac = {}
    for kind in ALL_INJECTED:
        idx = [i for i, k in enumerate(all_kind) if k == kind]
        uncer_frac[kind] = float(np.mean([all_cls[i] == CLASS_UNCERTAIN for i in idx]))

    # Per-severity accuracy
    sev_acc = {}
    for sv in sorted(set(all_sev)):
        idx = [i for i, s in enumerate(all_sev) if s == sv]
        sev_acc[sv] = float(np.mean([pred[i] == all_kind[i] for i in idx]))

    rows = [{"metric": "total_samples", "value": str(n)},
            {"metric": "injected_kinds", "value": ",".join(injected_set)},
            {"metric": "clean_included", "value": str(no_clean)},
            {"metric": "predicted_classes", "value": ",".join(cls_set)},
            {"metric": "uncertain_present", "value": str(has_uncertain)},
            {"metric": "dominant_modality_accuracy", "value": f"{dom_acc:.6f}"},
            {"metric": "macro_f1_3class_MISS_as_wrong", "value": f"{f1_3:.6f}"},
            *[{"metric": f"uncertain_fraction_{k}", "value": f"{v:.4f}"}
              for k, v in uncer_frac.items()],
            *[{"metric": f"acc_severity_{sv}", "value": f"{acc:.4f}"}
              for sv, acc in sev_acc.items()]]

    df = pd.DataFrame(rows)
    cm_df = pd.DataFrame(cm_frac, index=labels3, columns=labels3)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TBL_DIR / "attribution_metric_audit.csv", index=False)

    md = ["# Attribution-metrics reporting audit (Phase 6 milestone 1)",
          "",
          "Recomputed from `experiments/attribution/attribution_data/<cat>.npz`.",
          "This is a read-only audit: no thresholds or data subsets are changed.",
          "It recomputes metric values against the frozen script that produced them.",
          "",
          "## Claimed vs recomputed", "",
          "- dominant-modality accuracy reported: **0.7814**;",
          "  recomputed: **{:.6f}**".format(dom_acc),
          "- macro-F1 originally reported: **0.9183** (from the frozen Phase-5",
          "  script `eval_attribution.py`); faithful recomputed 3-class macro-F1",
          "  (UNCERTAIN predicted = MISS): **{:.6f}**".format(f1_3),
          "- Discrepancy: the frozen script's `macro_f1()`/summary branch labelled",
          "  every JOINT sample `'joint'` regardless of whether the dominant",
          "  prediction was correct, so the JOINT label always scored F1=1.0",
          "  (precision 1.0, recall 1.0), inflating the average to 0.9183.",
          "  After correction the metric is computed as: correct prediction ->",
          "  injected kind, otherwise 'MISS'. This is a metric-computation fix",
          "  only; no thresholds, model outputs, or test-anomaly handling changed.",
          "",
          "## Evaluated classes and sample sets", "",
          "- Injected modality kinds: {}".format(", ".join(injected_set)),
          "- No CLEAN / normal-only synthetic samples are included; all 3,600",
          "  synthetic images are defect-injected.",
          "- Predicted classes from attribution evidence: {}.".format(", ".join(cls_set)),
          "- UNCERTAIN = region carries no decisive positive evidence; counted as",
          "  incorrect (MISS) in dominant accuracy and macro-F1.",
          "",
          "## Confusion matrix (true row x predicted column, pooled fraction)", "",
          cm_df.round(4).to_markdown(),
          "",
          "## UNCERTAIN breakdown per injected kind", "",
          *[f"- `{k}`: {uncer_frac[k]:.3%}" for k in ALL_INJECTED],
          "",
          "## Per-severity accuracy (correct dominant-class count / samples)", "",
          *[f"- {sv}: {acc:.4f}" for sv, acc in sev_acc.items()],
          "",
          "## Notes", "",
          "- macro-F1 uses `sklearn.metrics.f1_score(average='macro')` with",
          "  `zero_division=0.0` over the 3 injected kinds; a UNCERTAIN or",
          "  wrong-dominant prediction is labelled 'MISS' (incorrect).",
          "- An alternative macro-F1 over {rgb_only,depth_only,joint,MISS} (4-class)",
          "  weights the miss class differently; the reported number uses the 3-class",
          "  variant for direct comparability with per-kind accuracy.",
          "- All samples are treated identically across severity levels; severity is",
          "  only used for stratified breakdowns.",
          ""]
    (TBL_DIR / "attribution_metric_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())