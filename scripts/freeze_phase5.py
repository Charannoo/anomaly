"""Phase 6 milestone 0: freeze Phase 5.

Records SHA-256 hashes of the 10 frozen E4 checkpoints, stores the
environment snapshot, and verifies the Phase 5 metric tables exist.
Deterministic; no model data is loaded.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "checkpoints" / "E4_concat_fusion"
ART_DIR = REPO_ROOT / "experiments" / "artifacts"
TABLES = [
    "E4_concat_metrics_native.csv", "E4_official_metric_verification.csv",
    "paper_baseline_table.csv", "E4_concat_calibration.json",
    "attribution_metrics.csv", "attribution_metrics_detail.csv",
    "gamma_intervention_summary.csv", "gamma_intervention_detail.csv",
    "gamma_intervention_pooled.csv", "real_evidence_metrics.csv",
    "real_evidence_summary.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import torch
    rows = []
    missing = []
    for pt in sorted(CKPT_DIR.glob("*_best.pt")):
        if not pt.is_file():
            missing.append(pt.name)
            continue
        rows.append({"category": pt.name.split(".")[0], "file": str(pt.relative_to(REPO_ROOT)),
                     "bytes": pt.stat().st_size, "sha256": sha256(pt)})
    if missing:
        print("MISSING checkpoints:", missing)
    if not rows:
        print("no E4 checkpoints found")
        return 1

    table_dir = REPO_ROOT / "experiments" / "tables"
    table_status = {t: (table_dir / t).is_file() for t in TABLES}
    if not all(table_status.values()):
        print("MISSING TABLES:", [t for t, ok in table_status.items() if not ok])

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "onnx": __import__("onnx").__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "openvino": __import__("openvino").__version__,
        "nncf": __import__("nncf").__version__,
        "numpy": __import__("numpy").__version__,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    import csv
    with open(ART_DIR / "e4_checkpoint_hashes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "file", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} hashes -> experiments/artifacts/e4_checkpoint_hashes.csv")
    print("environment -> experiments/artifacts/environment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())