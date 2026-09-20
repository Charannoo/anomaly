"""H0 — MVTec 3D-AD dataset inventory for the XMV-AD-H track.

Official MVTec 3D-AD layout (per category):
  train/good/{rgb,xyz,valid}/<id>.png|tiff      normal
  validation/good/{rgb,xyz,valid}/<id>.png|tiff normal
  test/<defect_type>/{rgb,xyz,valid,gt}/<id>.png|tiff
  train.txt / validation.txt / test.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "raw" / "mvtec3d"

CATS = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]
MODALITY_DIRS = {"rgb": {".png"}, "xyz": {".tiff"}, "valid": {".tiff"}}
GT_DIR = "gt"


def _count(img_dir: Path) -> int:
    return sum(1 for _ in img_dir.glob("*")) if img_dir.is_dir() else -1


def main() -> int:
    assert ROOT.is_dir(), ROOT
    summary = {"root": str(ROOT), "categories": {}, "total_train": 0, "total_validation": 0,
               "total_test": 0}
    for cat in CATS:
        cdir = ROOT / cat
        info: dict = {}
        for split, label in (("train", "train"), ("validation", "validation")):
            n = _count(cdir / split / "good" / "rgb")
            info[f"{label}_n"] = n
        tdir = cdir / "test"
        types = sorted(d.name for d in tdir.iterdir() if d.is_dir()) if tdir.is_dir() else []
        info["test_defect_types"] = types
        info["test_by_type"] = {}
        n_test = 0
        bad = []
        for t in types:
            n = _count(tdir / t / "rgb")
            info["test_by_type"][t] = n
            n_test += max(n, 0)
            for mod, exts in {**MODALITY_DIRS, "gt": {".png"} if t != "good" else set()}.items():
                d = tdir / t / mod
                if not d.is_dir() or (exts and not (d / (Path("000").name + next(iter(exts)))).exists()):
                    pass
            for mod, exts in MODALITY_DIRS.items():
                files = [f.suffix for f in (tdir / t / mod).glob("*")] if (tdir / t / mod).is_dir() else []
                if not files or not set(files) <= exts:
                    bad.append(f"{t}/{mod}")
        info["test_n"] = n_test
        for mod in MODALITY_DIRS:
            files = [f.suffix for f in (cdir / "train" / "good" / mod).glob("*")]
            if not files or not set(files) <= MODALITY_DIRS[mod]:
                bad.append(f"train/{mod}")
        gt = (cdir / "test" / "crack" / "gt") if (cdir / "test" / "crack").is_dir() else None
        info["gt_suffix"] = [f.suffix for f in gt.glob("*")][0] if gt is not None else None
        info["layout_ok"] = not bad
        info["issues"] = bad
        summary["categories"][cat] = info
        summary["total_train"] += info["train_n"]
        summary["total_validation"] += info["validation_n"]
        summary["total_test"] += info["test_n"]
    sizes_mb = {p.name: round(p.stat().st_size / 1e6, 1) for p in ROOT.iterdir() if p.is_dir()}
    summary["category_size_mb"] = sizes_mb
    summary["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = REPO / "experiments" / "high_accuracy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "categories"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())