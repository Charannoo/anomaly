"""H0.2 - Hash the frozen XMV-AD Lite research artifacts before XMV-AD-H work begins.

The original XMV-AD Lite track is COMPLETE and FROZEN. Before any high-accuracy
(XMV-AD-H) development we record full SHA256 fingerprints of the artifacts that
define the frozen track so that later phases can verify nothing was altered.

Outputs:
  experiments/high_accuracy/frozen_artifact_manifest.csv
  experiments/high_accuracy/frozen_artifact_manifest.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FROZEN_GLOBS = [
    "checkpoints/**",
    "deployment/**",
    "experiments/tables/*.csv",
    "experiments/tables/*.md",
    "experiments/tables/*.png",
    "experiments/tables/*.json",
    "docs/*.md",
    "configs/*.yaml",
    "src/**/*.py",
    "scripts/*.py",
    "tests/*.py",
    "pyproject.toml",
    "requirements.txt",
]

EXCLUDE_DIRS = {
    "checkpoints/E1_unit_tests_fixtures",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "experiments" / "high_accuracy"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pattern in FROZEN_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            stat = path.stat()
            rows.append(
                {
                    "relative_path": rel,
                    "bytes": stat.st_size,
                    "sha256": sha256_file(path),
                }
            )

    csv_path = out_dir / "frozen_artifact_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("relative_path,bytes,sha256\n")
        for r in rows:
            f.write(f"{r['relative_path']},{r['bytes']},{r['sha256']}\n")

    summary = {
        "frozen_at": None,  # filled by caller if needed
        "file_count": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "manifest_csv": csv_path.name,
    }
    with open(out_dir / "frozen_artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"hashed {len(rows)} files, {summary['total_bytes']} bytes total")
    print(f"manifest: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()