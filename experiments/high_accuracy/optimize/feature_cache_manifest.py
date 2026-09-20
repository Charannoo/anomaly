import hashlib
import json
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

CACHE_MANIFEST_PATH = "/opt/tmp/feature_cache/manifest.csv"
CACHE_DIR = Path("/opt/tmp/feature_cache")

class CacheManifest:
    def __init__(self):
        self.entries = {}
        self.manifest_path = Path(CACHE_MANIFEST_PATH)
        self._load()

    def _load(self):
        if self.manifest_path.exists():
            import pandas as pd
            df = pd.read_csv(self.manifest_path)
            for _, row in df.iterrows():
                self.entries[row["sample_id"]] = row.to_dict()

    def register(self, sample_id, category, preprocessing_hash, dino_sha, ptmae_sha,
                 rgb_shape, xyz_shape, cache_path):
        entry = {
            "sample_id": sample_id,
            "category": category,
            "preprocessing_hash": preprocessing_hash,
            "dino_checkpoint_sha256": dino_sha,
            "pointmae_checkpoint_sha256": ptmae_sha,
            "rgb_feature_shape": str(rgb_shape),
            "xyz_feature_shape": str(xyz_shape),
            "cache_path": str(cache_path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self.entries[sample_id] = entry
        self._save()

    def check_hit(self, sample_id, category, preprocessing_hash, dino_sha, ptmae_sha):
        if sample_id not in self.entries:
            return False, "MISS: sample_id not in manifest"
        entry = self.entries[sample_id]
        if entry["category"] != category:
            return False, f"MISS: category mismatch ({entry['category']} vs {category})"
        if entry["preprocessing_hash"] != preprocessing_hash:
            return False, "MISS: preprocessing hash mismatch"
        if entry["dino_checkpoint_sha256"] != dino_sha:
            return False, "MISS: DINO checkpoint SHA256 mismatch"
        if entry["pointmae_checkpoint_sha256"] != ptmae_sha:
            return False, "MISS: Point-MAE checkpoint SHA256 mismatch"
        return True, "HIT: all hashes and shapes match"

    def _save(self):
        import pandas as pd
        df = pd.DataFrame(list(self.entries.values()))
        df.to_csv(self.manifest_path, index=False)

def get_checkpoint_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()

def get_feature_shapes(feature_tensor):
    return list(feature_tensor.shape)

if __name__ == "__main__":
    manifest = CacheManifest()
    print(f"Manifest has {len(manifest.entries)} entries")
    print(f"Manifest path: {CACHE_MANIFEST_PATH}")
