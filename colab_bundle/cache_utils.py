#!/usr/bin/env python3
"""Shared helpers for the XMV-AD-H Colab bundle.

Reuses algorithms from the local repo (third_party/m3dm + scripts/high_accuracy)
but is fully self-contained so it runs on a fresh Colab runtime:
  * path/config handling (Colab vs local),
  * dataset sample enumeration (canonical MVTec-3D-AD layout),
  * shard cache + cache_manifest.json,
  * coreset selection (same greedy routine the repo uses),
  * exact nearest-neighbour evaluation (FAISS IndexFlatL2 with a chunked
    PyTorch cdist fallback) + PatchCore-style reweighted score and map.
"""

import os
import sys
import json
import time
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ----------------------------------------------------------------------
# Environment helpers
# ----------------------------------------------------------------------

def is_colab():
    return os.path.exists("/content")


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def ensure_dirs(paths):
    for p in paths:
        ensure_dir(p)
    return paths


def torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def fmt_duration(seconds):
    seconds = int(seconds)
    return "{:02d}:{:02d}:{:02d}".format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def hash_file_sha256(path, block_size=65536, want=16):
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:want]


def local_time():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DEFAULT_CONFIG_NAME = "config_colab.yaml"

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]


def _rewrite_paths_for_local(cfg, base):
    """When not running inside Colab, point every path at the bundle dir."""
    work = os.path.join(base, "work_local")
    cfg["bundle_root"] = base
    cfg["work_root"] = work
    cfg["data_root"] = os.path.join(work, "data")
    cfg["dataset_dir"] = os.path.join(work, "data", "mvtec3d")
    cfg["checkpoints_dir"] = os.path.join(work, "checkpoints")
    cfg["features_dir"] = os.path.join(work, "features")
    cfg["results_dir"] = os.path.join(base, "results")
    cfg["logs_dir"] = os.path.join(base, "logs")
    cfg["drive_root"] = os.path.join(base, "drive_mirror")
    cfg["persistent"] = {
        "dataset": os.path.join(work, "data"),
        "checkpoints": os.path.join(work, "checkpoints"),
        "features": os.path.join(work, "features"),
        "results": os.path.join(base, "results"),
        "logs": os.path.join(base, "logs"),
    }
    return cfg


def load_config(path=None):
    env_root = os.environ.get("XMVAD_BUNDLE_ROOT")
    if env_root:
        base = os.path.abspath(env_root)
        path = path or os.path.join(base, DEFAULT_CONFIG_NAME)
    elif path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, DEFAULT_CONFIG_NAME)

    if not os.path.exists(path):
        raise FileNotFoundError("config file not found: %s" % path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if env_root:
        _rewrite_paths_for_local(cfg, base)

    if not is_colab():
        local_base = os.path.dirname(os.path.abspath(path))
        cfg = _rewrite_paths_for_local(cfg, local_base)

    cfg.setdefault("categories", CATEGORIES)
    return cfg


def cfg_get(cfg, key, default=None):
    return cfg.get(key, default)


# ----------------------------------------------------------------------
# Dataset sample enumeration
# ----------------------------------------------------------------------

def _pair_sample(rgb_p, tiff_p, gt_p, defect, label, idx):
    stem = os.path.splitext(os.path.basename(rgb_p))[0]
    name = "%s_%s" % (defect, stem) if defect != "good" else stem
    return {
        "name": name,
        "defect": defect,
        "rgb": rgb_p,
        "xyz": tiff_p,
        "gt": gt_p,
        "label": label,
    }


def list_paths_split(category_dir, split):
    split_dir = os.path.join(category_dir, split)
    if not os.path.isdir(split_dir):
        return None
    return split_dir


def list_category_samples(data_root, category, include_combined=True):
    """Enumerate canonical MVTec-3D-AD layout.

    train/good/{rgb,xyz} -> label 0
    test/<defect>/{rgb,xyz,gt} -> label 0 for good, 1 otherwise.
    test/combined/ is included by default to match the M3DM loader protocol.
    """
    category_dir = os.path.join(data_root, category)
    out = {"train": [], "test": []}

    train_dir = list_paths_split(category_dir, "train")
    if train_dir is not None:
        good_dir = os.path.join(train_dir, "good")
        rgbs = sorted(__glob(good_dir, "rgb", "*.png"))
        tiffs = sorted(__glob(good_dir, "xyz", "*.tiff"))
        for i in range(min(len(rgbs), len(tiffs))):
            out["train"].append(_pair_sample(rgbs[i], tiffs[i], None, "good", 0, i))

    test_dir = list_paths_split(category_dir, "test")
    if test_dir is not None:
        for defect in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect)
            if not os.path.isdir(defect_dir):
                continue
            if defect == "combined" and not include_combined:
                continue
            label = 0 if defect == "good" else 1
            rgbs = sorted(__glob(defect_dir, "rgb", "*.png"))
            tiffs = sorted(__glob(defect_dir, "xyz", "*.tiff"))
            gts = sorted(__glob(defect_dir, "gt", "*.png")) if label == 1 else []
            for i in range(min(len(rgbs), len(tiffs))):
                gt_p = gts[i] if i < len(gts) else None
                out["test"].append(_pair_sample(rgbs[i], tiffs[i], gt_p, defect, label, i))
    return out


def __glob(base, sub, pattern):
    path = os.path.join(base, sub, pattern)
    return __import__("glob").glob(path)


def list_all_samples(data_root, categories, include_combined=True):
    counts = {}
    for cat in categories:
        s = list_category_samples(data_root, cat, include_combined)
        counts[cat] = {"train": len(s["train"]), "test": len(s["test"])}
    return counts


# ----------------------------------------------------------------------
# Shard cache + manifest
# ----------------------------------------------------------------------

def shard_filename(category, split, modality):
    return "%s_%s_%s.pt" % (category, split, modality)


def shard_path(features_dir, category, split, modality):
    return os.path.join(features_dir, shard_filename(category, split, modality))


MANIFEST_NAME = "cache_manifest.json"


def manifest_path(features_dir):
    return os.path.join(features_dir, MANIFEST_NAME)


class CacheManifest(object):
    def __init__(self, features_dir, checkpoints_dir=None):
        self.features_dir = features_dir
        self.path = manifest_path(features_dir)
        self.checkpoints_dir = checkpoints_dir
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {"created": local_time(), "features": {}}

    def save(self):
        ensure_dir(self.features_dir)
        self.data["updated"] = local_time()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    def _entry(self, category, split, modality, create=False):
        feats = self.data.setdefault("features", {})
        cat = feats.setdefault(category, {})
        sp = cat.setdefault(split, {})
        if create:
            return sp.setdefault(modality, {})
        return sp.get(modality, {})

    def is_complete(self, category, split, modality):
        e = self._entry(category, split, modality)
        return bool(e.get("complete", False))

    def set_complete(self, category, split, modality, meta):
        e = self._entry(category, split, modality, create=True)
        e.update(meta or {})
        e.update({"complete": True, "updated": local_time()})
        self.save()

    def status(self, category, split, modality):
        e = self._entry(category, split, modality)
        return e.get("complete", False)


def save_shard(shard, path):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    torch.save(shard, tmp)
    os.replace(tmp, path)


def load_shard(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def make_shard(category, split, modality, descriptors, per_sample, labels,
               sample_names, gt_list, grid, ckpt, precision, extra=None):
    n = len(sample_names)
    offsets = []
    pos = 0
    for i in range(n):
        a = pos
        b = pos + per_sample
        offsets.append((a, b))
        pos = b
    shard = {
        "category": category,
        "split": split,
        "modality": modality,
        "descriptors": descriptors,
        "per_sample": per_sample,
        "offsets": offsets,
        "labels": labels,
        "sample_names": sample_names,
        "grid": list(grid),
        "checkpoint": ckpt or "unknown",
        "precision": precision,
        "created": local_time(),
    }
    if modality != "xyz":
        pass
    if gt_list is not None:
        shard["gt"] = gt_list
    if extra:
        shard.update(extra)
    return shard


def shard_to_samples(shard):
    offsets = shard["offsets"]
    desc = shard["descriptors"]
    labels = shard["labels"]
    names = shard["sample_names"]
    gts = shard.get("gt") or [None] * len(labels)
    out = []
    for i, (a, b) in enumerate(offsets):
        out.append({
            "name": names[i],
            "desc": desc[a:b],
            "label": int(labels[i]),
            "gt": gts[i],
            "grid": tuple(shard["grid"]),
        })
    return out


def align_rgb_descriptors(desc):
    """(N*784, D) -> (N*3136, D) with the repo's canonical per-sample layout.

    Matches scripts/high_accuracy/extract_canonical_features.py exactly:
        rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(3136, -1)
    For a single sample this yields  out[c*784 + a] = patch[(4*a + c) % 784].
    This exact row order is what the repo's combined (H3) bank used, so it must
    be reproduced bit-for-bit for H3C alignment with the XYZ 56x56 grid.
    """
    n = desc.shape[0] // 784
    D = desc.shape[1]
    x = desc.view(n, 784, D).permute(1, 0, 2)          # (784, n, D)
    x = x.repeat(4, 1, 1)                              # copy-major (3136, n, D)
    perm = torch.arange(3136, dtype=torch.long, device=x.device)
    a = perm % 784
    c = perm // 784
    perm = (4 * a + c) % 784
    x = x[perm]                                        # (3136, n, D)
    return x.permute(1, 0, 2).reshape(n * 3136, D)


# ----------------------------------------------------------------------
# Coreset (greedy, same routine as run_cached_baselines.py)
# ----------------------------------------------------------------------

def coreset_indices(z_lib, f_coreset, coreset_eps, random_state, tag, cache_dir,
                    device="cuda"):
    """Return coreset indices into z_lib (a CPU torch tensor (N, D)).

    f_coreset <= 0 returns None (use the full train bank).
    """
    N, D = z_lib.shape
    if f_coreset is None or float(f_coreset) <= 0:
        return None
    # Cap coreset items to 15,000 for high speed and memory safety
    n = max(1, int(float(f_coreset) * N))
    n = min(n, 15000)
    if n >= N:
        return None

    cache_dir = ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, "coreset_%s_%d_%d_%d.pt" % (tag, N, D, n))
    if os.path.exists(cache_path):
        print("   [CORESET CACHE HIT] %s (%d, %d) -> %d indices" % (tag, N, D, n))
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("   [CORESET FIT] %s (%d, %d) -> selecting %d items..." % (tag, N, D, n))
    dev = device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32

    # Fast Gaussian Random Projection in PyTorch
    proj_d = min(D, 128)
    gen = torch.Generator(device="cpu").manual_seed(int(random_state or 42))
    R = torch.randn(D, proj_d, generator=gen, dtype=torch.float32) / math.sqrt(proj_d)
    R = R.to(dev, dtype=dtype)

    CHUNK = 50_000
    parts = []
    with torch.inference_mode():
        for start in range(0, N, CHUNK):
            seg = z_lib[start:start + CHUNK].to(dev, dtype=dtype)
            parts.append(seg @ R)
    proj_lib = torch.cat(parts, dim=0)

    coreset_idx = [0]
    last_item = proj_lib[0:1]
    min_dist_sq = ((proj_lib - last_item) ** 2).sum(dim=1, keepdim=True)

    with torch.inference_mode():
        for it in range(n - 1):
            dist_sq = ((proj_lib - last_item) ** 2).sum(dim=1, keepdim=True)
            min_dist_sq = torch.minimum(dist_sq, min_dist_sq)
            select_idx = int(torch.argmax(min_dist_sq))
            last_item = proj_lib[select_idx:select_idx + 1]
            min_dist_sq[select_idx] = 0
            coreset_idx.append(select_idx)

    result = torch.tensor(coreset_idx, dtype=torch.long)
    torch.save(result, cache_path)
    print("   [CORESET DONE] %s -> %d indices saved" % (tag, result.shape[0]))
    return result


# ----------------------------------------------------------------------
# Exact nearest neighbour (FAISS IndexFlatL2 preferred, torch fallback)
# ----------------------------------------------------------------------

class _TorchFlatIndex(object):
    def __init__(self, bank):
        self.bank = bank

    def search(self, query_np, k, chunk=8192):
        dev = torch_device()
        q = torch.from_numpy(np.ascontiguousarray(query_np)).to(dev)
        bank = self.bank.to(dev)
        parts_d, parts_i = [], []
        with torch.inference_mode():
            for s in range(0, q.shape[0], chunk):
                sub = q[s:s + chunk]
                dist = torch.cdist(sub, bank)
                vals, idxs = torch.topk(dist, k=k, largest=False, dim=1)
                parts_d.append(vals.cpu().numpy())
                parts_i.append(idxs.cpu().numpy())
        return np.concatenate(parts_d, axis=0), np.concatenate(parts_i, axis=0)


def build_faiss_index():
    """Return a callable that builds an exact-L2 index, or None if unavailable.

    Prefers a GPU IndexFlatL2 (exact, same results as CPU IndexFlatL2) because
    scoring millions of patch rows against the banks is far faster there.
    """
    try:
        import faiss
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()

            def build_gpu(bank_np):
                cfg_idx = faiss.GpuIndexFlatConfig()
                cfg_idx.device = 0
                cfg_idx.useFloat16CoarseQuantizer = False
                return faiss.GpuIndexFlatL2(res, bank_np.shape[1], cfg_idx)

            return build_gpu
        return lambda bank_np: faiss.IndexFlatL2(bank_np.shape[1])
    except Exception:
        return None


def make_memory_index(bank, prefer_faiss=True):
    """Build an exact-L2 index over `bank` (CPU torch or numpy fp32)."""
    if isinstance(bank, torch.Tensor):
        bank_np = bank.detach().cpu().numpy().astype(np.float32)
    else:
        bank_np = np.asarray(bank, dtype=np.float32)
    index = None
    if prefer_faiss:
        builder = build_faiss_index()
        if builder is not None:
            try:
                index = builder(bank_np)
                index.add(bank_np)
                return index, bank_np
            except Exception:
                index = None
    return _TorchFlatIndex(torch.from_numpy(bank_np)), bank_np


def rank_patch_anomaly(patch, bank_np, index, grid, n_reweight=3, blur=None,
                       device="cuda"):
    """PatchCore-style reweighted image score + 224x224 anomaly map.

    Exact nearest neighbour (IndexFlatL2 or chunked torch cdist), never an
    approximate index.  Mirrors compute_patch_anomaly_map() in
    scripts/high_accuracy/run_cached_baselines.py.
    """
    patch = patch.to(device).float()
    bank = torch.from_numpy(np.ascontiguousarray(bank_np)).to(device).float()
    with torch.inference_mode():
        if index is not None and not isinstance(index, _TorchFlatIndex):
            dists, idx = index.search(patch.cpu().numpy().astype(np.float32), 1)
            min_val = torch.from_numpy(dists[:, 0]).to(device)
            min_idx = torch.from_numpy(idx[:, 0]).long().to(device)
        else:
            q = patch
            dist_full = torch.cdist(q, bank)
            min_val, min_idx = torch.min(dist_full, dim=1)

        s_idx = int(torch.argmax(min_val))
        s_star = min_val[s_idx].float()

        m_test = patch[s_idx:s_idx + 1]
        m_star = bank[min_idx[s_idx]:min_idx[s_idx] + 1]

        w_dist = torch.linalg.norm(bank - m_star, dim=1)          # (bank,)
        _, nn_idx = torch.topk(w_dist, k=max(2, n_reweight + 1), largest=False)
        m_star_knn = torch.linalg.norm(m_test - bank[nn_idx[1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1], device=device, dtype=torch.float32))
        w = 1.0 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = float(w * s_star)

        s_map = min_val.view(1, 1, int(grid[0]), int(grid[1])).float()
        s_map = F.interpolate(s_map, size=(224, 224), mode="bilinear", align_corners=False)
        if blur is not None:
            s_map = blur(s_map)
        m = s_map.squeeze(0).squeeze(0).cpu().numpy()
    return s, m


def normalize_and_bank(train_desc, cfg, tag, cache_root=None):
    """Normalize train descriptors (M3DM run_coreset style) and build a bank."""
    mean = torch.mean(train_desc.float())
    std = torch.std(train_desc.float())

    f_coreset = float(cfg_get(cfg, "f_coreset", 0.1))
    eps = float(cfg_get(cfg, "coreset_eps", 0.9))
    idx = None
    if f_coreset > 0:
        cache_dir = os.path.join(cache_root, "coreset_cache") if cache_root else None
        idx = coreset_indices(train_desc, f_coreset, eps, 42, tag, cache_dir,
                              device=torch_device())
        if idx is not None:
            sub = train_desc[idx.long()].float()
            train_bank = (sub - mean) / (std + 1e-7)
        else:
            train_bank = (train_desc.float() - mean) / (std + 1e-7)
    else:
        train_bank = (train_desc.float() - mean) / (std + 1e-7)

    stats = {"mean": mean, "std": std}
    return train_bank, stats


def score_samples(samples, bank_np, index, stats, grid, blur, cfg):
    device = torch_device()
    n_reweight = int(cfg_get(cfg, "n_reweight", 3))
    scores, maps, labels, names = [], [], [], []
    for s in samples:
        p = (s["desc"].float() - stats["mean"]) / (stats["std"] + 1e-7)
        sc, m = rank_patch_anomaly(p, bank_np, index, grid, n_reweight, blur, device)
        scores.append(sc)
        maps.append(m)
        labels.append(s["label"])
        names.append(s["name"])
    return scores, maps, labels, names


def load_blur():
    from PIL import ImageFilter
    from torchvision import transforms

    class KNNGaussianBlur(torch.nn.Module):
        def __init__(self, radius=4):
            super().__init__()
            self.radius = radius
            self.unload = transforms.ToPILImage()
            self.load = transforms.ToTensor()
            self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

        def __call__(self, img):
            map_max = img.max()
            final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel))
            final_map = final_map.to(device=img.device, dtype=img.dtype) * map_max
            return final_map

    return KNNGaussianBlur(4)


# ----------------------------------------------------------------------
# Single-bank evaluation (used by run_h1 / run_h2 / run_h3)
# ----------------------------------------------------------------------

def load_split_samples(features_dir, category, split, modality):
    """Read a shard and return shard_to_samples() entries (CPU tensors)."""
    path = shard_path(features_dir, category, split, modality)
    if not os.path.exists(path):
        raise FileNotFoundError("missing shard: %s (run extract_features first)" % path)
    return shard_to_samples(load_shard(path))


def concat_descriptors(samples):
    if not samples:
        return torch.empty((0, 0))
    return torch.cat([s["desc"].float() for s in samples], dim=0)


def evaluate_bank(cfg, category, train_desc, grid, test_samples, tag, cache_root, blur=None):
    """Full memory-bank evaluation: normalize -> coreset -> exact NN -> metrics.

    Returns a row dict {"category", "I", "P", "AUPRO"} plus faces for fusion
    runners: (scores, maps, labels, names).  Maps are 224x224 and gt from the
    test shard is 224x224.
    """
    from metrics import compute_metrics
    bank_t, stats = normalize_and_bank(train_desc, cfg, tag, cache_root)
    bank_np = bank_t.detach().cpu().numpy().astype(np.float32)
    index, bank_np = make_memory_index(bank_np)

    scores, maps, labels, names = score_samples(test_samples, bank_np, index, stats,
                                                grid, blur, cfg)
    gts = [np.asarray(s["gt"], dtype=np.float32) for s in test_samples]
    pixel_scores = np.concatenate([m.ravel() for m in maps]) if maps else np.array([])
    pixel_labels = np.concatenate([g.ravel() for g in gts]) if gts else np.array([])

    m = compute_metrics(
        labels, scores, pixel_labels, pixel_scores, gts, maps,
        integration_limit=float(cfg_get(cfg, "aupro_integration_limit", 0.3)),
        num_thresholds=int(cfg_get(cfg, "aupro_num_thresholds", 100)),
    )
    row = {"category": category, "I": m["I"], "P": m["P"], "AUPRO": m["AUPRO"]}
    return row, scores, maps, labels, names
