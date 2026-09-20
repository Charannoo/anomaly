#!/usr/bin/env python3
"""Feature extraction for XMV-AD-H (Colab bundle).

Extracts frozen DINO ViT-B/8 (RGB) and Point-MAE (XYZ) features once per
category/split with atomic, resumable shard caching (cache_utils.CacheManifest
+ "<cat>_<split>_<modality>.pt" shards under cfg.features_dir).

The XYZ branch reproduces scripts/high_accuracy/extract_canonical_features.py
exactly: PointTransformer tokens are interpolated back onto every valid cloud
point (chunked_interpolating_points), scattered into a 224x224 grid, then
AvgPool2d(3) + AdaptiveAvgPool2d((56,56)) -> (3136, 1152) per sample.  The RGB
branch stores the raw (784, 768) patch tokens; the H3 runner up-samples them
with cache_utils.align_rgb_descriptors (canonical repeat layout) so the concat
bank is byte-for-byte aligned with the (56,56) XYZ grid.
"""

import gc
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import cache_utils
from cache_utils import (CacheManifest, cfg_get, ensure_dirs, ensure_dir,
                         is_colab, local_time, list_category_samples,
                         load_config, make_shard, save_shard, torch_device)

import backbones
from backbones import DINOBackbone, PointTransformer, chunked_interpolating_points

RGB_SIZE = 224
RGB_PER_SAMPLE = 784
XYZ_PER_SAMPLE = 3136


def _banner(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62, flush=True)


def _memory_safe_clip(feats):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


class Extractor(object):
    """Frozen DINO (RGB) + pure-PyTorch Point-MAE (XYZ) backbones."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        num_group = int(cfg_get(cfg, "num_group", 1024))
        group_size = int(cfg_get(cfg, "group_size", 128))

        print("[extractor] loading RGB backbone  : %s" % cfg_get(cfg, "rgb_backbone_name", "dino"))
        self.dino = DINOBackbone(name=cfg_get(cfg, "rgb_backbone_name", "vit_base_patch8_224_dino"),
                                 device=device)

        ckpt = os.path.join(cfg["checkpoints_dir"], "pointmae_pretrain.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                "Point-MAE checkpoint not found at %s - run bootstrap.py first." % ckpt)
        print("[extractor] loading XYZ backbone  : Point_MAE (pure torch) from %s" % ckpt)
        self.pt = PointTransformer(num_group=num_group, group_size=group_size).to(device)
        self.pt.eval()
        self.pt.load_model_from_ckpt(ckpt)
        for p in self.pt.parameters():
            p.requires_grad = False

        self.avg_pool = torch.nn.AvgPool2d(3, stride=1)
        self.resize_pool = torch.nn.AdaptiveAvgPool2d((56, 56))

        self.rgb_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        print("[extractor] ready on %s" % device, flush=True)

    # --- RGB -----------------------------------------------------------
    @torch.inference_mode()
    def extract_rgb_batch(self, samples):
        tensors = []
        for s in samples:
            with Image.open(s["rgb"]) as img:
                img = img.convert("RGB")
            tensors.append(self.rgb_transform(img))
        x = torch.stack(tensors, dim=0).to(self.device)
        feat = self.dino(x)                        # (B, 768, 28, 28)
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])  # (B*784, 768)
        return feat.detach().cpu()

    # --- XYZ -----------------------------------------------------------
    @torch.inference_mode()
    def extract_xyz_batch(self, samples):
        paths = [s["xyz"] for s in samples]
        clouds, valid, nzs = backbones.load_xyz_batch(paths, self.device)
        pts = clouds.permute(0, 2, 1).contiguous()
        feats, center, _ori, _ci = self.pt(pts, valid)   # (B,1152,G), (B,G,3)
        feats = feats.float()
        center = center.float()

        interp = chunked_interpolating_points(
            clouds.permute(0, 2, 1).contiguous(),
            center.permute(0, 2, 1).contiguous(),
            feats,
        )                                            # (B, 1152, P)

        B, D, P = interp.shape
        grid = torch.zeros(B, D, RGB_SIZE * RGB_SIZE, dtype=interp.dtype, device=interp.device)
        for b in range(B):
            nb = nzs[b].shape[0]
            nz = torch.as_tensor(nzs[b], dtype=torch.long, device=interp.device)
            grid[b, :, nz] = interp[b, :, :nb]
        grid2 = grid.view(B, D, RGB_SIZE, RGB_SIZE)
        out = self.resize_pool(self.avg_pool(grid2))     # (B,1152,56,56)
        out = out.reshape(B, D, XYZ_PER_SAMPLE).permute(0, 2, 1)
        return out.reshape(B * XYZ_PER_SAMPLE, D).detach().cpu()


# ----------------------------------------------------------------------
# Auto-tuning batch driver
# ----------------------------------------------------------------------

def _auto_tune(extract_fn, samples, batch_tries, tag):
    """Run `extract_fn` over `samples`, shrinking the batch on CUDA OOM."""
    errors = []
    for batch in batch_tries:
        parts = []
        try:
            for i in range(0, len(samples), batch):
                parts.append(extract_fn(samples[i:i + batch]))
                if len(parts) % 10 == 0:
                    _memory_safe_clip(None)
            return torch.cat(parts, dim=0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in msg:
                raise
            errors.append("%s: %s" % (batch, e))
            _memory_safe_clip(None)
            if batch == batch_tries[-1]:
                raise RuntimeError("extraction OOM at every batch size (%s): %s" % (tag, errors))
            print("   [OOM] %s at batch=%d, retrying with a smaller batch..." % (tag, batch),
                  flush=True)
    raise RuntimeError("no batch size succeeded for %s" % tag)


def _precision_tensor(desc, precision):
    return desc.half() if precision == "float16" else desc.float()


# ----------------------------------------------------------------------
# Per split extraction
# ----------------------------------------------------------------------

def _make_gt_transform():
    return transforms.Compose([
        transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def _extract_split(cfg, extractor, category, split, samples, manifest, precision):
    n = len(samples)
    if n == 0:
        print("[skip] %s %s: no samples" % (category, split))
        return

    # GT / labels / names are modality-independent.
    transform = _make_gt_transform()
    labels, names, gts = [], [], []
    for s in samples:
        labels.append(int(s["label"]))
        names.append(s["name"])
        if s.get("gt") and os.path.exists(s["gt"]):
            gt_img = Image.open(s["gt"]).convert("L")
            gt_t = transform(gt_img)
            gt_t = torch.where(gt_t > 0.5, 1.0, 0.0)[0].numpy()
        else:
            gt_t = np.zeros((RGB_SIZE, RGB_SIZE), dtype=np.float32)
        gts.append(gt_t.astype(np.float32))

    rgb_tries = [int(x) for x in cfg_get(cfg, "rgb_batch_tries", [64, 32, 16, 8, 4, 2, 1])]
    xyz_tries = [int(x) for x in cfg_get(cfg, "xyz_batch_tries", [16, 8, 4, 2, 1])]
    ckpt_info = os.path.join(cfg["checkpoints_dir"], "pointmae_pretrain.pth")

    # RGB shard (raw 28x28 tokens).
    if not manifest.is_complete(category, split, "rgb"):
        _banner("EXTRACT [%s %s] RGB DINO (%d samples)" % (category, split, n))
        desc = _auto_tune(extractor.extract_rgb_batch, samples, rgb_tries, "%s_%s_rgb" % (category, split))
        _memory_safe_clip(None)
        shard = make_shard(category, split, "rgb", _precision_tensor(desc, precision),
                           RGB_PER_SAMPLE, labels, names, gts, (28, 28),
                           "dino_%s" % cfg_get(cfg, "dino_hub_tag", "dino"),
                           precision, None)
        save_shard(shard, cache_utils.shard_path(cfg["features_dir"], category, split, "rgb"))
        manifest.set_complete(category, split, "rgb", {
            "samples": n, "grid": [28, 28], "D": int(desc.shape[1]),
            "checkpoint": ckpt_info, "precision": precision,
        })
        print("   -> %s finished (%d x %d)" % (category + "_" + split + "_rgb", n * RGB_PER_SAMPLE, desc.shape[1]))

    # XYZ shard (56x56 interpolated tokens).
    if not manifest.is_complete(category, split, "xyz"):
        _banner("EXTRACT [%s %s] XYZ Point-MAE (%d samples)" % (category, split, n))
        desc = _auto_tune(extractor.extract_xyz_batch, samples, xyz_tries, "%s_%s_xyz" % (category, split))
        _memory_safe_clip(None)
        shard = make_shard(category, split, "xyz", _precision_tensor(desc, precision),
                           XYZ_PER_SAMPLE, labels, names, gts, (56, 56),
                           "pointmae_pretrain.pth", precision, None)
        save_shard(shard, cache_utils.shard_path(cfg["features_dir"], category, split, "xyz"))
        manifest.set_complete(category, split, "xyz", {
            "samples": n, "grid": [56, 56], "D": int(desc.shape[1]),
            "checkpoint": ckpt_info, "precision": precision,
        })
        print("   -> %s finished (%d x %d)" % (category + "_" + split + "_xyz", n * XYZ_PER_SAMPLE, desc.shape[1]))


def _sync_category(cfg, category, manifest_path):
    """Mirror shards + manifest into cfg.persistent.features (Drive on Colab)."""
    dest = cfg["persistent"]["features"]
    src_dir = cfg["features_dir"]
    if os.path.abspath(dest) == os.path.abspath(src_dir):
        return
    ensure_dir(dest)
    for fname in os.listdir(src_dir):
        if fname.startswith(category + "_") or fname == cache_utils.MANIFEST_NAME:
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, fname))
    print("   [sync] %s -> %s" % (src_dir, dest), flush=True)
    if is_colab():
        try:
            os.sync()
        except Exception:
            pass


def run_extraction(cfg, categories=None, device=None):
    device = device or torch_device()
    categories = categories or cfg["categories"]
    ensure_dirs([cfg["features_dir"], cfg["persistent"]["features"]])

    dataset_dir = cfg["dataset_dir"]
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            "dataset not extracted at %s - run bootstrap.py first." % dataset_dir)

    extractor = Extractor(cfg, device)
    manifest = CacheManifest(cfg["features_dir"])

    summary = {"extracted_rgb": 0, "extracted_xyz": 0, "cached_rgb": 0, "cached_xyz": 0}
    t_start = time.time()
    for category in categories:
        _banner("CATEGORY %s" % category)
        samples_by_split = list_category_samples(dataset_dir, category,
                                                 include_combined=bool(cfg_get(cfg, "include_combined", True)))
        for split in ("train", "test"):
            samples = samples_by_split.get(split, [])
            if not samples:
                print("[skip] %s %s: no samples" % (category, split))
                continue
            for modality in ("rgb", "xyz"):
                if manifest.is_complete(category, split, modality):
                    print("[CACHE HIT] %s %s %s" % (category, split, modality))
                    summary["cached_%s" % modality] += 1
                else:
                    summary["extracted_%s" % modality] += 1
            _extract_split(cfg, extractor, category, split, samples, manifest,
                           cfg_get(cfg, "cache_precision", "float16"))
            _sync_category(cfg, category, manifest.path)

    manifest.save()
    elapsed = time.time() - t_start
    print("\n==============================================")
    print("EXTRACTION SUMMARY  (%.1fs)" % elapsed)
    print("  RGB  : extracted=%d cached=%d" % (summary["extracted_rgb"], summary["cached_rgb"]))
    print("  XYZ  : extracted=%d cached=%d" % (summary["extracted_xyz"], summary["cached_xyz"]))
    print("  manifest: %s" % manifest.path)
    print("==============================================")
    return summary


if __name__ == "__main__":
    cfg = load_config()
    run_extraction(cfg)