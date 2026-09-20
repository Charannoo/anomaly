#!/usr/bin/env python3
"""Bootstrap for the XMV-AD-H Colab bundle.

  * finds the dataset archive in Drive (config.dataset_archive_names), extracts it,
  * downloads + verifies the Point-MAE pretrained checkpoint (URL + size + sha),
  * restores already-computed feature/checkpoint caches from Drive into the
    working tree so re-runs skip backbone extraction entirely.

Works identically on a fresh Colab runtime and locally (paths get rewritten to
the bundle directory by cache_utils.load_config).
"""

import os
import shutil
import tarfile
import urllib.request

import cache_utils
from cache_utils import (cfg_get, ensure_dir, ensure_dirs, hash_file_sha256,
                         is_colab, load_config, local_time)


def _banner(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62, flush=True)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

def _find_archive(dataset_area, names):
    for n in names:
        p = os.path.join(dataset_area, n)
        if os.path.isfile(p):
            return p, n
    if os.path.isdir(dataset_area):
        for f in sorted(os.listdir(dataset_area)):
            if f.endswith((".tar.gz", ".tar.zst", ".tar.xz", ".tgz")):
                return os.path.join(dataset_area, f), f
    return None, None


def _extract_with_tarfile(arch, dest, members_func=None):
    with tarfile.open(arch, "r:*") as tf:
        members = members_func(tf) if members_func else tf.getmembers()
        safe = [m for m in members if not (m.isln() and "/" in (m.linkname or ""))]
        tf.extractall(dest, members=safe)


def _extract_zst(arch, dest):
    import zstandard
    out = os.path.join(dest, os.path.basename(arch)[:-len(".tar.zst")] + ".tar")
    with open(arch, "rb") as fin, open(out, "wb") as fout:
        dctx = zstandard.ZstdDecompressor()
        dctx.copy_stream(fin, fout)
    with tarfile.open(out, "r:*") as tf:
        tf.extractall(dest)
    os.remove(out)


def _locate_dataset_dir(dest, categories):
    """Find the directory containing every category (deepest match wins)."""
    if os.path.isdir(os.path.join(dest, "mvtec3d")):
        return os.path.join(dest, "mvtec3d")
    hits = []
    for root, dirs, files in os.walk(dest):
        if all(os.path.isdir(os.path.join(root, c)) for c in categories):
            hits.append(root)
    if not hits:
        return None
    return max(hits, key=lambda p: p.count(os.sep))


def locate_or_extract_dataset(cfg, categories=None):
    categories = categories or cfg["categories"]
    dataset_dir = cfg["dataset_dir"]
    if os.path.isdir(os.path.join(dataset_dir, "train")) or \
       all(os.path.isdir(os.path.join(dataset_dir, c)) for c in categories[:3]):
        print("[dataset] already extracted at %s" % dataset_dir)
        return dataset_dir

    ensure_dir(cfg["data_root"])
    arch, name = _find_archive(cfg["persistent"]["dataset"], cfg_get(cfg, "dataset_archive_names", []))
    if arch is None:
        raise FileNotFoundError(
            "No dataset archive found under %s (looked for %s). Put "
            "mvtec3d.tar.gz (or .tar.zst / .tar.xz) there first."
            % (cfg["persistent"]["dataset"], cfg_get(cfg, "dataset_archive_names", [])))

    print("[dataset] extracting %s -> %s" % (name, cfg["data_root"]))
    if name.endswith(".tar.zst"):
        _extract_zst(arch, cfg["data_root"])
    elif name.endswith((".tar.gz", ".tar.xz", ".tgz", ".tar")):
        _extract_with_tarfile(arch, cfg["data_root"])
    else:
        raise RuntimeError("unsupported archive format: %s" % name)

    found = _locate_dataset_dir(cfg["data_root"], categories)
    if found is None or not os.path.isdir(os.path.join(found, "train")):
        raise RuntimeError(
            "Extracted archive did not produce an MVTec-3D-AD tree at %s "
            "(expected <root>/<category>/{train,test})." % cfg["data_root"])
    if os.path.abspath(found) != os.path.abspath(dataset_dir):
        print("[dataset] dataset root located at %s (symlink to %s)" % (found, dataset_dir))
        ensure_dir(os.path.dirname(dataset_dir))
        if not os.path.exists(dataset_dir) and os.path.abspath(found) != os.path.abspath(dataset_dir):
            os.symlink(found, dataset_dir)
        if os.path.isdir(dataset_dir) and not any(
                c in os.listdir(dataset_dir) for c in categories[:3]):
            for c in categories:
                src = os.path.join(found, c)
                if os.path.isdir(src):
                    os.symlink(src, os.path.join(dataset_dir, c))
    print("[dataset] ready -> %s" % dataset_dir)
    return dataset_dir


# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------

def _verify_checkpoint(path, cfg):
    if not os.path.isfile(path):
        return "missing"
    size = os.path.getsize(path)
    exp = int(cfg_get(cfg, "pointmae_expected_bytes", 0))
    if exp and abs(size - exp) > 1024:
        return "size_mismatch (%d vs %d)" % (size, exp)
    prefix = cfg_get(cfg, "pointmae_expected_sha256_prefix", None)
    if prefix:
        got = hash_file_sha256(path)
        if not got.startswith(prefix):
            return "sha_mismatch (%s != %s)" % (got, prefix)
    return "ok"


def _download(url, path, expected_bytes=None, chunk=1 << 20):
    print("[download] %s -> %s" % (url, path))
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        done = 0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            out.write(block)
            done += len(block)
            if expected_bytes and (done % (8 << 20) == 0):
                print("   ... %5.1f%% (%d MB)" % (100.0 * done / expected_bytes, done >> 20),
                      flush=True)
    os.replace(tmp, path)
    print("[download] complete (%d bytes)" % done)


def ensure_pointmae_checkpoint(cfg):
    dst = cfg["checkpoints_dir"]
    ensure_dir(dst)
    parts = (cfg["persistent"]["checkpoints"], "pointmae_pretrain.pth")
    mirror = os.path.join(*parts)
    if os.path.isfile(mirror) and _verify_checkpoint(mirror, cfg) == "ok":
        if os.path.abspath(mirror) != os.path.abspath(os.path.join(dst, "pointmae_pretrain.pth")):
            shutil.copy2(mirror, os.path.join(dst, "pointmae_pretrain.pth"))
            print("[checkpoint] restored from Drive mirror %s" % mirror)
    target = os.path.join(dst, "pointmae_pretrain.pth")
    status = _verify_checkpoint(target, cfg)
    if status != "ok":
        if not is_colab():
            raise RuntimeError(
                "Point-MAE checkpoint at %s is %s. Put pointmae_pretrain.pth "
                "under %s and re-run."
                % (target, status, cfg["persistent"]["checkpoints"]))
        url = cfg_get(cfg, "pointmae_url")
        _download(url, target, expected_bytes=int(cfg_get(cfg, "pointmae_expected_bytes", 0)))
        status = _verify_checkpoint(target, cfg)
        if status != "ok":
            raise RuntimeError("downloaded checkpoint failed verification: %s" % status)
    print("[checkpoint] pointmae_pretrain.pth OK (%d bytes)" % os.path.getsize(target))
    return target


# ----------------------------------------------------------------------
# Drive cache restore
# ----------------------------------------------------------------------

def restore_caches(cfg):
    """Copy persistent (Drive) features/checkpoints into the working tree if
    the local copies are missing, so a fresh Colab session resumes instantly."""
    src_f = cfg["persistent"]["features"]
    dst_f = cfg["features_dir"]
    ensure_dirs([dst_f, cfg["checkpoints_dir"]])

    local_manifest = cache_utils.manifest_path(dst_f)
    persistent_manifest = cache_utils.manifest_path(src_f)
    copied = 0
    same = os.path.abspath(src_f) == os.path.abspath(dst_f)
    if os.path.isdir(src_f):
        for f in sorted(os.listdir(src_f)):
            src = os.path.join(src_f, f)
            dst = os.path.join(dst_f, f)
            if same:
                break
            if os.path.isfile(src) and (not os.path.exists(dst) or
                                        f == cache_utils.MANIFEST_NAME):
                shutil.copy2(src, dst)
                copied += 1
    if not same and os.path.exists(persistent_manifest) and not os.path.exists(local_manifest):
        shutil.copy2(persistent_manifest, local_manifest)
    print("[restore] features: %d files mirrored %s -> %s" % (copied, src_f, dst_f))

    src_c = os.path.join(cfg["persistent"]["checkpoints"], "pointmae_pretrain.pth")
    dst_c = os.path.join(cfg["checkpoints_dir"], "pointmae_pretrain.pth")
    if os.path.abspath(src_c) != os.path.abspath(dst_c) and \
       os.path.isfile(src_c) and not os.path.exists(dst_c):
        shutil.copy2(src_c, dst_c)
        print("[restore] checkpoint mirrored from Drive")


def setup(cfg=None, categories=None):
    cfg = cfg or load_config()
    categories = categories or cfg["categories"]
    ensure_dirs([cfg["data_root"], cfg["features_dir"], cfg["results_dir"],
                 cfg["logs_dir"], cfg["checkpoints_dir"],
                 cfg["persistent"]["features"], cfg["persistent"]["results"],
                 cfg["persistent"]["logs"], cfg["persistent"]["checkpoints"]])
    _banner("BOOTSTRAP")
    print("colab=%s  device_cuda=%s" % (is_colab(), cache_utils.torch_device()))
    print("data       : %s" % cfg["data_root"])
    print("features   : %s" % cfg["features_dir"])
    print("results    : %s" % cfg["results_dir"])

    restore_caches(cfg)
    locate_or_extract_dataset(cfg, categories)
    ensure_pointmae_checkpoint(cfg)

    n_shards = len([f for f in os.listdir(cfg["features_dir"])
                    if f.endswith(".pt") and not f.endswith(".tmp")])
    print("\n[bootstrap] done. %d feature shards present in %s"
          % (n_shards, cfg["features_dir"]))
    return cfg


if __name__ == "__main__":
    setup()