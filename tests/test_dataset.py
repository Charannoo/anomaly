"""Data-layer tests: discovery, pairing, loading (synthetic fixtures only).

No real MVTec 3D-AD data required. Builds a minimal fake
data/raw/mvtec3d tree in tmp_path with RGB PNGs + XYZ TIFFs + GT masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tifffile = pytest.importorskip("tifffile")

from xmvad.data.mvtec3d import (
    MVTec3DDataset,
    discover_categories,
    discover_dataset,
    discover_split,
    find_pairing_gaps,
    load_sample_arrays,
)


def _write_rgb(path: Path, size=(32, 24), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_xyz(path: Path, h=24, w=32, invalid_top_rows=2) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    xyz[..., 0] = xx.astype(np.float32) * 0.01 + 1.0
    xyz[..., 1] = yy.astype(np.float32) * 0.01 + 1.0
    xyz[..., 2] = (xx + yy).astype(np.float32) * 0.001 + 0.5
    xyz[:invalid_top_rows, :, :] = 0.0  # sentinel invalid rows
    tifffile.imwrite(str(path), xyz)
    return xyz


def _write_gt(path: Path, size=(32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[5:10, 5:10] = 255
    Image.fromarray(arr).save(path)


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec3d"
    # bagel/train/good: 2 paired samples
    for i in range(2):
        _write_rgb(root / "bagel" / "train" / "good" / "rgb" / f"{i:03d}.png")
        _write_xyz(root / "bagel" / "train" / "good" / "xyz" / f"{i:03d}.tiff")
    # bagel/test/good: 1 normal
    _write_rgb(root / "bagel" / "test" / "good" / "rgb" / "000.png")
    _write_xyz(root / "bagel" / "test" / "good" / "xyz" / "000.tiff")
    # bagel/test/crack: 1 anomalous + gt
    _write_rgb(root / "bagel" / "test" / "crack" / "rgb" / "000.png")
    _write_xyz(root / "bagel" / "test" / "crack" / "xyz" / "000.tiff")
    _write_gt(root / "bagel" / "test" / "crack" / "gt" / "000.png")
    # rope/train/good with one pairing gap (rgb without xyz)
    _write_rgb(root / "rope" / "train" / "good" / "rgb" / "000.png")
    _write_xyz(root / "rope" / "train" / "good" / "xyz" / "000.tiff")
    _write_rgb(root / "rope" / "train" / "good" / "rgb" / "001.png")
    return root


def test_discover_categories(fake_root: Path) -> None:
    assert discover_categories(fake_root) == ["bagel", "rope"]


def test_discover_split_counts(fake_root: Path) -> None:
    assert len(discover_split(fake_root, "bagel", "train")) == 2
    assert len(discover_split(fake_root, "bagel", "test")) == 2
    labels = sorted(r.label for r in discover_split(fake_root, "bagel", "test"))
    assert labels == [0, 1]


def test_pairing_gap_detected(fake_root: Path) -> None:
    gaps = find_pairing_gaps(fake_root, "rope", "train")
    assert any(g["kind"] == "missing_xyz" and g["stem"] == "001" for g in gaps)


def test_load_sample_arrays_shapes(fake_root: Path) -> None:
    rec = [r for r in discover_dataset(fake_root) if r.defect == "crack"][0]
    data = load_sample_arrays(rec)
    assert data["xyz"].shape == (24, 32, 3)
    assert data["depth"].shape == (24, 32)
    assert data["valid"].shape == (24, 32)
    assert data["gt"] is not None and data["gt"].shape == (24, 32)
    # invalid sentinel rows are masked out
    assert not data["valid"][:2].any()
    assert data["valid"][2:].all()


def test_torch_dataset_output_shapes(fake_root: Path) -> None:
    torch = pytest.importorskip("torch")
    recs = discover_dataset(fake_root)
    ds = MVTec3DDataset(recs, input_size=(16, 16))
    item = ds[0]
    assert item["rgb"].shape == (3, 16, 16)
    assert item["depth"].shape == (1, 16, 16)
    assert item["valid"].shape == (1, 16, 16)
    assert item["gt"].shape == (16, 16)
    assert item["depth"].isfinite().all()


def test_transforms_preserve_alignment() -> None:
    """Same geometric flip must hit RGB, depth, valid and GT identically."""
    torch = pytest.importorskip("torch")
    from xmvad.data.transforms import SynchronizedTransform

    # Left/right asymmetric fixtures: defect signal only on the left half.
    rgb = Image.new("RGB", (32, 16), (0, 0, 0))
    arr = np.array(rgb)
    arr[:, :16, 0] = 255
    rgb = Image.fromarray(arr)
    depth = np.zeros((16, 32), dtype=np.float32)
    depth[:, :16] = 1.0
    valid = np.ones((16, 32), dtype=bool)
    gt = np.zeros((16, 32), dtype=bool)
    gt[:, :16] = True

    t = SynchronizedTransform(input_size=(16, 32), train=True, hflip_p=1.0,
                              max_rotate_deg=0.0)
    out = t(rgb, depth, valid, gt)
    # After a forced hflip, the signal must be on the RIGHT half everywhere.
    assert out["rgb"].shape == (3, 16, 32)
    left_r = out["rgb"][0, :, :16].mean().item()
    right_r = out["rgb"][0, :, 16:].mean().item()
    assert right_r > left_r
    d = out["depth"][0].numpy()
    assert d[:, 16:].mean() > d[:, :16].mean()
    g = out["gt"].numpy()
    assert g[:, 16:].mean() > g[:, :16].mean()
