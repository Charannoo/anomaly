"""Tests for Normal Twin reference retrieval and prototype provenance."""

import pytest
import numpy as np

from xmvad.inspection.normal_twin import (
    PrototypeProvenanceIndex,
    get_prototype_provenance,
    retrieve_normal_twin,
    select_representative_patches
)


def test_prototype_provenance_preserved():
    """Verify that prototype provenance records retain all required metadata fields."""
    idx = PrototypeProvenanceIndex()
    rec = idx.get_provenance(42, "cookie")

    assert rec["prototype_id"] == 42
    assert rec["category"] == "cookie"
    assert "train/cookie/good/" in rec["training_sample_id"]
    assert len(rec["spatial_coord"]) == 2
    assert rec["rgb_xyz_aligned"] is True
    assert rec["is_train_normal"] is True


def test_paired_rgb_xyz_correspondence():
    """Verify paired RGB and XYZ IDs correspond to the exact same physical normal location."""
    for pid in [0, 10, 500, 14999]:
        prov = get_prototype_provenance(pid, "cable_gland")
        assert prov["rgb_bbox"] == prov["xyz_bbox"]
        assert prov["rgb_xyz_aligned"] is True


def test_no_test_sample_in_normal_memory():
    """Strict audit: verify that no test or anomalous sample ever appears in normal prototype memory."""
    idx = PrototypeProvenanceIndex()
    for cat in ["bagel", "cookie", "foam", "peach", "cable_gland"]:
        for pid in [0, 5, 25, 100, 999, 14999]:
            prov = idx.get_provenance(pid, cat)
            source = prov["training_sample_id"]
            assert "test" not in source, f"Violation: test sample found in normal memory: {source}"
            assert "train" in source, f"Invalid source path: {source}"
            assert "good" in source, f"Non-nominal sample found in normal memory: {source}"


def test_normal_twin_retrieval_deterministic():
    """Verify top-k retrieval results are deterministic and reproducible."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 150
    xyz = np.ones((100, 100, 3), dtype=np.float32) * 50.0
    valid = np.ones((100, 100), dtype=bool)

    twin1 = retrieve_normal_twin((25, 25), "cookie", mask, rgb, xyz, valid, k=5, seed_offset=42)
    twin2 = retrieve_normal_twin((25, 25), "cookie", mask, rgb, xyz, valid, k=5, seed_offset=42)

    assert twin1.prototype_id == twin2.prototype_id
    assert np.isclose(twin1.joint_score, twin2.joint_score)
    assert len(twin1.rgb_topk_matches) == 5
    assert len(twin1.xyz_topk_matches) == 5
    assert twin1.rgb_topk_matches[0].prototype_id == twin2.rgb_topk_matches[0].prototype_id


def test_select_representative_patches():
    """Verify selection of peak evidence, peak topology, and region medoid patches."""
    mask = np.zeros((56, 56), dtype=bool)
    mask[20:30, 20:30] = True

    anomaly_map = np.zeros((56, 56), dtype=np.float32)
    anomaly_map[22, 24] = 0.95  # Peak anomaly

    topology_map = np.zeros((56, 56), dtype=np.float32)
    topology_map[27, 28] = 0.99  # Peak topology

    patches = select_representative_patches(mask, anomaly_map, topology_map, grid_size=56)
    assert "peak_evidence" in patches
    assert "peak_topology" in patches
    assert "region_medoid" in patches
    assert patches["peak_evidence"] == (22, 24)
    assert patches["peak_topology"] == (27, 28)
