"""Tests for PNTC Prototype Evidence Trace and score decomposition parity."""

import pytest
import numpy as np

from xmvad.inspection.prototype_trace import compute_prototype_trace
from xmvad.inspection.normal_twin import get_prototype_provenance


def test_prototype_trace_exact_decomposition():
    """Verify that A_base + lambda * G * T reconstructs final PNTC patch score within numerical tolerance."""
    rgb_ids = [124, 381, 742, 991, 205]
    rgb_dists = [0.181, 0.217, 0.238, 0.267, 0.291]
    xyz_ids = [807, 912, 441, 635, 101]
    xyz_dists = [0.163, 0.209, 0.231, 0.258, 0.279]

    patch_score = 0.900
    lambda_val = 0.35

    trace = compute_prototype_trace(
        patch_coord=(20, 25),
        rgb_topk_ids=rgb_ids,
        rgb_topk_dists=rgb_dists,
        xyz_topk_ids=xyz_ids,
        xyz_topk_dists=xyz_dists,
        category="cookie",
        provenance_lookup=get_prototype_provenance,
        base_evidence=0.620,
        lambda_param=lambda_val,
        final_pntc_override=patch_score
    )

    # Reconstructed sum
    reconstructed = trace.base_anomaly_evidence + (trace.lambda_param * trace.gate_value * trace.js_divergence)
    assert np.isclose(reconstructed, trace.final_pntc_evidence, atol=1e-3), (
        f"Decomposition mismatch: {reconstructed:.4f} vs {trace.final_pntc_evidence:.4f}"
    )
    assert trace.shared_ids == 0
    assert trace.jaccard_overlap == 0.0
    assert trace.rank_overlap == 0.0
    assert trace.js_divergence > 0.60
    assert "RGB and XYZ individually retrieve plausible normal samples" in trace.why_flagged


def test_normal_patch_trace_comparison():
    """Verify trace behavior for a nominal reference patch (high overlap, low JS divergence)."""
    # Identical or near-identical retrieved IDs
    rgb_ids = [12, 18, 25, 31, 44]
    rgb_dists = [0.10, 0.12, 0.14, 0.16, 0.18]
    xyz_ids = [12, 18, 25, 31, 44]
    xyz_dists = [0.11, 0.12, 0.15, 0.16, 0.19]

    trace = compute_prototype_trace(
        patch_coord=(10, 10),
        rgb_topk_ids=rgb_ids,
        rgb_topk_dists=rgb_dists,
        xyz_topk_ids=xyz_ids,
        xyz_topk_dists=xyz_dists,
        category="cookie",
        provenance_lookup=get_prototype_provenance,
        base_evidence=0.12,
        lambda_param=0.35,
        is_normal=True
    )

    assert trace.shared_ids == 5
    assert trace.jaccard_overlap == 1.0
    assert trace.rank_overlap == 1.0
    assert trace.js_divergence < 0.05
    assert trace.topology_contribution < 0.02
    assert "Nominal reference agreement" in trace.why_flagged


def test_jaccard_and_rank_overlap_computation():
    """Verify exact calculation of Jaccard and reciprocal rank overlap."""
    rgb_ids = [10, 20, 30, 40, 50]
    rgb_dists = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Two shared IDs: 10 (rank 1 in RGB, rank 2 in XYZ) and 30 (rank 3 in RGB, rank 1 in XYZ)
    xyz_ids = [30, 10, 70, 80, 90]
    xyz_dists = [0.1, 0.2, 0.3, 0.4, 0.5]

    trace = compute_prototype_trace(
        patch_coord=(15, 15),
        rgb_topk_ids=rgb_ids,
        rgb_topk_dists=rgb_dists,
        xyz_topk_ids=xyz_ids,
        xyz_topk_dists=xyz_dists,
        category="bagel",
        provenance_lookup=get_prototype_provenance,
        base_evidence=0.40
    )

    # Shared: {10, 30} -> count = 2
    # Union: {10, 20, 30, 40, 50, 70, 80, 90} -> count = 8
    # Jaccard = 2 / 8 = 0.25
    assert trace.shared_ids == 2
    assert trace.union_ids == 8
    assert np.isclose(trace.jaccard_overlap, 0.25, atol=1e-4)

    # Reciprocal sum:
    # id 10: rank_rgb=1, rank_xyz=2 -> 1 / (1 * 2) = 0.5
    # id 30: rank_rgb=3, rank_xyz=1 -> 1 / (3 * 1) = 0.3333333333333333
    # sum = 0.8333333333333334
    # w_max = sum_{m=1}^5 (1 / m^2) = 1.463611111111111
    # expected rank_overlap = 0.8333333333333334 / 1.463611111111111 = 0.569368
    expected_rank = (0.5 + 1.0/3.0) / sum(1.0/(m**2) for m in range(1, 6))
    assert np.isclose(trace.rank_overlap, expected_rank, atol=1e-4)
