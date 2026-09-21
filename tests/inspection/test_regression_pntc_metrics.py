"""Regression test to guarantee that frozen PNTC detector metrics are 100% unchanged."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_frozen_pntc_canonical_metrics_inspection_suite():
    """Verify that canonical metric artifacts preserve the exact canonical frozen metrics."""
    raw_verification_file = os.path.join(PROJECT_ROOT, "results", "H5_raw_metric_verification.csv")
    assert os.path.exists(raw_verification_file), f"Verification artifact missing: {raw_verification_file}"

    df = pd.read_csv(raw_verification_file)
    mean_row = df[df["category"] == "MEAN"].iloc[0]

    mean_h5d_i = float(mean_row["H5D_I"])
    mean_h5d_p = float(mean_row["H5D_P"])
    mean_h5d_a = float(mean_row["H5D_AUPRO"])

    # Canonical frozen metrics
    CANONICAL_I = 0.96541000
    CANONICAL_P = 0.99416000
    CANONICAL_A = 0.96939000

    assert np.isclose(mean_h5d_i, CANONICAL_I, atol=1e-5), f"I-AUROC mismatch: {mean_h5d_i} vs {CANONICAL_I}"
    assert np.isclose(mean_h5d_p, CANONICAL_P, atol=1e-5), f"P-AUROC mismatch: {mean_h5d_p} vs {CANONICAL_P}"
    assert np.isclose(mean_h5d_a, CANONICAL_A, atol=1e-5), f"AUPRO mismatch: {mean_h5d_a} vs {CANONICAL_A}"

    # Also verify baseline metrics
    mean_h4b_i = float(mean_row["H4B_I"])
    mean_h4b_p = float(mean_row["H4B_P"])
    mean_h4b_a = float(mean_row["H4B_AUPRO"])

    assert np.isclose(mean_h4b_i, 0.95401000, atol=1e-5)
    assert np.isclose(mean_h4b_p, 0.99109000, atol=1e-5)
    assert np.isclose(mean_h4b_a, 0.95852000, atol=1e-5)
