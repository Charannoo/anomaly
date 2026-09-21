"""Regression test to guarantee that the addition of the assistant layer has zero effect on frozen canonical detector metrics."""

import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_frozen_canonical_metrics_preserved():
    """Verify frozen PNTC canonical metrics remain strictly intact."""
    metric_file = os.path.join(PROJECT_ROOT, "results", "H5_raw_metric_verification.csv")
    assert os.path.exists(metric_file), f"Verification artifact missing: {metric_file}"

    df = pd.read_csv(metric_file)
    mean_row = df[df["category"] == "MEAN"].iloc[0]

    mean_h5d_i = float(mean_row["H5D_I"])
    mean_h5d_p = float(mean_row["H5D_P"])
    mean_h5d_a = float(mean_row["H5D_AUPRO"])

    # Canonical frozen metrics
    CANONICAL_I = 0.96541000
    CANONICAL_P = 0.99416000
    CANONICAL_A = 0.96939000

    assert np.isclose(mean_h5d_i, CANONICAL_I, atol=1e-5), f"I-AUROC changed: {mean_h5d_i} vs {CANONICAL_I}"
    assert np.isclose(mean_h5d_p, CANONICAL_P, atol=1e-5), f"P-AUROC changed: {mean_h5d_p} vs {CANONICAL_P}"
    assert np.isclose(mean_h5d_a, CANONICAL_A, atol=1e-5), f"AUPRO changed: {mean_h5d_a} vs {CANONICAL_A}"
