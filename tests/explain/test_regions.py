"""Tests for anomaly region extraction, noise filtering, and ranking."""

import numpy as np
import pytest
from xmvad.explain.regions import extract_anomaly_regions


def test_empty_anomaly_map():
    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    regions, thresh, binary = extract_anomaly_regions(anomaly_map, threshold=0.5)
    assert len(regions) == 0
    assert np.sum(binary) == 0
    assert thresh == 0.5


def test_single_connected_component():
    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    # 20x20 square anomaly with score 0.8
    anomaly_map[30:50, 30:50] = 0.8
    regions, thresh, binary = extract_anomaly_regions(anomaly_map, threshold=0.5, min_area_px=10)
    assert len(regions) == 1
    assert regions[0]["id"] == 1
    assert regions[0]["area_px"] == 400
    assert np.isclose(regions[0]["mean_score"], 0.8)


def test_multiple_component_ranking():
    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    # Component 1: small high severity (10x10 = 100px, score 0.95) -> importance = 0.95 * 10 = 9.5
    anomaly_map[10:20, 10:20] = 0.95
    # Component 2: large moderate severity (30x30 = 900px, score 0.70) -> importance = 0.70 * 30 = 21.0
    anomaly_map[40:70, 40:70] = 0.70

    regions, thresh, binary = extract_anomaly_regions(anomaly_map, threshold=0.5, min_area_px=10)
    assert len(regions) == 2
    # Larger integrated importance should be rank 1
    assert regions[0]["id"] == 1
    assert regions[0]["area_px"] == 900
    assert regions[1]["id"] == 2
    assert regions[1]["area_px"] == 100


def test_noise_spike_filtering():
    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    # Isolated 2-pixel spike
    anomaly_map[5, 5] = 0.99
    anomaly_map[5, 6] = 0.99
    regions, thresh, binary = extract_anomaly_regions(anomaly_map, threshold=0.5, min_area_px=10)
    # Filtered out by opening and min_area
    assert len(regions) == 0
