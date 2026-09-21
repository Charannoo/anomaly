"""Automated tests for PNTC Inspect FastAPI endpoints."""

import pytest
from starlette.testclient import TestClient
from xmvad.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_system_status_endpoint(client):
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert data["canonical_tag"] == "h5d-pntc-verified"
    assert data["canonical_metrics"]["I-AUROC"] == 0.96541000
    assert data["canonical_metrics"]["P-AUROC"] == 0.99416000
    assert data["canonical_metrics"]["AUPRO@0.3"] == 0.96939000


def test_list_inspections_endpoint(client):
    res = client.get("/api/inspections")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert data["total"] >= 6
    assert len(data["items"]) >= 6


def test_get_single_inspection_endpoint(client):
    res = client.get("/api/inspections/INSP-01_strong_defect")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "INSP-01_strong_defect"
    assert "report" in data
    assert data["report"]["pntc"]["decision"] == "anomalous"
    assert len(data["report"]["defects"]) >= 1


def test_get_inspection_artifact_endpoint(client):
    res = client.get("/api/inspections/INSP-01_strong_defect/artifacts/overlay")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"


def test_analytics_endpoint(client):
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.json()
    assert "total_inspections" in data
    assert "anomalies" in data
    assert "normal" in data
    assert data["total_inspections"] >= 6


def test_model_info_endpoint(client):
    res = client.get("/api/model/info")
    assert res.status_code == 200
    data = res.json()
    assert data["canonical_metrics"]["I-AUROC"] == 0.96541000
    assert data["architecture_parameters"]["k_neighbors"] == 5
    assert len(data["benchmark_comparison"]) >= 4


def test_demo_cases_endpoint(client):
    res = client.get("/api/demo/cases")
    assert res.status_code == 200
    cases = res.json()
    assert len(cases) >= 6
    assert cases[0]["id"] == "01_strong_defect"


def test_assistant_provider_endpoint(client):
    res = client.get("/api/assistant/provider")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data
    assert "configured" in data
    # Ensure no secret keys exposed
    assert "api_key" not in data
    assert "key" not in data
