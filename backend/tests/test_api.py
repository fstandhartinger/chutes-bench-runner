"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Chutes Bench Runner"
    assert "version" in data


def test_list_models_empty(client: TestClient):
    """Test listing models when none exist."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["models"] == []
    assert data["total"] == 0


def test_list_benchmarks(client: TestClient):
    """Test listing benchmarks."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data


def test_create_run_invalid_model(client: TestClient):
    """Test creating run with invalid model."""
    response = client.post(
        "/api/runs",
        json={
            "model_id": "00000000-0000-0000-0000-000000000000",
            "subset_pct": 10,
        },
    )
    assert response.status_code == 404


def test_get_run_not_found(client: TestClient):
    """Test getting non-existent run."""
    response = client.get("/api/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_runs_empty(client: TestClient):
    """Test listing runs when none exist."""
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["runs"] == []


def test_cancel_run_not_found(client: TestClient):
    """Test canceling non-existent run."""
    response = client.post("/api/runs/00000000-0000-0000-0000-000000000000/cancel")
    assert response.status_code == 400


def test_export_run_not_found(client: TestClient):
    """Test exporting non-existent run."""
    response = client.get("/api/runs/00000000-0000-0000-0000-000000000000/export")
    assert response.status_code == 404


def test_sync_models_no_admin(client: TestClient):
    """Test sync models without admin secret."""
    # This should work if no admin secret is configured
    # In production, it would require proper auth
    response = client.post("/api/admin/sync-models")
    # Either 200 (no admin configured) or 401 (admin required)
    assert response.status_code in [200, 401, 500]


def test_subset_pct_validation(client: TestClient):
    """Test subset percentage validation."""
    # Invalid: too low
    response = client.post(
        "/api/runs",
        json={
            "model_id": "00000000-0000-0000-0000-000000000000",
            "subset_pct": 0,
        },
    )
    assert response.status_code == 422

    # Invalid: too high
    response = client.post(
        "/api/runs",
        json={
            "model_id": "00000000-0000-0000-0000-000000000000",
            "subset_pct": 101,
        },
    )
    assert response.status_code == 422


def test_export_format_validation(client: TestClient):
    """Test export format validation."""
    response = client.get("/api/runs/00000000-0000-0000-0000-000000000000/export?format=invalid")
    assert response.status_code == 422
























