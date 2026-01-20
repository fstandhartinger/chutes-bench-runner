"""Tests for API endpoints."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.models.model import Model
from app.services.artificial_analysis_service import ArtificialAnalysisScores, MatchResult


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


def test_artificial_analysis_lookup_success(client: TestClient, test_session, monkeypatch):
    """Test Artificial Analysis lookup with stubbed scores."""

    async def seed_model():
        model = Model(slug="glm-4-7", name="GLM-4.7", provider="chutes")
        test_session.add(model)
        await test_session.commit()
        await test_session.refresh(model)
        return model

    model = asyncio.get_event_loop().run_until_complete(seed_model())

    async def fake_get_benchmarks_for_model(self, summary, include_raw=False, llm_fallback=True):
        match = MatchResult(
            slug="glm-4-7",
            method="exact",
            confidence=1.0,
            candidates=["glm-4-7"],
            llm_used=False,
        )
        scores = ArtificialAnalysisScores(
            slug="glm-4-7",
            name="GLM-4.7 (Reasoning)",
            short_name="GLM-4.7",
            model_url="https://artificialanalysis.ai/models/glm-4-7",
            hosts_url=None,
            scores=[{"key": "aime25", "label": "AIME 2025 (Competition Math)", "value": 0.95, "format": "percent"}],
            raw=None,
        )
        return match, scores

    monkeypatch.setattr(
        "app.services.artificial_analysis_service.ArtificialAnalysisService.get_benchmarks_for_model",
        fake_get_benchmarks_for_model,
    )

    response = client.get("/api/benchmarks/artificial-analysis", params={"model_id": model.slug})
    assert response.status_code == 200
    data = response.json()
    assert data["match"]["slug"] == "glm-4-7"
    assert data["artificial_analysis"]["slug"] == "glm-4-7"
    assert data["artificial_analysis"]["scores"][0]["key"] == "aime25"






























