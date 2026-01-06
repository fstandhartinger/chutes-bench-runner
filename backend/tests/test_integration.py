"""Integration tests against deployed API."""
import os
import httpx
import pytest

# Use the deployed API for integration tests
API_URL = os.getenv("API_URL", "https://chutes-bench-runner-api-v2.onrender.com")


@pytest.fixture
def client():
    """Create HTTP client."""
    return httpx.Client(base_url=API_URL, timeout=30.0)


class TestHealthEndpoints:
    """Test health and root endpoints."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint returns app info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Chutes Bench Runner"
        assert "version" in data
        assert data["docs"] == "/docs"


class TestModelsAPI:
    """Test models endpoints."""

    def test_list_models(self, client):
        """Test listing models returns non-empty list."""
        response = client.get("/api/models?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert len(data["models"]) > 0

    def test_models_have_required_fields(self, client):
        """Test models have required fields."""
        response = client.get("/api/models?limit=1")
        assert response.status_code == 200
        data = response.json()
        if data["models"]:
            model = data["models"][0]
            assert "id" in model
            assert "slug" in model
            assert "name" in model


class TestBenchmarksAPI:
    """Test benchmarks endpoints."""

    def test_list_benchmarks(self, client):
        """Test listing benchmarks."""
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        data = response.json()
        assert "benchmarks" in data
        assert len(data["benchmarks"]) == 19

    def test_benchmarks_have_required_fields(self, client):
        """Test benchmarks have required fields."""
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        data = response.json()
        for benchmark in data["benchmarks"]:
            assert "name" in benchmark
            assert "display_name" in benchmark
            assert "description" in benchmark
            assert "is_enabled" in benchmark


class TestRunsAPI:
    """Test runs endpoints."""

    def test_list_runs(self, client):
        """Test listing runs."""
        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data

    def test_get_run_not_found(self, client):
        """Test getting non-existent run returns 404."""
        response = client.get("/api/runs/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_create_run_invalid_model(self, client):
        """Test creating run with invalid model returns 404."""
        response = client.post(
            "/api/runs",
            json={
                "model_id": "00000000-0000-0000-0000-000000000000",
                "subset_pct": 10,
            },
        )
        assert response.status_code == 404


class TestAuthAPI:
    """Test auth endpoints."""

    def test_auth_status(self, client):
        """Test auth status endpoint."""
        response = client.get("/api/auth/status")
        assert response.status_code == 200
        data = response.json()
        assert "authenticated" in data
        assert "idp_configured" in data

    def test_login_redirects(self, client):
        """Test login endpoint redirects to OAuth provider."""
        response = client.get("/api/auth/login", follow_redirects=False)
        # Should redirect to Chutes IDP
        assert response.status_code in [302, 307]


class TestCORS:
    """Test CORS headers."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses."""
        response = client.options(
            "/api/models",
            headers={
                "Origin": "https://chutes-bench-runner-ui.onrender.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestExportValidation:
    """Test export functionality."""

    def test_export_format_validation(self, client):
        """Test export format validation."""
        response = client.get(
            "/api/runs/00000000-0000-0000-0000-000000000000/export?format=invalid"
        )
        assert response.status_code == 422

    def test_export_run_not_found(self, client):
        """Test exporting non-existent run."""
        response = client.get(
            "/api/runs/00000000-0000-0000-0000-000000000000/export?format=csv"
        )
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
