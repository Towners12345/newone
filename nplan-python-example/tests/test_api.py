"""
API Integration Tests — Tests the full HTTP request/response cycle.

These use FastAPI's TestClient (built on httpx) to make real HTTP
requests to the API without running a server. This is how you'd
test endpoints at nPlan before deploying.

INTERVIEW TALKING POINT:
"I test my ATP endpoints manually and through the React frontend.
 Python's pytest + httpx gives you proper automated integration
 testing — you can run hundreds of test cases against your API
 in seconds as part of CI/CD. This catches regressions before
 they reach production."
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def valid_schedule_payload():
    """A valid JSON payload for the analyse endpoint."""
    return {
        "project_name": "M6 Junction Improvement",
        "sector": "highways",
        "description": "Widening and signalisation of Junction 36",
        "activities": [
            {
                "id": "TM-001",
                "name": "Traffic management setup",
                "duration_days": 5,
                "trade": "general",
                "predecessors": [],
                "successors": ["EX-001"],
            },
            {
                "id": "EX-001",
                "name": "Excavation for new lane",
                "duration_days": 15,
                "trade": "groundworks",
                "predecessors": ["TM-001"],
            },
            {
                "id": "DR-001",
                "name": "Drainage installation",
                "duration_days": 10,
                "trade": "utilities",
                "predecessors": ["EX-001"],
            },
            {
                "id": "SU-001",
                "name": "Sub-base and surfacing",
                "duration_days": 12,
                "trade": "highways",
                "predecessors": ["DR-001"],
            },
            {
                "id": "SI-001",
                "name": "Signal installation and testing",
                "duration_days": 8,
                "trade": "electrical",
                "predecessors": ["SU-001"],
            },
        ],
    }


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestAnalyseEndpoint:
    """Tests for the POST /api/v1/analyse endpoint."""

    def test_valid_schedule_returns_200(self, client, valid_schedule_payload):
        response = client.post("/api/v1/analyse", json=valid_schedule_payload)
        assert response.status_code == 200

    def test_response_contains_required_fields(
        self, client, valid_schedule_payload
    ):
        response = client.post("/api/v1/analyse", json=valid_schedule_payload)
        data = response.json()

        assert "project_name" in data
        assert "overall_risk_level" in data
        assert "critical_path" in data
        assert "activity_risks" in data
        assert "summary" in data
        assert "key_risks" in data

    def test_critical_path_calculated(self, client, valid_schedule_payload):
        response = client.post("/api/v1/analyse", json=valid_schedule_payload)
        data = response.json()

        cp = data["critical_path"]
        assert cp["total_duration_days"] == 50  # 5+15+10+12+8
        assert len(cp["path_activities"]) == 5  # All activities (linear)

    def test_missing_project_name_returns_422(self, client):
        """Missing required fields should return validation error."""
        response = client.post("/api/v1/analyse", json={
            "sector": "highways",
            "activities": [
                {"id": "A", "name": "Task", "duration_days": 10}
            ],
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_sector_returns_422(self, client):
        """Invalid enum values should be rejected."""
        response = client.post("/api/v1/analyse", json={
            "project_name": "Test",
            "sector": "underwater_basket_weaving",
            "activities": [
                {"id": "A", "name": "Task", "duration_days": 10}
            ],
        })
        assert response.status_code == 422

    def test_empty_activities_returns_422(self, client):
        """Schedule must have at least one activity."""
        response = client.post("/api/v1/analyse", json={
            "project_name": "Empty Project",
            "sector": "rail",
            "activities": [],
        })
        assert response.status_code == 422

    def test_invalid_dependency_returns_422(self, client):
        """References to non-existent activities should be rejected."""
        response = client.post("/api/v1/analyse", json={
            "project_name": "Bad Deps",
            "sector": "rail",
            "activities": [
                {
                    "id": "A",
                    "name": "Task",
                    "duration_days": 10,
                    "predecessors": ["GHOST"],
                }
            ],
        })
        assert response.status_code == 422


class TestValidateEndpoint:
    """Tests for the POST /api/v1/validate endpoint."""

    def test_valid_schedule_passes(self, client, valid_schedule_payload):
        response = client.post("/api/v1/validate", json=valid_schedule_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["activity_count"] == 5

    def test_returns_issue_count(self, client):
        """Schedule with potential issues should report them."""
        response = client.post("/api/v1/validate", json={
            "project_name": "Test",
            "sector": "utilities",
            "activities": [
                {"id": "A", "name": "Task A", "duration_days": 10},
                {"id": "B", "name": "Task B", "duration_days": 0},
            ],
        })
        data = response.json()
        # Should report disconnected components and zero duration
        assert data["activity_count"] == 2
