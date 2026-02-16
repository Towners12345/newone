"""
Risk Engine Tests — Tests for graph-based schedule analysis.

These test the deterministic logic (no LLM calls needed).
At nPlan, this kind of test ensures the graph algorithms work
correctly before ML models are applied on top.
"""

import pytest

from app.models import ScheduleActivity, ScheduleInput, ProjectSector, RiskLevel
from app.services.risk_engine import RiskEngine


@pytest.fixture
def linear_schedule():
    """A simple linear schedule: A → B → C."""
    return ScheduleInput(
        project_name="Linear Project",
        sector=ProjectSector.COMMERCIAL_BUILDINGS,
        activities=[
            ScheduleActivity(
                id="A", name="Foundations", duration_days=20, trade="groundworks"
            ),
            ScheduleActivity(
                id="B", name="Superstructure", duration_days=40,
                trade="steelwork", predecessors=["A"]
            ),
            ScheduleActivity(
                id="C", name="Fit-out", duration_days=30,
                trade="M&E", predecessors=["B"]
            ),
        ],
    )


@pytest.fixture
def parallel_schedule():
    """
    A schedule with parallel paths:
    A → B → D
    A → C → D
    """
    return ScheduleInput(
        project_name="Parallel Project",
        sector=ProjectSector.RAIL,
        activities=[
            ScheduleActivity(id="A", name="Site setup", duration_days=10),
            ScheduleActivity(
                id="B", name="Track work", duration_days=60,
                trade="rail", predecessors=["A"]
            ),
            ScheduleActivity(
                id="C", name="Signalling", duration_days=30,
                trade="electrical", predecessors=["A"]
            ),
            ScheduleActivity(
                id="D", name="Testing", duration_days=14,
                trade="general", predecessors=["B", "C"]
            ),
        ],
    )


class TestRiskEngineGraphBuilding:
    """Tests for building the schedule graph."""

    def test_graph_node_count(self, linear_schedule):
        engine = RiskEngine()
        graph = engine.build_graph(linear_schedule)
        assert graph.number_of_nodes() == 3

    def test_graph_edge_count(self, linear_schedule):
        engine = RiskEngine()
        graph = engine.build_graph(linear_schedule)
        # A→B and B→C = 2 edges
        assert graph.number_of_edges() == 2

    def test_parallel_graph_edges(self, parallel_schedule):
        engine = RiskEngine()
        graph = engine.build_graph(parallel_schedule)
        # A→B, A→C, B→D, C→D = 4 edges
        assert graph.number_of_edges() == 4

    def test_activities_indexed(self, linear_schedule):
        engine = RiskEngine()
        engine.build_graph(linear_schedule)
        assert "A" in engine.activities
        assert engine.activities["A"].name == "Foundations"


class TestCriticalPath:
    """Tests for critical path calculation."""

    def test_linear_critical_path(self, linear_schedule):
        """In a linear schedule, ALL activities are on the critical path."""
        engine = RiskEngine()
        engine.build_graph(linear_schedule)
        cp = engine.find_critical_path()

        assert cp.path_activities == ["A", "B", "C"]
        assert cp.total_duration_days == 90  # 20 + 40 + 30

    def test_parallel_critical_path(self, parallel_schedule):
        """Critical path should be the LONGEST path through the graph."""
        engine = RiskEngine()
        engine.build_graph(parallel_schedule)
        cp = engine.find_critical_path()

        # Path A→B→D = 10+60+14 = 84 days (longer)
        # Path A→C→D = 10+30+14 = 54 days (shorter)
        assert cp.total_duration_days == 84
        assert "B" in cp.path_activities  # Track work is on critical path
        assert "C" not in cp.path_activities  # Signalling is NOT

    def test_bottleneck_identification(self, linear_schedule):
        """Bottleneck should be the longest activity on the critical path."""
        engine = RiskEngine()
        engine.build_graph(linear_schedule)
        cp = engine.find_critical_path()

        assert cp.bottleneck_activity == "B"  # Superstructure at 40 days

    def test_empty_schedule_handled(self):
        """Single-activity schedule should work."""
        schedule = ScheduleInput(
            project_name="Tiny Project",
            sector=ProjectSector.RESIDENTIAL,
            activities=[
                ScheduleActivity(id="X", name="Everything", duration_days=100),
            ],
        )
        engine = RiskEngine()
        engine.build_graph(schedule)
        cp = engine.find_critical_path()

        assert cp.total_duration_days == 100


class TestScheduleIntegrity:
    """Tests for schedule integrity validation."""

    def test_valid_schedule_no_issues(self, linear_schedule):
        engine = RiskEngine()
        engine.build_graph(linear_schedule)
        issues = engine.validate_schedule_integrity()
        assert len(issues) == 0

    def test_detects_zero_duration_activities(self):
        schedule = ScheduleInput(
            project_name="Zero Duration Test",
            sector=ProjectSector.UTILITIES,
            activities=[
                ScheduleActivity(id="M1", name="Milestone", duration_days=0),
                ScheduleActivity(
                    id="A", name="Work", duration_days=10,
                    predecessors=["M1"]
                ),
            ],
        )
        engine = RiskEngine()
        engine.build_graph(schedule)
        issues = engine.validate_schedule_integrity()

        # Should flag the zero-duration activity
        assert any("zero duration" in issue.lower() for issue in issues)

    def test_detects_isolated_activities(self):
        schedule = ScheduleInput(
            project_name="Isolated Test",
            sector=ProjectSector.ENERGY,
            activities=[
                ScheduleActivity(id="A", name="Connected", duration_days=10),
                ScheduleActivity(
                    id="B", name="Also connected", duration_days=5,
                    predecessors=["A"]
                ),
                ScheduleActivity(id="C", name="Isolated", duration_days=20),
            ],
        )
        engine = RiskEngine()
        engine.build_graph(schedule)
        issues = engine.validate_schedule_integrity()

        assert any("disconnected" in issue.lower() for issue in issues)


class TestHighRiskActivities:
    """Tests for risk identification."""

    def test_identifies_high_risk_activities(self, parallel_schedule):
        """Long-duration and high-dependency activities should be flagged."""
        engine = RiskEngine()
        engine.build_graph(parallel_schedule)
        risky = engine.find_high_risk_activities()

        # Activity B (60 days, longest duration) should be highest risk
        assert "B" in risky
        # Should return at least 1 risky activity
        assert len(risky) >= 1
