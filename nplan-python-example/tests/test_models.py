"""
Model Tests — Validates Pydantic schemas work correctly.

INTERVIEW TALKING POINT:
"Testing in Python uses pytest, which is more concise than PHP's
 PHPUnit. The key pattern is the same: arrange test data, act on it,
 assert the result. Pydantic makes testing easier because invalid
 data raises clear errors — you test that valid data works AND that
 invalid data is properly rejected."
"""

import pytest
from datetime import date
from pydantic import ValidationError

from app.models import (
    ScheduleActivity,
    ScheduleInput,
    ProjectSector,
    ActivityStatus,
)


# ──────────────────────────────────────────────────────────────
# Fixtures — reusable test data (like setUp() in PHPUnit)
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_activity():
    """A valid activity for testing."""
    return ScheduleActivity(
        id="ACT-001",
        name="Excavation and earthworks",
        duration_days=14,
        status=ActivityStatus.NOT_STARTED,
        trade="groundworks",
        predecessors=[],
        successors=["ACT-002"],
    )


@pytest.fixture
def sample_schedule():
    """A minimal valid schedule for testing."""
    return ScheduleInput(
        project_name="Highway Bridge Replacement",
        sector=ProjectSector.HIGHWAYS,
        description="Replace aging concrete bridge on A590",
        activities=[
            ScheduleActivity(
                id="ACT-001",
                name="Site setup and traffic management",
                duration_days=7,
                trade="general",
            ),
            ScheduleActivity(
                id="ACT-002",
                name="Demolition of existing bridge",
                duration_days=21,
                trade="demolition",
                predecessors=["ACT-001"],
            ),
            ScheduleActivity(
                id="ACT-003",
                name="Foundation piling",
                duration_days=28,
                trade="groundworks",
                predecessors=["ACT-002"],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────
# Activity Model Tests
# ──────────────────────────────────────────────────────────────

class TestScheduleActivity:
    """Tests for the ScheduleActivity model."""

    def test_valid_activity_creation(self, sample_activity):
        """Valid data should create an activity without errors."""
        assert sample_activity.id == "ACT-001"
        assert sample_activity.duration_days == 14
        assert sample_activity.trade == "groundworks"

    def test_activity_name_whitespace_cleaning(self):
        """Names with excess whitespace should be cleaned."""
        activity = ScheduleActivity(
            id="ACT-001",
            name="  Excavation   and   earthworks  ",
            duration_days=10,
        )
        assert activity.name == "Excavation and earthworks"

    def test_activity_rejects_empty_name(self):
        """Empty names should be rejected."""
        with pytest.raises(ValidationError):
            ScheduleActivity(id="ACT-001", name="", duration_days=10)

    def test_activity_rejects_negative_duration(self):
        """Negative durations are physically impossible."""
        with pytest.raises(ValidationError):
            ScheduleActivity(
                id="ACT-001", name="Test", duration_days=-5
            )

    def test_activity_zero_duration_allowed(self):
        """Zero duration is valid (milestones)."""
        activity = ScheduleActivity(
            id="MS-001", name="Project milestone", duration_days=0
        )
        assert activity.duration_days == 0

    def test_activity_default_status(self):
        """Default status should be NOT_STARTED."""
        activity = ScheduleActivity(
            id="ACT-001", name="Test", duration_days=10
        )
        assert activity.status == ActivityStatus.NOT_STARTED


# ──────────────────────────────────────────────────────────────
# Schedule Model Tests
# ──────────────────────────────────────────────────────────────

class TestScheduleInput:
    """Tests for the ScheduleInput model."""

    def test_valid_schedule_creation(self, sample_schedule):
        """Valid schedule should be created without errors."""
        assert sample_schedule.project_name == "Highway Bridge Replacement"
        assert len(sample_schedule.activities) == 3

    def test_schedule_requires_activities(self):
        """Schedule must have at least one activity."""
        with pytest.raises(ValidationError):
            ScheduleInput(
                project_name="Empty Project",
                sector=ProjectSector.HIGHWAYS,
                activities=[],
            )

    def test_schedule_rejects_invalid_dependencies(self):
        """References to non-existent activities should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ScheduleInput(
                project_name="Bad Deps Project",
                sector=ProjectSector.RAIL,
                activities=[
                    ScheduleActivity(
                        id="ACT-001",
                        name="Task A",
                        duration_days=10,
                        predecessors=["NONEXISTENT"],  # This doesn't exist!
                    ),
                ],
            )
        assert "unknown predecessor" in str(exc_info.value).lower()

    def test_schedule_valid_dependencies_accepted(self):
        """Valid dependency references should work fine."""
        schedule = ScheduleInput(
            project_name="Valid Deps",
            sector=ProjectSector.COMMERCIAL_BUILDINGS,
            activities=[
                ScheduleActivity(id="A", name="First", duration_days=5),
                ScheduleActivity(
                    id="B", name="Second", duration_days=10,
                    predecessors=["A"],
                ),
            ],
        )
        assert schedule.activities[1].predecessors == ["A"]

    def test_sector_enum_validation(self):
        """Invalid sectors should be rejected."""
        with pytest.raises(ValidationError):
            ScheduleInput(
                project_name="Test",
                sector="space_station",  # Not a valid sector!
                activities=[
                    ScheduleActivity(id="A", name="Task", duration_days=1),
                ],
            )
