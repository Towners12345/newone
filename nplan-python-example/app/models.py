"""
Data Models — Pydantic schemas for schedule and risk data.

IMPORTANT FOR YOUR INTERVIEW:
Pydantic is to Python what your PHP type-checking + validation is, but built
into the language. At nPlan, every piece of data flowing through the system
(schedules, activities, risk forecasts) would be defined as Pydantic models.

This is equivalent to:
- Your PHP atp_normalize_job_payload() function → Pydantic does this automatically
- Your JS extractJobResult() parser → Pydantic validates on construction
- Your trade profile schemas → Pydantic models with validators

Key concept: "Parse, don't validate" — instead of checking data after the fact,
Pydantic creates valid objects or raises clear errors. No more silent corruption.
"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import date
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Enums — like your TRADE_PROFILES keys, but type-safe
# ──────────────────────────────────────────────────────────────

class ProjectSector(str, Enum):
    """Construction sectors nPlan covers."""
    HIGHWAYS = "highways"
    RAIL = "rail"
    COMMERCIAL_BUILDINGS = "commercial_buildings"
    PUBLIC_BUILDINGS = "public_buildings"
    HEAVY_INFRASTRUCTURE = "heavy_infrastructure"
    OIL_AND_GAS = "oil_and_gas"
    UTILITIES = "utilities"
    DATA_CENTERS = "data_centers"
    RESIDENTIAL = "residential"
    ENERGY = "energy"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActivityStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"


# ──────────────────────────────────────────────────────────────
# Schedule Models — represents nPlan's core data structures
# ──────────────────────────────────────────────────────────────

class ScheduleActivity(BaseModel):
    """
    A single activity in a construction schedule.

    At nPlan, each schedule has 10,000+ of these forming a directed
    acyclic graph (DAG). The relationships between them are what their
    GNN models analyse for risk prediction.

    This is similar to a single material line in your ATP quote —
    but instead of materials, it's tasks with durations and dependencies.
    """
    id: str = Field(..., description="Unique activity identifier")
    name: str = Field(..., min_length=1, max_length=500)
    duration_days: int = Field(..., ge=0, description="Planned duration in days")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: ActivityStatus = ActivityStatus.NOT_STARTED

    # Dependencies — the graph edges that GNNs analyse
    predecessors: list[str] = Field(default_factory=list)
    successors: list[str] = Field(default_factory=list)

    # Resource info
    trade: Optional[str] = None  # e.g., "groundworks", "steelwork", "M&E"
    resource_hours: Optional[float] = None

    @field_validator("name")
    @classmethod
    def clean_activity_name(cls, v: str) -> str:
        """
        Strip excess whitespace — same pattern as your stripTags()
        and __atp_tokenize() functions, but using Python decorators.
        """
        return " ".join(v.split())


class ScheduleInput(BaseModel):
    """
    Input payload for analysing a construction schedule.

    This is the Python equivalent of your atp_normalize_job_payload() —
    it takes messy input and produces clean, validated data.

    The difference: Pydantic does this AUTOMATICALLY from the type hints.
    No manual parsing code needed.
    """
    project_name: str = Field(..., min_length=1, max_length=200)
    sector: ProjectSector
    description: str = Field("", max_length=5000)
    total_budget_gbp: Optional[float] = Field(None, ge=0)
    target_completion: Optional[date] = None
    activities: list[ScheduleActivity] = Field(
        ...,
        min_length=1,
        description="At least one activity required"
    )

    @field_validator("activities")
    @classmethod
    def validate_dependency_references(cls, activities: list[ScheduleActivity]):
        """
        Ensure all predecessor/successor IDs reference real activities.
        This is graph integrity validation — exactly what nPlan's
        Schedule Integrity Checker does.
        """
        valid_ids = {a.id for a in activities}
        for activity in activities:
            for pred_id in activity.predecessors:
                if pred_id not in valid_ids:
                    raise ValueError(
                        f"Activity '{activity.id}' references unknown "
                        f"predecessor '{pred_id}'"
                    )
        return activities


# ──────────────────────────────────────────────────────────────
# Risk Output Models — what the service returns
# ──────────────────────────────────────────────────────────────

class ActivityRisk(BaseModel):
    """Risk assessment for a single activity."""
    activity_id: str
    activity_name: str
    risk_level: RiskLevel
    delay_probability: float = Field(..., ge=0, le=1)
    estimated_delay_days: int = Field(0, ge=0)
    risk_factors: list[str] = Field(default_factory=list)
    mitigation_suggestions: list[str] = Field(default_factory=list)


class CriticalPathInfo(BaseModel):
    """Information about the schedule's critical path."""
    path_activities: list[str]  # Activity IDs on critical path
    total_duration_days: int
    bottleneck_activity: Optional[str] = None
    path_risk_level: RiskLevel = RiskLevel.MEDIUM


class ScheduleRiskReport(BaseModel):
    """
    Complete risk analysis report for a schedule.

    This is the output equivalent of your ATP quote result —
    a structured response containing multiple sections of analysis.
    """
    project_name: str
    sector: ProjectSector
    overall_risk_level: RiskLevel
    confidence_score: float = Field(..., ge=0, le=1)

    # Detailed analysis
    activity_risks: list[ActivityRisk]
    critical_path: CriticalPathInfo
    summary: str
    key_risks: list[str]
    recommendations: list[str]

    # Metadata
    analysis_model: str = "schedule-risk-v1"
    activities_analysed: int = 0


# ──────────────────────────────────────────────────────────────
# Health / Status Models
# ──────────────────────────────────────────────────────────────

class HealthCheck(BaseModel):
    """Service health status — every production API needs this."""
    status: str = "healthy"
    version: str
    llm_available: bool = False
