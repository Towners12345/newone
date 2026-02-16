"""
Main FastAPI Application — Schedule Risk Analyser API.

INTERVIEW TALKING POINT:
"FastAPI routes are functionally identical to WordPress REST API
 endpoints that I've built for ATP. The key differences are:
 - Type hints replace manual parameter validation
 - Pydantic models replace my atp_normalize_job_payload() function
 - async/await replaces PHP's synchronous processing
 - Auto-generated docs replace my manual API documentation

 The patterns are the same: receive request, validate input, process
 through a pipeline, return structured response. I'd be productive
 in this environment very quickly."

To run:
    uvicorn app.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs     — Interactive Swagger UI
    http://localhost:8000/redoc    — Clean API documentation
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import (
    ScheduleInput,
    ScheduleRiskReport,
    HealthCheck,
    ActivityRisk,
    RiskLevel,
)
from app.services.risk_engine import RiskEngine
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# App lifecycle — runs on startup/shutdown
# ──────────────────────────────────────────────────────────────

llm_client = LLMClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hooks.
    At nPlan, this would initialise database connections, load ML models,
    connect to Azure services, etc.
    """
    logger.info("Starting Schedule Risk Analyser...")
    yield
    logger.info("Shutting down.")


# ──────────────────────────────────────────────────────────────
# Create the FastAPI app
# ──────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered construction schedule risk analysis service. "
        "Analyses schedule graphs, identifies critical paths, and "
        "uses LLMs for intelligent risk assessment."
    ),
    lifespan=lifespan,
)

# CORS middleware — same concept as your WordPress CORS headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# API Routes — equivalent to your register_rest_route() calls
# ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health endpoint — every production service needs this.
    Azure Container Apps / Kubernetes use this to know if
    the service is alive and ready for traffic.
    """
    return HealthCheck(
        status="healthy",
        version=settings.app_version,
        llm_available=llm_client.is_available,
    )


@app.post("/api/v1/analyse", response_model=ScheduleRiskReport)
async def analyse_schedule(schedule: ScheduleInput):
    """
    Main endpoint — analyse a construction schedule for risks.

    This is the Python equivalent of your /wp-json/atp/v1/generate endpoint.
    Same flow:
    1. Receive and validate input (Pydantic does this automatically)
    2. Process through deterministic pipeline (graph analysis)
    3. Enhance with LLM analysis (AI risk assessment)
    4. Return structured response

    The key difference: FastAPI validates the input BEFORE your code runs.
    If someone sends bad data, they get a clear 422 error automatically.
    No manual checking needed.
    """
    try:
        # Step 1: Build the schedule graph
        engine = RiskEngine()
        graph = engine.build_graph(schedule)
        logger.info(
            f"Built graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        # Step 2: Validate schedule integrity
        issues = engine.validate_schedule_integrity()
        if issues:
            logger.warning(f"Schedule integrity issues: {issues}")

        # Step 3: Calculate critical path
        critical_path = engine.find_critical_path()
        logger.info(
            f"Critical path: {len(critical_path.path_activities)} activities, "
            f"{critical_path.total_duration_days} days"
        )

        # Step 4: Identify high-risk activities
        high_risk_ids = engine.find_high_risk_activities()

        # Step 5: LLM-enhanced analysis (if available)
        llm_result = await llm_client.analyse_schedule_risks(
            schedule, critical_path
        )

        # Step 6: Assemble the response
        # (Combine deterministic graph analysis with LLM insights)
        if llm_result:
            # Use LLM results, validated against our schema
            activity_risks = [
                ActivityRisk(**risk)
                for risk in llm_result.get("activity_risks", [])
            ]
            return ScheduleRiskReport(
                project_name=schedule.project_name,
                sector=schedule.sector,
                overall_risk_level=RiskLevel(
                    llm_result.get("overall_risk_level", "medium")
                ),
                confidence_score=llm_result.get("confidence_score", 0.7),
                activity_risks=activity_risks,
                critical_path=critical_path,
                summary=llm_result.get("summary", "Analysis complete."),
                key_risks=llm_result.get("key_risks", []) + issues,
                recommendations=llm_result.get("recommendations", []),
                activities_analysed=len(schedule.activities),
            )
        else:
            # Fallback: deterministic-only analysis (no LLM)
            activity_risks = [
                ActivityRisk(
                    activity_id=aid,
                    activity_name=engine.activities[aid].name,
                    risk_level=RiskLevel.HIGH if aid in high_risk_ids
                    else RiskLevel.MEDIUM,
                    delay_probability=0.6 if aid in high_risk_ids else 0.3,
                    estimated_delay_days=0,
                    risk_factors=["High dependency count"]
                    if aid in high_risk_ids else [],
                    mitigation_suggestions=[],
                )
                for aid in list(engine.activities.keys())[:20]
            ]
            return ScheduleRiskReport(
                project_name=schedule.project_name,
                sector=schedule.sector,
                overall_risk_level=critical_path.path_risk_level,
                confidence_score=0.5,
                activity_risks=activity_risks,
                critical_path=critical_path,
                summary=(
                    f"Deterministic analysis of {len(schedule.activities)} "
                    f"activities. Critical path duration: "
                    f"{critical_path.total_duration_days} days. "
                    f"LLM analysis unavailable."
                ),
                key_risks=issues or ["No major structural issues detected"],
                recommendations=[
                    "Enable LLM analysis for detailed risk assessment"
                ],
                activities_analysed=len(schedule.activities),
            )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/validate")
async def validate_schedule(schedule: ScheduleInput):
    """
    Validate a schedule's structural integrity without full analysis.

    Like nPlan's free Schedule Integrity Checker — a lighter-weight
    endpoint that checks for common problems.
    """
    engine = RiskEngine()
    engine.build_graph(schedule)
    issues = engine.validate_schedule_integrity()

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "activity_count": len(schedule.activities),
        "dependency_count": engine.graph.number_of_edges()
        if engine.graph else 0,
    }
