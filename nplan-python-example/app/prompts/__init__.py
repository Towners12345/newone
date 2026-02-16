"""
Prompt Templates — Domain-grounded prompts for construction AI.

INTERVIEW TALKING POINT:
This is EXACTLY the same pattern as your generators.js trade profiles
and prompt builders, but in Python. You can directly reference this:

"I've already built production prompt templates with domain guardrails
 for 10+ construction trades. The pattern is identical in Python — I'd
 bring that same discipline around hallucination prevention and domain
 grounding to nPlan's prompts for Barry and Schedule Studio."
"""


SYSTEM_PROMPT = """You are an expert construction project risk analyst with deep 
knowledge of capital project delivery, scheduling, and risk management.

You analyse construction schedules and identify risks based on:
- Activity dependencies and critical path analysis
- Historical patterns from similar projects
- Trade-specific risk factors (weather sensitivity, supply chain, regulatory)
- Resource constraints and bottleneck identification

CRITICAL GUARDRAILS (same principle as ATP trade profiles):
- Do NOT fabricate statistics or probability values. Use qualitative assessments 
  if you lack data for quantitative ones.
- Do NOT invent specific delay durations. Use ranges based on typical project 
  experience (e.g., "2-4 weeks" not "exactly 17 days").
- Do NOT make assumptions about site conditions, ground conditions, or local 
  regulations unless explicitly provided.
- Reference UK construction standards and practices (CDM 2015, NEC4 contracts, 
  BS standards) only when genuinely applicable.
- When uncertain, say "this requires further investigation" rather than guessing.

Respond ONLY with valid JSON matching the requested schema. No markdown, no 
preamble, no explanations outside the JSON structure."""


def build_risk_analysis_prompt(
    project_name: str,
    sector: str,
    description: str,
    activities_summary: str,
    critical_path_summary: str,
) -> str:
    """
    Build a structured prompt for schedule risk analysis.

    This mirrors your renderContext() + buildSummaryMessages() pattern
    from generators.js — assembling context into a structured prompt.
    """
    return f"""Analyse the following construction schedule for risks.

PROJECT: {project_name}
SECTOR: {sector}
DESCRIPTION: {description}

ACTIVITIES SUMMARY:
{activities_summary}

CRITICAL PATH:
{critical_path_summary}

Respond with JSON in this exact structure:
{{
    "overall_risk_level": "low|medium|high|critical",
    "confidence_score": 0.0 to 1.0,
    "key_risks": ["risk 1", "risk 2", ...],
    "recommendations": ["recommendation 1", ...],
    "summary": "2-3 sentence overall assessment",
    "activity_risks": [
        {{
            "activity_id": "id",
            "activity_name": "name",
            "risk_level": "low|medium|high|critical",
            "delay_probability": 0.0 to 1.0,
            "estimated_delay_days": integer,
            "risk_factors": ["factor 1", ...],
            "mitigation_suggestions": ["suggestion 1", ...]
        }}
    ]
}}"""


def build_schedule_summary(activities: list) -> str:
    """
    Summarise activities for prompt injection.

    Same pattern as your briefMaterials() function in generators.js —
    condensing a data list into a compact text representation for the LLM.
    """
    lines = []
    for a in activities[:50]:  # Cap to prevent token overflow
        deps = f" (depends on: {', '.join(a.predecessors)})" if a.predecessors else ""
        trade = f" [{a.trade}]" if a.trade else ""
        lines.append(
            f"- {a.id}: {a.name} ({a.duration_days} days){trade}{deps}"
        )
    return "\n".join(lines)
