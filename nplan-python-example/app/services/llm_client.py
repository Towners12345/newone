"""
LLM Client — Anthropic API integration for AI-powered analysis.

INTERVIEW TALKING POINT:
This is the Python equivalent of your PHP AI calls in the ATP pipeline
and your JS generators.js prompt builders. Same pattern:
1. Build structured prompt with domain context
2. Call LLM API
3. Parse structured response
4. Validate output (prevent hallucination)

"I've built production LLM integrations in PHP and JavaScript.
 The Python version uses the same patterns — structured prompts,
 JSON response parsing, output validation — just with better typing
 through Pydantic and cleaner async handling through FastAPI."
"""

import json
import logging
from typing import Optional

from app.config import get_settings
from app.models import (
    ScheduleInput,
    ActivityRisk,
    RiskLevel,
    ScheduleRiskReport,
    CriticalPathInfo,
)
from app.prompts import SYSTEM_PROMPT, build_risk_analysis_prompt, build_schedule_summary

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Handles all LLM interactions for schedule risk analysis.

    Equivalent to your ATP's AI pipeline — but in Python using
    the Anthropic SDK directly.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    @property
    def client(self):
        """Lazy-initialise the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.settings.anthropic_api_key
                )
            except Exception as e:
                logger.warning(f"Failed to initialise Anthropic client: {e}")
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if LLM is configured and available."""
        return bool(self.settings.anthropic_api_key and self.client)

    async def analyse_schedule_risks(
        self,
        schedule: ScheduleInput,
        critical_path: CriticalPathInfo,
    ) -> Optional[dict]:
        """
        Send schedule data to LLM for risk analysis.

        This mirrors your generate-quote-progress.php pipeline:
        1. Build context from input data
        2. Construct structured prompt
        3. Call LLM
        4. Parse and validate response
        """
        if not self.is_available:
            logger.info("LLM not available — returning None")
            return None

        # Build the prompt (same pattern as your renderContext + generators)
        activities_summary = build_schedule_summary(schedule.activities)
        critical_path_summary = (
            f"Critical path: {' → '.join(critical_path.path_activities)}\n"
            f"Total duration: {critical_path.total_duration_days} days\n"
            f"Bottleneck: {critical_path.bottleneck_activity or 'None identified'}"
        )

        prompt = build_risk_analysis_prompt(
            project_name=schedule.project_name,
            sector=schedule.sector.value,
            description=schedule.description,
            activities_summary=activities_summary,
            critical_path_summary=critical_path_summary,
        )

        try:
            # Call the Anthropic API
            response = self.client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response
            raw_text = response.content[0].text

            # Parse JSON response — same as your JSON.parse in the frontend
            # but with better error handling
            parsed = self._parse_llm_json(raw_text)

            if parsed:
                # Validate the LLM output against our schema
                # This is the "hallucination prevention" step
                return self._validate_llm_output(parsed, schedule)

            return None

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None

    def _parse_llm_json(self, text: str) -> Optional[dict]:
        """
        Safely parse JSON from LLM response.

        Same pattern as your extractJobResult() and the JSON.parse
        try/catch blocks throughout your React code — LLMs don't
        always return clean JSON.
        """
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            # Try to find JSON object in the response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            return None

    def _validate_llm_output(
        self, data: dict, schedule: ScheduleInput
    ) -> dict:
        """
        Validate LLM output against known constraints.

        THIS IS YOUR HALLUCINATION PREVENTION — same concept as your
        unit-corrections.php and completeness audit. The LLM might
        return activity IDs that don't exist, risk levels outside
        valid values, or probabilities > 1.0.
        """
        valid_activity_ids = {a.id for a in schedule.activities}
        valid_risk_levels = {"low", "medium", "high", "critical"}

        # Validate overall risk level
        if data.get("overall_risk_level") not in valid_risk_levels:
            data["overall_risk_level"] = "medium"

        # Clamp confidence score
        conf = data.get("confidence_score", 0.5)
        data["confidence_score"] = max(0.0, min(1.0, float(conf)))

        # Validate individual activity risks
        validated_risks = []
        for risk in data.get("activity_risks", []):
            # Only include risks for activities that actually exist
            if risk.get("activity_id") in valid_activity_ids:
                # Clamp probability values
                risk["delay_probability"] = max(
                    0.0, min(1.0, float(risk.get("delay_probability", 0.5)))
                )
                risk["estimated_delay_days"] = max(
                    0, int(risk.get("estimated_delay_days", 0))
                )
                if risk.get("risk_level") not in valid_risk_levels:
                    risk["risk_level"] = "medium"
                validated_risks.append(risk)

        data["activity_risks"] = validated_risks

        # Ensure lists are actually lists
        for key in ("key_risks", "recommendations"):
            if not isinstance(data.get(key), list):
                data[key] = []

        return data
