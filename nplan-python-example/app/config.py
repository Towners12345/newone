"""
Configuration — Environment-aware settings using Pydantic.

This is how modern Python apps handle config. Instead of PHP constants
or WordPress get_option(), you define a typed Settings class that
automatically reads from environment variables.

At nPlan, this would pull secrets from Azure Key Vault in production
and from .env files in development.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings — auto-loaded from environment variables."""

    # API Keys
    anthropic_api_key: str = ""

    # Service config
    app_name: str = "Schedule Risk Analyser"
    app_version: str = "0.1.0"
    debug: bool = False

    # LLM config
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.3  # Low temp = more consistent/factual

    # Azure config (for production deployment)
    azure_keyvault_url: str = ""
    azure_storage_connection: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings loader.

    lru_cache means this only runs once — the Settings object is then
    reused for every request. This is a common Python pattern you'll
    see everywhere at companies like nPlan.
    """
    return Settings()
