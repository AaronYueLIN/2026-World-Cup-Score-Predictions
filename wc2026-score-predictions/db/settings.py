"""Unified config management -- Pydantic BaseSettings (production-grade)

Replaces the dual-track os.getenv approach in db/config.py.
All config read from env vars / .env, type-safe, single source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# project root (parent of db/ where settings.py lives)
_ROOT = Path(__file__).resolve().parent.parent

_DEV_DB = f"sqlite:///{_ROOT / 'quantbet.db'}"
_PROD_DB = "postgresql://quantbet:quantbet_dev@db:5432/quantbet"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime environment ---
    app_env: Literal["development", "production", "test"] = "development"

    # --- Database ---
    database_url: str = ""
    database_url_read_only: str = ""

    # --- API ---
    api_token: str = ""

    # --- Model ---
    model_version: str = "v9"

    # --- Logging ---
    log_level: str = "INFO"

    # --- LLM (DeepSeek / OpenAI compatible) ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    @model_validator(mode="after")
    def _fill_defaults(self) -> "Settings":
        if not self.database_url:
            if self.app_env == "test":
                self.database_url = "sqlite:///:memory:"
            elif self.app_env == "production":
                self.database_url = _PROD_DB
            else:
                self.database_url = _DEV_DB
        return self

    # ---------------------------------------------------------------
    # computed properties
    # ---------------------------------------------------------------
    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def database_url_read(self) -> str:
        """Read-only connection URL (uses replica if available, otherwise falls back to primary)."""
        return self.database_url_read_only or self.database_url

    @property
    def engine_kwargs(self) -> dict:
        kwargs: dict = {"echo": self.is_development, "pool_pre_ping": True}
        if self.is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["pool_size"] = 1
            kwargs["max_overflow"] = 0
        else:
            kwargs.update({
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600,
            })
        return kwargs


settings = Settings()
