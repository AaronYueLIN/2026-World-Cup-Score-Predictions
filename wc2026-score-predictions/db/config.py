"""Database config -- compatibility layer (production-grade: real implementation in db/settings.py)

Kept so existing imports don't break.
New code should use from db.settings import settings directly.
"""
from __future__ import annotations

from db.settings import settings

APP_ENV: str = settings.app_env
DATABASE_URL: str = settings.database_url
IS_SQLITE: bool = settings.is_sqlite
IS_PRODUCTION: bool = settings.is_production
IS_DEVELOPMENT: bool = settings.is_development
ENGINE_KWARGS: dict = settings.engine_kwargs
