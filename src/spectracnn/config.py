"""Runtime configuration (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from ``SPECTRACNN_*`` environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="SPECTRACNN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Path("models/spectracnn.pt")
    meta_path: Path = Path("models/spectracnn.json")
    web_dir: Path = Path("web")
    api_key: str | None = None
    cors_origins: str = "*"
    log_level: str = "INFO"
    model_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
