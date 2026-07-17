from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_PROJECT_ROOT / ".env", _BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    primary_model: str = "groq/openai/gpt-oss-120b"
    fallback_model: str = "gemini/gemini-3.5-flash"
    fallback_model_2: str = "gemini/gemini-3.1-flash-lite"

    agent_max_steps: int = 8
    agent_max_critiques: int = 2
    tool_timeout_seconds: int = 20

    taskforge_api_port: int = 8001
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    taskforge_db: str = "tasks.db"
    agent_db: str = "agent.db"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def backend_root(self) -> Path:
        return _BACKEND_ROOT

    @property
    def db_path(self) -> Path:
        p = Path(self.taskforge_db)
        return p if p.is_absolute() else _BACKEND_ROOT / p

    @property
    def agent_db_path(self) -> Path:
        p = Path(self.agent_db)
        return p if p.is_absolute() else _BACKEND_ROOT / p

    @property
    def mcp_server_script(self) -> Path:
        return _BACKEND_ROOT / "mcp_server" / "server.py"


@lru_cache
def get_settings() -> Settings:
    return Settings()
