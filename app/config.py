"""NEXTGEN AI v20 — Web App Configuration.

Uses environment variables with safe defaults so it works locally (SQLite)
and on Vercel (Neon Postgres).
"""
import os
from functools import lru_cache


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Settings:
    # --- Database ---
    # Local default: SQLite file. On Vercel/Neon set DATABASE_URL to the
    # Postgres connection string (the app auto-switches to psycopg2 driver).
    database_url: str = _env("DATABASE_URL", "sqlite:///./nextgen.db")

    # --- Security ---
    jwt_secret: str = _env("JWT_SECRET", "nextgen-change-me-in-production-1234567890")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = int(_env("JWT_EXPIRE_MINUTES", "1440"))

    # --- Model backend ---
    # The deployed site cannot run the 9 GB Ollama model itself. It connects
    # back to YOUR machine's Ollama through a tunnel (e.g. Cloudflare Tunnel).
    # Local default is localhost; set OLLAMA_URL to your tunnel URL in Vercel.
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    default_model: str = _env("DEFAULT_MODEL", "nextgen-trained")
    default_system_prompt: str = _env(
        "DEFAULT_SYSTEM_PROMPT",
        "You are NEXTGEN AI v20 — a fully autonomous AI software engineering "
        "operating system, combining ChatGPT conversation, opencode tool-use, "
        "and Claude deep reasoning. Be concise, direct and helpful. Use "
        "markdown for formatting.",
    )

    # --- Admin ---
    admin_email: str = _env("ADMIN_EMAIL", "admin@nextgen.ai")
    admin_password: str = _env("ADMIN_PASSWORD", "admin12345")

    # --- Misc ---
    app_name: str = "NEXTGEN AI v20"
    cors_origins: list = _env("CORS_ORIGINS", "*").split(",")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
