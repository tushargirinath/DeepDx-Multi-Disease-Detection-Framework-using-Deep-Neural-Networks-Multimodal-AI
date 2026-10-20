"""
Central configuration. Reads from environment / .env via pydantic-settings.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/config.py -> app -> root)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Groq Cloud (Primary LLM) ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # --- Legacy keys (optional) ---
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5"
    openrouter_api_key: str = ""
    openrouter_model: str = "minimax/minimax-m3.1-flash:free"

    # --- Model files ---
    cnn_weights_path: str = "models/best_model.weights.h5"
    class_names_path: str = "class_names.json"
    img_size: int = 224

    # --- Contextual-memory sources ---
    enable_biogpt: bool = False
    enable_medpalm: bool = False

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------- Derived helpers ----------
    @property
    def weights_file(self) -> Path:
        p = Path(self.cnn_weights_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def class_names_file(self) -> Path:
        p = Path(self.class_names_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.startswith("gsk_"))

    @property
    def has_claude(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.startswith("sk-ant-"))

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.startswith("sk-or-"))

    def load_class_names(self) -> list[str]:
        """Load the ordered class-name list. Falls back to a generated list."""
        f = self.class_names_file
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                names = json.load(fh)
            if isinstance(names, list) and names:
                return [str(n) for n in names]
        # Last-resort fallback so the app never hard-crashes.
        return [f"class_{i}" for i in range(30)]


@lru_cache
def get_settings() -> Settings:
    return Settings()
