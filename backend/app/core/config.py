import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    vector_store_path: str = "data/vector_store"
    config_path: str = "data/vector_store/config.json"

    ollama_model: str = "llama3.2:latest"
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.0

    frontend_origin: str = "http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


def load_pipeline_config() -> dict:
    """Load the pipeline configuration exported during Phase 2.7."""

    config_file = Path(settings.config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Pipeline config not found at {config_file}. "
            "Make sure data/vector_store/config.json exists."
        )

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)