from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Multimodal Search Platform"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # LLM Provider
    llm_provider: Literal["google_ai_studio", "vertex_ai"] = "google_ai_studio"
    gemini_api_key: str | None = Field(default=None, description="API Key for Google AI Studio")
    gemini_model: str = "gemini-1.5-flash"
    vertex_project: str | None = None
    vertex_location: str = "us-central1"

    # Embedding Model (MobileCLIP2-S0)
    model_name: str = "MobileCLIP2-S0"
    pretrained_tag: str = Field(
        default="dfndr2b",
        description="Pretrained tag for OpenCLIP MobileCLIP2-S0 (dfndr2b corresponds to DataComp-DFN-2B).",
    )
    embedding_dim: int = 512
    device: str = "auto"
    checkpoint_dir: Path = Path("./checkpoints")

    # Qdrant Vector DB
    qdrant_path: str | None = "./data/qdrant_storage"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection_name: str = "fashion_catalog"

    # Search & Multi-Agent Reflection Thresholds
    min_confidence_score: float = 0.70
    top_k_candidates: int = 10
    max_reflection_retries: int = 2


settings = Settings()
