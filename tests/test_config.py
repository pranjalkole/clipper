from pathlib import Path

from src.core.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.app_name == "Multimodal Search Platform"
    assert settings.app_env == "development"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.embedding_dim == 512
    assert settings.min_confidence_score == 0.70
    assert settings.top_k_candidates == 10
    assert settings.max_reflection_retries == 2
    assert settings.collection_name == "fashion_catalog"
    assert settings.llm_provider in ["google_ai_studio", "vertex_ai"]
    assert isinstance(settings.checkpoint_dir, Path)


def test_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Custom Search App")
    monkeypatch.setenv("MIN_CONFIDENCE_SCORE", "0.85")
    monkeypatch.setenv("LLM_PROVIDER", "vertex_ai")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("TOP_K_CANDIDATES", "25")

    settings = Settings()
    assert settings.app_name == "Custom Search App"
    assert settings.min_confidence_score == 0.85
    assert settings.llm_provider == "vertex_ai"
    assert settings.embedding_dim == 768
    assert settings.top_k_candidates == 25
