from unittest.mock import MagicMock, patch

import pytest

from src.core.llm import get_chat_model


def test_get_chat_model_missing_key_raises_value_error():
    """Verifies that an informative ValueError is raised when no API key is set for Google AI Studio."""
    with patch.dict("os.environ", {}, clear=True), patch("src.core.llm.settings.gemini_api_key", None):
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
            get_chat_model(provider="google_ai_studio")


def test_get_chat_model_google_ai_studio_initialization():
    """Verifies ChatGoogleGenerativeAI is initialized with API key for Google AI Studio."""
    with patch("src.core.llm.ChatGoogleGenerativeAI") as mock_chat_cls:
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        model = get_chat_model(
            model_name="gemini-1.5-flash",
            temperature=0.3,
            provider="google_ai_studio",
            api_key="test-api-key-12345",
        )

        mock_chat_cls.assert_called_once_with(
            model="gemini-1.5-flash",
            google_api_key="test-api-key-12345",
            temperature=0.3,
        )
        assert model == mock_instance


def test_get_chat_model_vertex_ai_initialization():
    """Verifies ChatGoogleGenerativeAI is initialized with vertexai=True, project, and location."""
    with patch("src.core.llm.ChatGoogleGenerativeAI") as mock_chat_cls:
        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        with patch("src.core.llm.settings.vertex_project", "my-gcp-project"):
            with patch("src.core.llm.settings.vertex_location", "us-east4"):
                model = get_chat_model(
                    model_name="gemini-1.5-flash",
                    temperature=0.0,
                    provider="vertex_ai",
                )

                mock_chat_cls.assert_called_once_with(
                    model="gemini-1.5-flash",
                    vertexai=True,
                    project="my-gcp-project",
                    location="us-east4",
                    temperature=0.0,
                )
                assert model == mock_instance


def test_get_chat_model_unsupported_provider():
    """Verifies error handling for unsupported LLM provider names."""
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_chat_model(provider="invalid_provider")
