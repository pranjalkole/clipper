import argparse
import logging
import os
import sys
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings

logger = logging.getLogger(__name__)


def get_chat_model(
    model_name: str | None = None,
    temperature: float = 0.0,
    provider: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Factory function for Google Gemini ChatModel instances.
    Supports dual-mode deployment:
      - 'google_ai_studio': Zero-cost rapid prototyping using GEMINI_API_KEY.
      - 'vertex_ai': Enterprise deployment on Google Cloud Vertex AI via GCP project & location.

    Args:
        model_name: Gemini model name (defaults to settings.gemini_model, e.g. 'gemini-1.5-flash').
        temperature: Sampling temperature for generation.
        provider: 'google_ai_studio' or 'vertex_ai' (defaults to settings.llm_provider).
        api_key: Optional explicit API key override.
        **kwargs: Extra parameters passed to ChatGoogleGenerativeAI.

    Returns:
        Configured BaseChatModel instance.
    """
    selected_provider = (provider or settings.llm_provider).lower().strip()
    target_model = model_name or settings.gemini_model

    if selected_provider == "google_ai_studio":
        resolved_key = (
            api_key
            or settings.gemini_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured for Google AI Studio provider. "
                "Please acquire a free key at https://aistudio.google.com/ and set GEMINI_API_KEY "
                "in your .env file or environment."
            )

        logger.info("Initializing ChatGoogleGenerativeAI (Google AI Studio, model=%s)", target_model)
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=resolved_key,
            temperature=temperature,
            **kwargs,
        )

    if selected_provider == "vertex_ai":
        project = settings.vertex_project or os.getenv("VERTEX_PROJECT")
        location = settings.vertex_location or os.getenv("VERTEX_LOCATION", "us-central1")
        logger.info(
            "Initializing ChatGoogleGenerativeAI (Vertex AI, model=%s, project=%s, location=%s)",
            target_model,
            project,
            location,
        )
        return ChatGoogleGenerativeAI(
            model=target_model,
            vertexai=True,
            project=project,
            location=location,
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{selected_provider}'. Supported options: 'google_ai_studio', 'vertex_ai'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Gemini 1.5 Flash connectivity via Google AI Studio or Vertex AI."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the architectural benefit of MobileCLIP2 in 1 concise sentence.",
        help="Test prompt to verify model response.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=settings.llm_provider,
        choices=["google_ai_studio", "vertex_ai"],
        help="LLM provider backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.gemini_model,
        help=f"Model identifier (default: {settings.gemini_model}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override for Google AI Studio.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Google Gemini LLM Connectivity Verification")
    print("=" * 60)
    print(f"Provider : {args.provider}")
    print(f"Model    : {args.model}")
    print(f"Prompt   : '{args.prompt}'")
    print("-" * 60)

    try:
        chat_model = get_chat_model(
            model_name=args.model,
            provider=args.provider,
            api_key=args.api_key,
            temperature=0.2,
        )
    except ValueError as exc:
        print(f"\n[CONFIGURATION NOTICE]: {exc}")
        print("\nTo test live connectivity, obtain a free tier API key from:")
        print("  -> https://aistudio.google.com/")
        print("Then run with:")
        print("  GEMINI_API_KEY='your-key' python -m src.core.llm")
        sys.exit(0)

    try:
        t0 = time.perf_counter()
        response = chat_model.invoke(args.prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"[STATUS] : SUCCESS (Response latency: {elapsed_ms:.1f}ms)")
        print(f"[RESPONSE]:\n{response.content}")
        print("=" * 60)
    except Exception as exc:
        print(f"\n[ERROR]: Failed to invoke Gemini LLM: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
