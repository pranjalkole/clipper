"""Models package for vision-language embedders and data schemas."""

from src.models.embedder import MultimodalEmbedder, get_embedder, reset_embedder

__all__ = ["MultimodalEmbedder", "get_embedder", "reset_embedder"]
