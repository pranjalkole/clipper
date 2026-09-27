"""Models package for vision-language embedders, schemas, and CIR engines."""

from src.models.embedder import MultimodalEmbedder, get_embedder, reset_embedder
from src.models.schemas import (
    AgentStateSchema,
    CatalogComposedSearchInput,
    CatalogHybridSearchInput,
    ComposedSearchRequest,
    ComposedSearchResponse,
    ProductCandidate,
    ProductDetailsLookupInput,
    SearchFilterParams,
    SearchRequest,
    SearchResponse,
    WISERConfig,
    WISERRefinement,
    WISERVerificationResult,
)
from src.models.wiser import (
    WISERComposedRetriever,
    WISERComposer,
    WISEREditor,
    WISERRefiner,
    WISERVerifier,
)

__all__ = [
    "AgentStateSchema",
    "CatalogComposedSearchInput",
    "CatalogHybridSearchInput",
    "ComposedSearchRequest",
    "ComposedSearchResponse",
    "MultimodalEmbedder",
    "ProductCandidate",
    "ProductDetailsLookupInput",
    "SearchFilterParams",
    "SearchRequest",
    "SearchResponse",
    "WISERComposedRetriever",
    "WISERComposer",
    "WISERConfig",
    "WISEREditor",
    "WISERRefiner",
    "WISERRefinement",
    "WISERVerificationResult",
    "WISERVerifier",
    "get_embedder",
    "reset_embedder",
]
