"""
Retrieval agent node executing hybrid or composed semantic search against Qdrant.
"""

import logging
from typing import Any, Callable

from src.agents.state import SearchAgentState
from src.agents.tools import create_catalog_tools
from src.db.qdrant import QdrantCatalogClient
from src.models.embedder import MultimodalEmbedder
from src.models.wiser import WISERComposedRetriever

logger = logging.getLogger(__name__)


def create_retrieval_node(
    client: QdrantCatalogClient | None = None,
    embedder: MultimodalEmbedder | None = None,
    retriever: WISERComposedRetriever | None = None,
) -> Callable[[SearchAgentState], dict[str, Any]]:
    """
    Factory creating retrieval node with injected database, embedder, and tools.
    """
    tools = create_catalog_tools(client=client, embedder=embedder, retriever=retriever)
    tools_by_name = {t.name: t for t in tools}
    hybrid_tool = tools_by_name["catalog_hybrid_search"]
    composed_tool = tools_by_name["catalog_composed_search"]

    def retrieval_node(state: SearchAgentState) -> dict[str, Any]:
        is_cir = state.get("is_composed_query", False)
        query_text = state.get("extracted_keywords") or state.get("query_text")
        query_image = state.get("query_image")
        modification_text = state.get("modification_text") or query_text
        current_filters = state.get("current_filters") or {}
        limit = state.get("limit", 10)

        candidates: list[dict[str, Any]] = []

        try:
            if is_cir and (query_image or modification_text):
                logger.info(
                    "Executing WISER composed search: mod='%s', filters=%s",
                    modification_text,
                    current_filters,
                )
                tool_args = {
                    "modification_text": modification_text or "similar product",
                    "reference_image": query_image,
                    "limit": limit,
                    **current_filters,
                }
                # Filter out None and keys not supported by tool
                clean_args = {k: v for k, v in tool_args.items() if v is not None}
                resp = composed_tool.invoke(clean_args)
                if isinstance(resp, dict):
                    candidates = resp.get("candidates", [])
                elif isinstance(resp, list):
                    candidates = resp
            else:
                logger.info(
                    "Executing catalog hybrid search: text='%s', filters=%s",
                    query_text,
                    current_filters,
                )
                tool_args = {
                    "query_text": query_text,
                    "query_image": query_image,
                    "limit": limit,
                    **current_filters,
                }
                clean_args = {k: v for k, v in tool_args.items() if v is not None}
                candidates = hybrid_tool.invoke(clean_args)

        except Exception as exc:
            logger.error("Error executing catalog retrieval: %s", exc, exc_info=True)
            candidates = []

        logger.info("Retrieval returned %d candidates", len(candidates))
        return {
            "candidates": candidates,
            "total_found": len(candidates),
        }

    return retrieval_node
