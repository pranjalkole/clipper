"""
LangGraph compilation connecting Router, Retrieval, Reflection, and Synthesizer nodes.
Provides compiled StateGraph instance and high-level execution interfaces.
"""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.reflection import create_reflection_node
from src.agents.retrieval_agent import create_retrieval_node
from src.agents.router import create_router_node
from src.agents.state import SearchAgentState, create_initial_state, get_typed_candidates
from src.agents.synthesizer import create_synthesizer_node
from src.core.config import settings
from src.core.llm import get_chat_model
from src.db.qdrant import QdrantCatalogClient, get_qdrant_client
from src.models.embedder import MultimodalEmbedder, get_embedder
from src.models.schemas import SearchFilterParams, SearchResponse
from src.models.wiser import WISERComposedRetriever

logger = logging.getLogger(__name__)


def build_search_graph(
    client: QdrantCatalogClient | None = None,
    embedder: MultimodalEmbedder | None = None,
    retriever: WISERComposedRetriever | None = None,
    llm: BaseChatModel | None = None,
    min_confidence_score: float | None = None,
    max_retries: int | None = None,
) -> CompiledStateGraph:
    """
    Builds and compiles the 4-node LangGraph Multi-Agent Search Graph.

    Workflow:
      START -> router -> retrieval -> reflection
      reflection conditional branch:
        - "retry" -> retrieval
        - "proceed" -> synthesizer -> END
    """
    # Initialize node implementations with injected dependencies
    router_node = create_router_node(llm=llm)
    retrieval_node = create_retrieval_node(client=client, embedder=embedder, retriever=retriever)
    reflection_node = create_reflection_node(
        min_confidence_score=min_confidence_score,
        max_retries=max_retries,
    )
    synthesizer_node = create_synthesizer_node(llm=llm)

    # Define StateGraph
    workflow = StateGraph(SearchAgentState)

    # Register nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define fixed transitions
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retrieval")
    workflow.add_edge("retrieval", "reflection")

    # Define conditional routing from reflection
    def _route_after_reflection(state: SearchAgentState) -> str:
        decision = state.get("route", "proceed")
        logger.info("Routing after reflection: '%s'", decision)
        return decision

    workflow.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {
            "retry": "retrieval",
            "proceed": "synthesizer",
        },
    )

    workflow.add_edge("synthesizer", END)

    compiled_graph = workflow.compile()
    logger.info("LangGraph multi-agent search workflow compiled successfully.")
    return compiled_graph


def run_search_agent(
    query_text: str | None = None,
    query_image: str | None = None,
    modification_text: str | None = None,
    filters: SearchFilterParams | dict[str, Any] | None = None,
    limit: int = 10,
    graph: CompiledStateGraph | None = None,
) -> SearchResponse:
    """
    High-level entry point to execute the multi-agent search graph and return
    a standardized SearchResponse model for API consumption.
    """
    active_graph = graph or get_search_graph()
    initial_state = create_initial_state(
        query_text=query_text,
        query_image=query_image,
        modification_text=modification_text,
        filters=filters,
        limit=limit,
    )

    output_state = active_graph.invoke(initial_state)

    typed_candidates = get_typed_candidates(output_state)
    return SearchResponse(
        query=query_text or modification_text,
        candidates=typed_candidates,
        total_found=len(typed_candidates),
        reasoning=output_state.get("final_response"),
        filters_applied=output_state.get("current_filters"),
        reflection_notes=output_state.get("reflection_notes", []),
    )


# Module-level singleton graph instance
_cached_graph: CompiledStateGraph | None = None


def get_search_graph() -> CompiledStateGraph:
    """
    Convenience provider returning a compiled search graph using default singletons.
    Safe for production API initialization.
    """
    global _cached_graph
    if _cached_graph is None:
        try:
            llm = get_chat_model()
        except Exception:
            llm = None

        _cached_graph = build_search_graph(
            client=get_qdrant_client(),
            embedder=get_embedder(),
            llm=llm,
        )
    return _cached_graph
