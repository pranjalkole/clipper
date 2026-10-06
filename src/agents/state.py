"""
State definitions and schemas for LangGraph multi-agent search workflow.
"""

from typing import Annotated, Any, TypedDict
import operator

from src.core.config import settings
from src.models.schemas import ProductCandidate, SearchFilterParams


def _append_or_extend(existing: list[str] | None, new_items: list[str] | str) -> list[str]:
    """Custom reducer supporting string or list appending for reflection notes."""
    result = list(existing or [])
    if isinstance(new_items, list):
        result.extend(new_items)
    elif isinstance(new_items, str):
        result.append(new_items)
    return result


class SearchAgentState(TypedDict, total=False):
    """
    Mutable state definition for the LangGraph multi-agent search workflow.
    Tracks queries, extracted filter constraints, candidate pools, critique scores,
    and self-reflection loop iterations.
    """

    # User Input
    query_text: str | None
    query_image: str | None
    modification_text: str | None
    is_composed_query: bool
    limit: int

    # Intent & Constraint Extraction (Router)
    extracted_keywords: str | None
    extracted_filters: dict[str, Any]
    current_filters: dict[str, Any]
    relaxed_filters: list[str]

    # Candidate Retrieval
    candidates: list[dict[str, Any]]
    total_found: int

    # Critique & Self-Reflection
    confidence_score: float
    reflection_score: float
    reflection_notes: Annotated[list[str], _append_or_extend]
    retry_count: int
    max_retries: int
    route: str  # "proceed" | "retry"

    # Multimodal Synthesis
    final_response: str | None


def create_initial_state(
    query_text: str | None = None,
    query_image: str | None = None,
    modification_text: str | None = None,
    filters: SearchFilterParams | dict[str, Any] | None = None,
    limit: int = 10,
    max_retries: int | None = None,
) -> SearchAgentState:
    """
    Initializes a clean SearchAgentState dictionary with sensible defaults.
    """
    filter_dict: dict[str, Any] = {}
    if isinstance(filters, SearchFilterParams):
        filter_dict = {k: v for k, v in filters.model_dump().items() if v is not None}
    elif isinstance(filters, dict):
        filter_dict = dict(filters)

    is_cir = bool(modification_text or (query_image and query_text and "same" in query_text.lower()))

    return {
        "query_text": query_text,
        "query_image": query_image,
        "modification_text": modification_text,
        "is_composed_query": is_cir,
        "limit": limit,
        "extracted_keywords": query_text,
        "extracted_filters": filter_dict.copy(),
        "current_filters": filter_dict.copy(),
        "relaxed_filters": [],
        "candidates": [],
        "total_found": 0,
        "confidence_score": 0.0,
        "reflection_score": 0.0,
        "reflection_notes": [],
        "retry_count": 0,
        "max_retries": max_retries if max_retries is not None else settings.max_reflection_retries,
        "route": "proceed",
        "final_response": None,
    }


def get_typed_candidates(state: SearchAgentState) -> list[ProductCandidate]:
    """
    Converts dictionary candidates stored in state into typed ProductCandidate models.
    """
    raw_candidates = state.get("candidates", [])
    results: list[ProductCandidate] = []
    for item in raw_candidates:
        if isinstance(item, ProductCandidate):
            results.append(item)
        elif isinstance(item, dict):
            try:
                results.append(ProductCandidate.model_validate(item))
            except Exception:
                pass
    return results
