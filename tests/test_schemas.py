import pytest
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


def test_product_candidate_creation_and_attribution():
    """Validates ProductCandidate serialization, field defaults, and score attribution."""
    candidate = ProductCandidate(
        id=12345,
        product_display_name="Nike Running Sneakers",
        master_category="Footwear",
        brand="Nike",
        price=110.0,
        rating=4.7,
        score=0.88,
        wiser_t2i_score=0.90,
        wiser_comp_score=0.85,
        wiser_mod_score=0.82,
        retrieval_path="wiser_fused",
    )

    data = candidate.model_dump()
    assert data["id"] == 12345
    assert data["product_display_name"] == "Nike Running Sneakers"
    assert data["score"] == 0.88
    assert data["wiser_t2i_score"] == 0.90
    assert data["wiser_comp_score"] == 0.85
    assert data["retrieval_path"] == "wiser_fused"


def test_product_candidate_from_payload_and_score():
    """Validates factory method instantiating ProductCandidate from Qdrant payload."""
    payload = {
        "id": 999,
        "product_display_name": "Levi's Denim Jacket",
        "master_category": "Apparel",
        "sub_category": "Topwear",
        "gender": "Men",
        "brand": "Levi's",
        "price": 89.99,
        "rating": 4.4,
        "in_stock": True,
        "tags": ["jacket", "denim", "blue"],
    }
    candidate = ProductCandidate.from_payload_and_score(
        payload=payload,
        score=0.825,
        rank=1,
        retrieval_path="t2i",
        wiser_t2i_score=0.85,
        wiser_comp_score=0.80,
    )

    assert candidate.id == 999
    assert candidate.brand == "Levi's"
    assert candidate.score == 0.825
    assert candidate.rank == 1
    assert candidate.retrieval_path == "t2i"
    assert candidate.wiser_t2i_score == 0.85
    assert candidate.wiser_comp_score == 0.80


def test_wiser_config_defaults_and_validation():
    """Validates WISERConfig parameter bounds and default settings."""
    cfg = WISERConfig()
    assert cfg.alpha == 1.0
    assert cfg.w_t2i == 0.45
    assert cfg.w_comp == 0.35
    assert cfg.w_mod == 0.20
    assert cfg.confidence_threshold == 0.70
    assert cfg.pool_factor == 2
    assert cfg.max_refinement_retries == 2

    # Custom valid config
    custom = WISERConfig(alpha=1.2, confidence_threshold=0.75)
    assert custom.alpha == 1.2
    assert custom.confidence_threshold == 0.75


def test_search_filter_params():
    """Validates SearchFilterParams filtering constraints."""
    filters = SearchFilterParams(
        master_category="Apparel",
        gender="Women",
        min_price=20.0,
        max_price=100.0,
        in_stock_only=True,
    )
    dumped = filters.model_dump()
    assert dumped["master_category"] == "Apparel"
    assert dumped["gender"] == "Women"
    assert dumped["min_price"] == 20.0
    assert dumped["max_price"] == 100.0
    assert dumped["in_stock_only"] is True


def test_composed_search_request_and_response():
    """Validates ComposedSearchRequest and ComposedSearchResponse models."""
    req = ComposedSearchRequest(
        modification_text="same polo shirt in navy blue",
        limit=5,
    )
    assert req.modification_text == "same polo shirt in navy blue"
    assert req.limit == 5
    assert req.wiser_config.confidence_threshold == 0.70

    resp = ComposedSearchResponse(
        modification_text="same polo shirt in navy blue",
        target_caption="Navy blue men's short-sleeve polo shirt",
        candidates=[],
        total_found=0,
        confidence_score=0.78,
        verified=True,
        iterations_count=1,
    )
    assert resp.verified is True
    assert resp.confidence_score == 0.78


def test_agent_state_schema():
    """Validates AgentStateSchema for multi-agent workflows."""
    state = AgentStateSchema(query_text="red dress under $50")
    assert state.query_text == "red dress under $50"
    assert state.is_composed_query is False
    assert state.verified is False
    assert state.retry_count == 0
    assert state.max_retries == 2
