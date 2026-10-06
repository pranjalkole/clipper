"""
Test suite for LangGraph multi-agent orchestration, nodes, and self-reflection loops.
"""

from unittest.mock import MagicMock
import pytest
from qdrant_client import QdrantClient

from src.agents.graph import build_search_graph, run_search_agent
from src.agents.reflection import create_reflection_node, _compute_reflection_metrics
from src.agents.retrieval_agent import create_retrieval_node
from src.agents.router import create_router_node, _extract_heuristics
from src.agents.state import create_initial_state, get_typed_candidates
from src.agents.synthesizer import create_synthesizer_node
from src.db.qdrant import QdrantCatalogClient
from src.models.schemas import ProductCandidate, SearchFilterParams, SearchResponse
from src.models.wiser import WISERComposedRetriever, WISERConfig


@pytest.fixture
def test_catalog():
    """Sets up an in-memory Qdrant client populated with sample catalog products."""
    qc = QdrantClient(":memory:")
    client = QdrantCatalogClient(client=qc, collection_name="test_agent_catalog", embedding_dim=4)
    client.init_collection(recreate=True)

    items = [
        {
            "id": 101,
            "product_display_name": "Nike Pegasus 40 Running Shoes",
            "master_category": "Footwear",
            "sub_category": "Shoes",
            "gender": "Men",
            "brand": "Nike",
            "price": 130.0,
            "rating": 4.7,
            "in_stock": True,
            "tags": ["running", "shoes", "nike"],
        },
        {
            "id": 102,
            "product_display_name": "Puma Smash V2 Sneakers",
            "master_category": "Footwear",
            "sub_category": "Shoes",
            "gender": "Men",
            "brand": "Puma",
            "price": 55.0,
            "rating": 4.2,
            "in_stock": True,
            "tags": ["sneakers", "casual", "puma"],
        },
        {
            "id": 103,
            "product_display_name": "Levi's Slim Fit Denim Jeans",
            "master_category": "Apparel",
            "sub_category": "Bottomwear",
            "gender": "Men",
            "brand": "Levi's",
            "price": 70.0,
            "rating": 4.5,
            "in_stock": True,
            "tags": ["jeans", "denim", "pants"],
        },
    ]

    # Pre-calculated unit vectors
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],  # item 101
        [0.0, 1.0, 0.0, 0.0],  # item 102
        [0.0, 0.0, 1.0, 0.0],  # item 103
    ]
    client.upsert_catalog_items(items, embeddings)

    mock_embedder = MagicMock()
    # By default, match item 101
    mock_embedder.embed_text.side_effect = lambda t: [1.0, 0.0, 0.0, 0.0]
    mock_embedder.embed_image.side_effect = lambda img: [1.0, 0.0, 0.0, 0.0]

    retriever = WISERComposedRetriever(
        embedder=mock_embedder,
        client=client,
        llm=None,
        config=WISERConfig(confidence_threshold=0.50),
    )

    return {
        "client": client,
        "embedder": mock_embedder,
        "retriever": retriever,
    }


# =====================================================================
# 1. State Schema & Helper Tests
# =====================================================================

def test_initial_state_defaults():
    """Verifies that create_initial_state creates properly initialized state."""
    state = create_initial_state(query_text="blue running shoes", limit=5)
    assert state["query_text"] == "blue running shoes"
    assert state["limit"] == 5
    assert state["candidates"] == []
    assert state["retry_count"] == 0
    assert state["route"] == "proceed"
    assert state["is_composed_query"] is False


def test_get_typed_candidates():
    """Verifies conversion of dictionary candidates to ProductCandidate models."""
    state = {
        "candidates": [
            {
                "id": 1,
                "product_display_name": "Test Shirt",
                "price": 25.0,
                "score": 0.85,
            }
        ]
    }
    typed = get_typed_candidates(state)
    assert len(typed) == 1
    assert isinstance(typed[0], ProductCandidate)
    assert typed[0].id == 1
    assert typed[0].score == 0.85


# =====================================================================
# 2. Router Node Tests
# =====================================================================

def test_router_heuristic_extraction():
    """Verifies regex/heuristic filter extraction from natural language queries."""
    query = "Find me Nike running shoes for men under $80 in stock"
    filters, keywords = _extract_heuristics(query)

    assert filters.get("brand") == "Nike"
    assert filters.get("gender") == "Men"
    assert filters.get("max_price") == 80.0
    assert filters.get("in_stock_only") is True
    assert filters.get("master_category") == "Footwear"
    assert "running shoes" in keywords


def test_router_node_execution():
    """Verifies router node merges extracted filters with pre-existing filters."""
    router = create_router_node(llm=None)
    state = create_initial_state(
        query_text="Puma shoes under $60",
        filters={"min_rating": 4.0},
    )

    result = router(state)
    assert result["current_filters"].get("brand") == "Puma"
    assert result["current_filters"].get("max_price") == 60.0
    assert result["current_filters"].get("min_rating") == 4.0
    assert result["is_composed_query"] is False


def test_router_composed_query_detection():
    """Verifies router detects composed image query intent."""
    router = create_router_node(llm=None)
    state = create_initial_state(
        query_text="same sneakers in blue colour",
        query_image="data:image/jpeg;base64,mock...",
    )
    result = router(state)
    assert result["is_composed_query"] is True


# =====================================================================
# 3. Retrieval Node Tests
# =====================================================================

def test_retrieval_node_hybrid_search(test_catalog):
    """Verifies retrieval node retrieves matching candidates from catalog."""
    retriever_node = create_retrieval_node(
        client=test_catalog["client"],
        embedder=test_catalog["embedder"],
        retriever=test_catalog["retriever"],
    )

    state = {
        "extracted_keywords": "Nike sneakers",
        "current_filters": {"brand": "Nike"},
        "limit": 5,
        "is_composed_query": False,
    }

    result = retriever_node(state)
    assert "candidates" in result
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["brand"] == "Nike"
    assert result["candidates"][0]["score"] > 0.90


# =====================================================================
# 4. Reflection Node Tests
# =====================================================================

def test_reflection_high_score_proceeds():
    """Verifies reflection node approves candidates exceeding confidence threshold."""
    reflection = create_reflection_node(min_confidence_score=0.70, max_retries=2)
    state = {
        "candidates": [{"id": 1, "score": 0.95, "price": 50.0, "brand": "Nike"}],
        "current_filters": {"brand": "Nike", "max_price": 100.0},
        "retry_count": 0,
    }

    result = reflection(state)
    assert result["route"] == "proceed"
    assert result["confidence_score"] >= 0.70
    assert len(result["reflection_notes"]) == 1


def test_reflection_low_score_triggers_retry_and_relaxes_price():
    """Verifies reflection node triggers retry and expands max_price on low similarity."""
    reflection = create_reflection_node(min_confidence_score=0.70, max_retries=2)
    state = {
        "candidates": [{"id": 1, "score": 0.40, "price": 50.0}],
        "current_filters": {"max_price": 50.0},
        "relaxed_filters": [],
        "retry_count": 0,
        "extracted_keywords": "expensive luxury shoes",
    }

    result = reflection(state)
    assert result["route"] == "retry"
    assert result["retry_count"] == 1
    assert result["current_filters"]["max_price"] == 75.0  # 50.0 * 1.5
    assert len(result["relaxed_filters"]) == 1
    assert "max_price expanded" in result["relaxed_filters"][0]


def test_reflection_max_retries_safeguard():
    """Verifies reflection node halts retrying once max_retries is reached."""
    reflection = create_reflection_node(min_confidence_score=0.70, max_retries=2)
    state = {
        "candidates": [{"id": 1, "score": 0.35}],
        "current_filters": {},
        "retry_count": 2,  # Already at max
    }

    result = reflection(state)
    assert result["route"] == "proceed"
    assert "Max reflection retries" in result["reflection_notes"][0]


# =====================================================================
# 5. Synthesizer Node Tests
# =====================================================================

def test_synthesizer_deterministic_fallback():
    """Verifies synthesizer fallback produces clean markdown with candidate details."""
    synthesizer = create_synthesizer_node(llm=None)
    state = {
        "query_text": "running shoes",
        "candidates": [
            {
                "id": 101,
                "product_display_name": "Nike Pegasus 40 Running Shoes",
                "brand": "Nike",
                "price": 130.0,
                "rating": 4.7,
                "score": 0.92,
                "in_stock": True,
            }
        ],
        "relaxed_filters": ["max_price expanded to $150"],
        "confidence_score": 0.92,
    }

    result = synthesizer(state)
    assert "final_response" in result
    text = result["final_response"]
    assert "Nike Pegasus 40 Running Shoes" in text
    assert "$130.00" in text
    assert "max_price expanded to $150" in text


# =====================================================================
# 6. End-to-End StateGraph Integration Tests
# =====================================================================

def test_graph_direct_proceed_path(test_catalog):
    """
    Verifies end-to-end graph execution when initial retrieval score is high:
    START -> router -> retrieval -> reflection (proceed) -> synthesizer -> END
    """
    graph = build_search_graph(
        client=test_catalog["client"],
        embedder=test_catalog["embedder"],
        retriever=test_catalog["retriever"],
        llm=None,
        min_confidence_score=0.70,
    )

    initial_state = create_initial_state(
        query_text="Nike shoes",
        limit=5,
    )

    final_state = graph.invoke(initial_state)

    assert final_state["route"] == "proceed"
    assert final_state["retry_count"] == 0
    assert len(final_state["candidates"]) > 0
    assert final_state["candidates"][0]["brand"] == "Nike"
    assert final_state["final_response"] is not None
    assert "Nike Pegasus" in final_state["final_response"]


def test_graph_reflection_loop_relaxation(test_catalog):
    """
    Verifies self-reflection loop:
    Initial restrictive filter (price <= $60) misses the $130 Nike shoe.
    Reflection detects low score/empty match, relaxes price to $90, then $135,
    allowing candidate to match and successfully synthesize.
    """
    graph = build_search_graph(
        client=test_catalog["client"],
        embedder=test_catalog["embedder"],
        retriever=test_catalog["retriever"],
        llm=None,
        min_confidence_score=0.70,
        max_retries=2,
    )

    # Price <= 50 cannot match Nike ($130), so initial retrieval returns empty
    initial_state = create_initial_state(
        query_text="Nike shoes under $50",
        limit=5,
    )

    final_state = graph.invoke(initial_state)

    # Verify reflection loop was entered
    assert final_state["retry_count"] > 0
    assert len(final_state["relaxed_filters"]) > 0
    assert final_state["final_response"] is not None


def test_graph_unmatchable_query_terminates_safely(test_catalog):
    """
    Verifies that an unmatchable query exits cleanly after max_retries without hanging.
    """
    graph = build_search_graph(
        client=test_catalog["client"],
        embedder=test_catalog["embedder"],
        retriever=test_catalog["retriever"],
        llm=None,
        min_confidence_score=0.99,  # Impossibly high threshold
        max_retries=2,
    )

    initial_state = create_initial_state(
        query_text="NonexistentBrand luxury spacesuit",
        limit=5,
    )

    final_state = graph.invoke(initial_state)
    assert final_state["retry_count"] == 2
    assert final_state["route"] == "proceed"
    assert final_state["final_response"] is not None


def test_run_search_agent_helper(test_catalog):
    """
    Verifies run_search_agent returns a standardized SearchResponse.
    """
    graph = build_search_graph(
        client=test_catalog["client"],
        embedder=test_catalog["embedder"],
        retriever=test_catalog["retriever"],
        llm=None,
    )

    response = run_search_agent(
        query_text="Nike running shoes",
        limit=3,
        graph=graph,
    )

    assert isinstance(response, SearchResponse)
    assert response.total_found > 0
    assert response.candidates[0].brand == "Nike"
    assert response.reasoning is not None
