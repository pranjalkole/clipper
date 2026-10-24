"""
Test suite for Phase 6: Streamlit Interactive Multimodal Web Frontend.
Validates:
  - ApiClient health, metrics, search_text, search_image, and search_hybrid methods
  - Error handling and exception propagation in ApiClient
  - Product URL generation and card helpers
  - Session state initialization
  - Streamlit AppTest smoke execution
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.frontend.app import init_session_state
from src.frontend.client import ApiClient, ApiClientError
from src.frontend.components.product_grid import _get_product_image_url
from src.models.schemas import (
    ComposedSearchResponse,
    HealthResponse,
    MetricsResponse,
    ProductCandidate,
    SearchFilterParams,
    SearchResponse,
    WISERConfig,
)


@pytest.fixture
def mock_candidate() -> ProductCandidate:
    """Fixture providing a sample product candidate."""
    return ProductCandidate(
        id=15970,
        product_display_name="Turtle Check Men Navy Blue Shirt",
        master_category="Apparel",
        sub_category="Topwear",
        brand="Turtle",
        gender="Men",
        price=45.99,
        rating=4.5,
        rating_count=120,
        score=0.88,
        retrieval_path="t2i",
    )


# ============================================================================
# 1. ApiClient Unit Tests
# ============================================================================

def test_api_client_health_check_success():
    """Verifies that check_health returns HealthResponse on 200 OK."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    mock_payload = {
        "status": "healthy",
        "version": "0.1.0",
        "app_name": "Test Platform",
        "environment": "test",
        "qdrant_connected": True,
        "embedder_loaded": True,
        "uptime_seconds": 12.34,
        "timestamp": 123456789.0,
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        health = client.check_health()
        assert health is not None
        assert health.status == "healthy"
        assert health.qdrant_connected is True


def test_api_client_health_check_failure():
    """Verifies that check_health returns None on connection error."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        health = client.check_health()
        assert health is None


def test_api_client_get_metrics_success():
    """Verifies that get_metrics returns MetricsResponse on 200 OK."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    mock_payload = {
        "total_requests": 15,
        "p50_latency_ms": 35.0,
        "p90_latency_ms": 60.0,
        "p95_latency_ms": 80.0,
        "p99_latency_ms": 95.0,
        "mean_latency_ms": 42.0,
        "throughput_req_per_sec": 4.5,
        "total_tokens_consumed": 2400,
        "simulated_cost_usd": 0.0036,
        "cost_per_query_usd": 0.00024,
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        metrics = client.get_metrics()
        assert metrics is not None
        assert metrics.total_requests == 15
        assert metrics.p95_latency_ms == 80.0


def test_api_client_search_text_success(mock_candidate):
    """Verifies search_text posts JSON payload and parses SearchResponse."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    mock_payload = {
        "query": "navy blue shirt",
        "candidates": [mock_candidate.model_dump()],
        "total_found": 1,
        "reasoning": "Selected high-rated navy shirt.",
        "latency_ms": 42.5,
        "reflection_notes": [],
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_post.return_value = mock_resp

        filters = SearchFilterParams(min_price=20.0, max_price=60.0)
        res = client.search_text(query_text="navy blue shirt", filters=filters, limit=5)

        assert isinstance(res, SearchResponse)
        assert res.total_found == 1
        assert len(res.candidates) == 1
        assert res.candidates[0].id == 15970

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["query_text"] == "navy blue shirt"
        assert kwargs["json"]["limit"] == 5
        assert kwargs["json"]["filters"]["min_price"] == 20.0


def test_api_client_search_text_error():
    """Verifies search_text raises ApiClientError on non-200 responses."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        with pytest.raises(ApiClientError) as exc_info:
            client.search_text(query_text="test query")
        assert "status 500" in str(exc_info.value)


def test_api_client_search_image_success(mock_candidate):
    """Verifies search_image posts multipart file payload and parses SearchResponse."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    mock_payload = {
        "query": None,
        "candidates": [mock_candidate.model_dump()],
        "total_found": 1,
        "reasoning": "Visual similarity match.",
        "latency_ms": 38.0,
        "reflection_notes": [],
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_post.return_value = mock_resp

        dummy_bytes = b"\xff\xd8\xff\xe0"  # Small dummy JPEG header
        res = client.search_image(image_bytes=dummy_bytes, limit=10)

        assert isinstance(res, SearchResponse)
        assert res.total_found == 1
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "file" in kwargs["files"]


def test_api_client_search_hybrid_success(mock_candidate):
    """Verifies search_hybrid executes Composed Image Retrieval."""
    client = ApiClient(base_url="http://mocked-gateway:8000")
    mock_payload = {
        "modification_text": "same in red",
        "target_caption": "A red shirt with check pattern",
        "candidates": [mock_candidate.model_dump()],
        "total_found": 1,
        "confidence_score": 0.85,
        "verified": True,
        "iterations_count": 1,
        "latency_ms": 65.0,
        "refinement_notes": [],
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_post.return_value = mock_resp

        res = client.search_hybrid(
            image_bytes=b"fake-bytes",
            modification_text="same in red",
            limit=5,
            wiser_config=WISERConfig(confidence_threshold=0.75),
        )

        assert isinstance(res, ComposedSearchResponse)
        assert res.target_caption == "A red shirt with check pattern"
        assert res.verified is True


# ============================================================================
# 2. UI Helper Unit Tests
# ============================================================================

def test_product_image_url_canonical():
    """Verifies that _get_product_image_url formats valid catalog asset endpoint URLs."""
    url = _get_product_image_url(15970)
    assert url == "http://127.0.0.1:8000/catalog/images/15970"

    url_custom = _get_product_image_url(15970, base_url="http://custom-api:9000")
    assert url_custom == "http://custom-api:9000/catalog/images/15970"


def test_session_state_init():
    """Verifies session state default attributes initialization."""
    with patch("streamlit.session_state", {}) as mock_state:
        init_session_state()
        assert "search_response" in mock_state
        assert "cir_ref_bytes" in mock_state
        assert "pending_filter_updates" in mock_state
        assert mock_state["cir_mod_prompt"] == ""


def test_product_card_similar_quick_action_filter_propagation():
    """Verifies that render_product_card stages category lock and color filter in pending_filter_updates."""
    from src.frontend.components.product_grid import render_product_card
    from src.models.schemas import ProductCandidate

    prod = ProductCandidate(
        id=3150,
        product_display_name="Nike Men's Lunarswift Black Shoe",
        master_category="Footwear",
        sub_category="Shoes",
        article_type="Sports Shoes",
        base_colour="Black",
        gender="Men",
        season="Fall",
        year=2010,
        usage="Sports",
        brand="Nike",
        price=55.05,
        rating=3.7,
        rating_count=1352,
        in_stock=True,
        tags=["black", "shoes"],
        score=0.97,
    )

    mock_state = {}
    with patch("streamlit.session_state", mock_state), \
         patch("src.frontend.components.product_grid._fetch_image_bytes", return_value=b"fake-image"), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.rerun"):

        render_product_card(prod, index=0)

        pending = mock_state.get("pending_filter_updates", {})
        assert pending.get("filter_categories") == ["Footwear"]
        assert pending.get("filter_sub_categories") == ["Shoes"]
        assert pending.get("filter_colors") == ["Black"]
        assert "active_query_to_run" in mock_state
        assert mock_state["active_query_to_run"][0] == "image"


# ============================================================================
# 3. Streamlit App Smoke & Execution Test
# ============================================================================

def test_streamlit_app_loads():
    """Smoke test ensuring app.py executes cleanly without runtime syntax or import errors."""
    app_path = Path(__file__).parent.parent / "src" / "frontend" / "app.py"
    app_test = AppTest.from_file(str(app_path), default_timeout=15.0)
    app_test.run()
    assert not app_test.exception
