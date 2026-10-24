"""
Test suite for FastAPI REST API Gateway & Observability (Phase 5).
Verifies:
  - GET /health: Health status and dependency checks
  - GET /metrics: Real-time latency percentiles, throughput, token and cost simulation
  - POST /search/text: Text-to-image semantic search
  - POST /search/image: Visual similarity search via multipart and base64 JSON
  - POST /search/hybrid: WISER Composed Image Retrieval (CIR)
  - POST /ingest: Catalog ingestion (sync & background tasks)
  - Global error handlers and CORS headers
"""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.core.telemetry import get_metrics_tracker
from src.models.schemas import (
    ComposedSearchResponse,
    ProductCandidate,
    SearchResponse,
)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_jpeg_bytes() -> bytes:
    """Generates a small in-memory 10x10 RGB JPEG image."""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_base64_image(dummy_jpeg_bytes) -> str:
    """Returns a base64 data URI string for the dummy image."""
    b64 = base64.b64encode(dummy_jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@pytest.fixture(autouse=True)
def reset_metrics():
    """Resets the metrics tracker before each test for clean isolation."""
    tracker = get_metrics_tracker()
    tracker.reset()
    yield
    tracker.reset()


# ============================================================================
# 1. Health & Metrics Endpoint Tests
# ============================================================================

def test_health_check_endpoint(client):
    """Verifies that /health returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["version"] == "0.1.0"
    assert "uptime_seconds" in data
    assert "qdrant_connected" in data
    assert "embedder_loaded" in data


def test_health_check_api_v1_prefix(client):
    """Verifies that /api/v1/health alias works identically."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("healthy", "degraded")


def test_metrics_endpoint(client):
    """Verifies that /metrics returns valid performance statistics."""
    tracker = get_metrics_tracker()
    tracker.record_request(latency_ms=45.0, input_tokens=100, output_tokens=50)
    tracker.record_request(latency_ms=85.0, input_tokens=200, output_tokens=80)

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 2
    assert data["p50_latency_ms"] > 0
    assert data["p95_latency_ms"] >= data["p50_latency_ms"]
    assert data["total_tokens_consumed"] == 430
    assert data["simulated_cost_usd"] > 0
    assert data["cost_per_query_usd"] > 0


@patch("src.api.routes.get_image_store")
def test_get_catalog_image_success(mock_get_store, client):
    """Verifies GET /catalog/images/{id} returns JPEG bytes on success."""
    mock_store = MagicMock()
    mock_store.get_image_bytes.return_value = b"\xff\xd8\xff\xe0mock-jpeg-bytes"
    mock_get_store.return_value = mock_store

    response = client.get("/catalog/images/15970")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8\xff\xe0mock-jpeg-bytes"


@patch("src.api.routes.get_image_store")
def test_get_catalog_image_not_found(mock_get_store, client):
    """Verifies GET /catalog/images/{id} returns 404 when image is missing."""
    mock_store = MagicMock()
    mock_store.get_image_bytes.return_value = None
    mock_get_store.return_value = mock_store

    response = client.get("/catalog/images/9999999")
    assert response.status_code == 404



# ============================================================================
# 2. Text Search Endpoint Tests
# ============================================================================

@patch("src.api.routes.run_search_agent")
def test_search_text_success(mock_run_search, client):
    """Verifies POST /search/text executes agentic search and returns valid schema."""
    mock_run_search.return_value = SearchResponse(
        query="running shoes",
        candidates=[
            ProductCandidate(
                id=101,
                product_display_name="Nike Air Zoom Pegasus",
                score=0.88,
                brand="Nike",
                price=120.0,
            )
        ],
        total_found=1,
        reasoning="Grounded recommendation for running shoes",
        latency_ms=42.0,
    )

    payload = {
        "query_text": "running shoes",
        "filters": {"brand": "Nike", "max_price": 150.0},
        "limit": 5,
    }
    response = client.post("/search/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "running shoes"
    assert data["total_found"] == 1
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["product_display_name"] == "Nike Air Zoom Pegasus"
    assert data["candidates"][0]["score"] == 0.88

    # Assert request metrics recorded
    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 1


def test_search_text_validation_error(client):
    """Verifies 422 is returned when query_text is missing."""
    response = client.post("/search/text", json={"limit": 5})
    assert response.status_code == 422


# ============================================================================
# 3. Visual Search Endpoint Tests
# ============================================================================

@patch("src.api.routes.run_search_agent")
def test_search_image_multipart_upload(mock_run_search, client, dummy_jpeg_bytes):
    """Verifies POST /search/image accepts multipart image file upload."""
    mock_run_search.return_value = SearchResponse(
        query="[visual query]",
        candidates=[
            ProductCandidate(
                id=202,
                product_display_name="Blue Denim Jacket",
                score=0.91,
                master_category="Apparel",
            )
        ],
        total_found=1,
    )

    files = {"file": ("jacket.jpg", dummy_jpeg_bytes, "image/jpeg")}
    data = {"limit": "4"}
    response = client.post("/search/image", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["total_found"] == 1
    assert result["candidates"][0]["id"] == 202


@patch("src.api.routes.run_search_agent")
def test_search_image_json_base64(mock_run_search, client, dummy_base64_image):
    """Verifies POST /search/image accepts application/json with base64 image."""
    mock_run_search.return_value = SearchResponse(
        query="[visual query]",
        candidates=[
            ProductCandidate(
                id=303,
                product_display_name="Puma Suede Classic",
                score=0.85,
            )
        ],
        total_found=1,
    )

    payload = {
        "query_image_base64": dummy_base64_image,
        "limit": 3,
    }
    response = client.post("/search/image", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["total_found"] == 1
    assert result["candidates"][0]["id"] == 303


def test_search_image_missing_image(client):
    """Verifies 422 when neither file nor base64 image is supplied."""
    response = client.post("/search/image", json={"limit": 5})
    assert response.status_code == 422


# ============================================================================
# 4. Hybrid Composed Search (WISER CIR) Endpoint Tests
# ============================================================================

@patch("src.api.routes.WISERComposedRetriever")
def test_search_hybrid_json(mock_wiser_cls, client, dummy_base64_image):
    """Verifies POST /search/hybrid executes composed search from JSON request."""
    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = ComposedSearchResponse(
        modification_text="same in navy blue with collar",
        target_caption="Navy blue casual shirt with polo collar",
        candidates=[
            ProductCandidate(
                id=404,
                product_display_name="Navy Blue Polo T-shirt",
                score=0.89,
                wiser_t2i_score=0.90,
                wiser_comp_score=0.87,
                wiser_mod_score=0.88,
            )
        ],
        total_found=1,
        confidence_score=0.89,
        verified=True,
    )
    mock_wiser_cls.return_value = mock_instance

    payload = {
        "reference_image_base64": dummy_base64_image,
        "modification_text": "same in navy blue with collar",
        "limit": 5,
    }
    response = client.post("/search/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["modification_text"] == "same in navy blue with collar"
    assert data["target_caption"] == "Navy blue casual shirt with polo collar"
    assert data["verified"] is True
    assert len(data["candidates"]) == 1


@patch("src.api.routes.WISERComposedRetriever")
def test_search_hybrid_multipart(mock_wiser_cls, client, dummy_jpeg_bytes):
    """Verifies POST /search/hybrid accepts multipart upload with modification text."""
    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = ComposedSearchResponse(
        modification_text="leather strap",
        target_caption="Analog watch with brown leather strap",
        candidates=[],
        total_found=0,
    )
    mock_wiser_cls.return_value = mock_instance

    files = {"file": ("watch.jpg", dummy_jpeg_bytes, "image/jpeg")}
    data = {"modification_text": "leather strap", "limit": "2"}
    response = client.post("/search/hybrid", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["modification_text"] == "leather strap"


# ============================================================================
# 5. Catalog Ingestion Endpoint Tests
# ============================================================================

@patch("src.api.routes.ingest_catalog")
def test_ingest_synchronous(mock_ingest, client):
    """Verifies POST /ingest synchronously triggers ingestion and returns summary."""
    mock_ingest.return_value = {
        "items_indexed": 50,
        "elapsed_seconds": 1.25,
        "throughput_items_per_sec": 40.0,
        "collection_name": "fashion_catalog",
        "collection_stats": {"points_count": 50},
    }

    payload = {"limit": 50, "batch_size": 16, "recreate": False, "background": False}
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["items_indexed"] == 50
    assert data["throughput_items_per_sec"] == 40.0


@patch("src.api.routes.ingest_catalog")
def test_ingest_background_task(mock_ingest, client):
    """Verifies POST /ingest schedules background indexing when background=True."""
    payload = {"limit": 20, "background": True}
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "background" in data["message"].lower()


# ============================================================================
# 6. Error Handlers & Middleware Tests
# ============================================================================

def test_cors_headers_present(client):
    """Verifies CORS headers are properly configured on responses."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
