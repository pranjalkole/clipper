"""
FastAPI route definitions for the Multimodal Semantic Search & Agentic Intelligence Platform.
Endpoints:
  - GET  /health: System health and dependency status
  - GET  /metrics: Real-time latency percentiles, tokens/sec, and simulated cost
  - POST /search/text: Text-to-image semantic search via LangGraph multi-agent RAG
  - POST /search/image: Visual similarity search (multipart upload or JSON base64)
  - POST /search/hybrid: Composed image retrieval (WISER CIR: image + modification text)
  - POST /ingest: Trigger catalog ingestion (sync or background execution)
"""

import base64
import logging
import time

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from src.agents.graph import run_search_agent
from src.core.config import settings
from src.core.telemetry import get_metrics_tracker, trace_span
from src.db.qdrant import get_qdrant_client
from src.models.embedder import get_embedder
from src.models.schemas import (
    ComposedSearchRequest,
    ComposedSearchResponse,
    HealthResponse,
    ImageSearchRequest,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    SearchFilterParams,
    SearchResponse,
    TextSearchRequest,
)
from src.models.wiser import WISERComposedRetriever
from src.pipeline.ingestion import ingest_catalog

logger = logging.getLogger(__name__)

router = APIRouter()
_server_start_time = time.time()


# ============================================================================
# 1. System Health & Observability Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and dependency status",
)
async def health_check() -> HealthResponse:
    """Verifies service readiness, Qdrant connectivity, and embedder availability."""
    with trace_span("health_check"):
        now = time.time()
        uptime = round(now - _server_start_time, 2)

        qdrant_ok = False
        try:
            client = get_qdrant_client()
            stats = client.get_stats()
            qdrant_ok = stats is not None and "points_count" in stats
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant health check failed: %s", exc)

        embedder_ok = False
        try:
            embedder = get_embedder()
            embedder_ok = embedder is not None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedder health check failed: %s", exc)

        overall_status = "healthy" if (qdrant_ok and embedder_ok) else "degraded"

        return HealthResponse(
            status=overall_status,
            version="0.1.0",
            app_name=settings.app_name,
            environment=settings.app_env,
            qdrant_connected=qdrant_ok,
            embedder_loaded=embedder_ok,
            uptime_seconds=uptime,
            timestamp=round(now, 3),
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Observability"],
    summary="Real-time latency percentiles, throughput, and simulated costs",
)
async def get_metrics() -> MetricsResponse:
    """Returns real-time P50, P90, P95, P99 latencies, throughput, and simulated Gemini costs."""
    with trace_span("get_metrics"):
        tracker = get_metrics_tracker()
        return tracker.get_metrics_summary()


# ============================================================================
# 2. Semantic & Multimodal Search Endpoints
# ============================================================================

@router.post(
    "/search/text",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Text-to-image semantic search via LangGraph agentic workflow",
)
async def search_text(request: TextSearchRequest) -> SearchResponse:
    """
    Executes intelligent agentic search from a textual query string.
    Employs router constraint extraction, MobileCLIP2-S0 text embedding,
    Qdrant hybrid retrieval, and self-reflective critique.
    """
    t0 = time.perf_counter()
    with trace_span("search.text", attributes={"query": request.query_text, "limit": request.limit}):
        try:
            response = run_search_agent(
                query_text=request.query_text,
                filters=request.filters,
                limit=request.limit,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            response.latency_ms = elapsed_ms

            # Estimate token metrics for reporting
            input_tokens = len(request.query_text.split()) * 4
            output_tokens = len(response.reasoning.split()) * 4 if response.reasoning else 30
            get_metrics_tracker().record_request(
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return response
        except Exception as exc:
            logger.exception("Text search execution failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Semantic search failed: {exc}",
            ) from exc


@router.post(
    "/search/image",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Visual similarity search (accepts multipart file upload or JSON Base64)",
)
async def search_image(
    request: Request,
) -> SearchResponse:
    """
    Executes visual image-to-image similarity search.
    Supports either:
      - multipart/form-data: raw image file ('file' or 'image') + optional form fields.
      - application/json: ImageSearchRequest payload with base64 string or path.
    """
    t0 = time.perf_counter()
    content_type = request.headers.get("content-type", "")

    image_spec: str | None = None
    filters: SearchFilterParams | None = None
    limit: int = 10

    with trace_span("search.image"):
        if "multipart/form-data" in content_type:
            form = await request.form()
            upload: UploadFile | None = form.get("file") or form.get("image")  # type: ignore[assignment]
            if not upload:
                raise HTTPException(
                    status_code=422,
                    detail="Multipart request must contain 'file' or 'image' field.",
                )
            file_bytes = await upload.read()
            if not file_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded image file is empty.",
                )
            b64_encoded = base64.b64encode(file_bytes).decode("utf-8")
            image_spec = f"data:{upload.content_type or 'image/jpeg'};base64,{b64_encoded}"

            # Parse optional form fields
            limit_val = form.get("limit")
            if limit_val is not None:
                try:
                    limit = int(limit_val)
                except ValueError:
                    pass

            filters_raw = form.get("filters")
            if filters_raw and isinstance(filters_raw, str):
                try:
                    filters = SearchFilterParams.model_validate_json(filters_raw)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to parse filters json: %s", exc)
        else:
            try:
                body = await request.json()
                req = ImageSearchRequest.model_validate(body)
                image_spec = req.query_image_base64 or req.query_image_path
                filters = req.filters
                limit = req.limit
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid JSON payload for image search: {exc}",
                ) from exc

        if not image_spec:
            raise HTTPException(
                status_code=422,
                detail="No image provided (query_image_base64, query_image_path, or multipart file required).",
            )

        try:
            response = run_search_agent(
                query_image=image_spec,
                filters=filters,
                limit=limit,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            response.latency_ms = elapsed_ms

            get_metrics_tracker().record_request(
                latency_ms=elapsed_ms,
                input_tokens=258,  # Approximate CLIP vision token representation
                output_tokens=30,
            )
            return response
        except Exception as exc:
            logger.exception("Visual search execution failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Visual search failed: {exc}",
            ) from exc


@router.post(
    "/search/hybrid",
    response_model=ComposedSearchResponse,
    tags=["Search"],
    summary="Composed image retrieval (WISER Zero-Shot CIR: image + modifying prompt)",
)
async def search_hybrid(
    request: Request,
) -> ComposedSearchResponse:
    """
    Executes Zero-Shot Composed Image Retrieval (CIR) via the WISER architecture:
    Accepts a reference image and a natural language modification instruction
    (e.g., 'same shoe in navy blue with flat laces').
    Supports both multipart/form-data and application/json requests.
    """
    t0 = time.perf_counter()
    content_type = request.headers.get("content-type", "")

    req_obj: ComposedSearchRequest

    with trace_span("search.hybrid"):
        if "multipart/form-data" in content_type:
            form = await request.form()
            mod_text = form.get("modification_text") or form.get("query_text")
            if not mod_text or not isinstance(mod_text, str):
                raise HTTPException(
                    status_code=422,
                    detail="Field 'modification_text' is required for composed hybrid search.",
                )

            upload: UploadFile | None = form.get("file") or form.get("image")  # type: ignore[assignment]
            b64_spec = None
            if upload is not None:
                content = await upload.read()
                if content:
                    b64_encoded = base64.b64encode(content).decode("utf-8")
                    b64_spec = f"data:{upload.content_type or 'image/jpeg'};base64,{b64_encoded}"

            limit_val = form.get("limit")
            limit = 10
            if limit_val is not None:
                try:
                    limit = int(limit_val)
                except ValueError:
                    pass

            filters_raw = form.get("filters")
            filters: SearchFilterParams | None = None
            if filters_raw and isinstance(filters_raw, str):
                try:
                    filters = SearchFilterParams.model_validate_json(filters_raw)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to parse filters json: %s", exc)

            req_obj = ComposedSearchRequest(
                reference_image_base64=b64_spec,
                modification_text=mod_text,
                filters=filters,
                limit=limit,
            )
        else:
            try:
                body = await request.json()
                req_obj = ComposedSearchRequest.model_validate(body)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid payload for composed search: {exc}",
                ) from exc

        try:
            retriever = WISERComposedRetriever()
            response = retriever.retrieve(req_obj)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            response.latency_ms = elapsed_ms

            get_metrics_tracker().record_request(
                latency_ms=elapsed_ms,
                input_tokens=320,
                output_tokens=60,
            )
            return response
        except Exception as exc:
            logger.exception("Hybrid CIR search failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hybrid search failed: {exc}",
            ) from exc


# ============================================================================
# 3. Catalog Ingestion Pipeline Endpoint
# ============================================================================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Trigger catalog ingestion and embedding upsert into Qdrant",
)
async def trigger_ingestion(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Indexes catalog items from Hugging Face dataset into Qdrant vector database.
    Can run synchronously (default) or asynchronously in background tasks.
    """
    with trace_span("ingest", attributes={"limit": request.limit, "recreate": request.recreate}):
        if request.background:
            logger.info("Scheduling catalog ingestion in background task (limit=%s)...", request.limit)
            background_tasks.add_task(
                ingest_catalog,
                limit=request.limit,
                batch_size=request.batch_size,
                recreate=request.recreate,
                streaming=request.streaming,
                dataset_name=request.dataset_name,
            )
            return IngestResponse(
                status="accepted",
                items_indexed=0,
                elapsed_seconds=0.0,
                throughput_items_per_sec=0.0,
                collection_name=settings.collection_name,
                total_points=0,
                message=f"Ingestion started in background for up to {request.limit or 'all'} items.",
            )

        try:
            logger.info("Executing catalog ingestion synchronously (limit=%s)...", request.limit)
            stats = ingest_catalog(
                limit=request.limit,
                batch_size=request.batch_size,
                recreate=request.recreate,
                streaming=request.streaming,
                dataset_name=request.dataset_name,
            )
            return IngestResponse(
                status="success",
                items_indexed=stats.get("items_indexed", 0),
                elapsed_seconds=stats.get("elapsed_seconds", 0.0),
                throughput_items_per_sec=stats.get("throughput_items_per_sec", 0.0),
                collection_name=stats.get("collection_name", settings.collection_name),
                total_points=stats.get("collection_stats", {}).get("points_count", 0),
                message="Ingestion completed successfully.",
            )
        except Exception as exc:
            logger.exception("Ingestion failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Catalog ingestion failed: {exc}",
            ) from exc
