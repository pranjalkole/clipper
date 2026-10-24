"""
HTTP client for communicating with the FastAPI Gateway backend.
"""

import base64
import logging
import os
from typing import Any
import httpx

from src.models.schemas import (
    ComposedSearchResponse,
    HealthResponse,
    MetricsResponse,
    SearchFilterParams,
    SearchResponse,
    WISERConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.getenv("CLIPPER_API_URL", "http://127.0.0.1:8000")


class ApiClientError(Exception):
    """Raised when an API call fails or the backend service is unreachable."""

    def __init__(self, message: str, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class ApiClient:
    """Client for querying the Clipper FastAPI Gateway."""

    def __init__(self, base_url: str | None = None, timeout: float = 45.0):
        self.base_url = (base_url or DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> HealthResponse | None:
        """Checks API health and service availability."""
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return HealthResponse.model_validate(resp.json())
                return None
        except Exception as exc:
            logger.debug("Health check failed for %s: %s", url, exc)
            return None

    def get_metrics(self) -> MetricsResponse | None:
        """Fetches real-time performance and latency telemetry."""
        url = f"{self.base_url}/metrics"
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return MetricsResponse.model_validate(resp.json())
                return None
        except Exception as exc:
            logger.debug("Metrics fetch failed: %s", exc)
            return None

    def search_text(
        self,
        query_text: str,
        filters: SearchFilterParams | None = None,
        limit: int = 10,
    ) -> SearchResponse:
        """Executes text-to-image semantic search via LangGraph multi-agent."""
        url = f"{self.base_url}/search/text"
        payload: dict[str, Any] = {
            "query_text": query_text,
            "limit": limit,
        }
        if filters:
            payload["filters"] = filters.model_dump(exclude_none=True)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    raise ApiClientError(
                        f"Text search failed with status {resp.status_code}",
                        status_code=resp.status_code,
                        detail=resp.text,
                    )
                return SearchResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            raise ApiClientError(f"HTTP connection error: {exc}") from exc

    def search_image(
        self,
        image_bytes: bytes,
        filename: str = "query.jpg",
        content_type: str = "image/jpeg",
        filters: SearchFilterParams | None = None,
        limit: int = 10,
    ) -> SearchResponse:
        """Executes visual similarity search using multipart upload."""
        url = f"{self.base_url}/search/image"
        files = {"file": (filename, image_bytes, content_type)}
        data: dict[str, Any] = {"limit": str(limit)}
        if filters:
            data["filters"] = filters.model_dump_json(exclude_none=True)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, files=files, data=data)
                if resp.status_code != 200:
                    raise ApiClientError(
                        f"Visual search failed with status {resp.status_code}",
                        status_code=resp.status_code,
                        detail=resp.text,
                    )
                return SearchResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            raise ApiClientError(f"HTTP connection error: {exc}") from exc

    def search_hybrid(
        self,
        image_bytes: bytes | None,
        modification_text: str,
        filename: str = "reference.jpg",
        content_type: str = "image/jpeg",
        filters: SearchFilterParams | None = None,
        limit: int = 10,
        wiser_config: WISERConfig | None = None,
    ) -> ComposedSearchResponse:
        """Executes Composed Image Retrieval (WISER CIR: image + modification prompt)."""
        url = f"{self.base_url}/search/hybrid"

        if image_bytes:
            files = {"file": (filename, image_bytes, content_type)}
            data: dict[str, Any] = {
                "modification_text": modification_text,
                "limit": str(limit),
            }
            if filters:
                data["filters"] = filters.model_dump_json(exclude_none=True)
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, files=files, data=data)
                    if resp.status_code != 200:
                        raise ApiClientError(
                            f"Hybrid search failed with status {resp.status_code}",
                            status_code=resp.status_code,
                            detail=resp.text,
                        )
                    return ComposedSearchResponse.model_validate(resp.json())
            except httpx.HTTPError as exc:
                raise ApiClientError(f"HTTP connection error: {exc}") from exc
        else:
            # Fallback to JSON payload without file if image was base64 or omitted
            payload: dict[str, Any] = {
                "modification_text": modification_text,
                "limit": limit,
            }
            if filters:
                payload["filters"] = filters.model_dump(exclude_none=True)
            if wiser_config:
                payload["wiser_config"] = wiser_config.model_dump()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code != 200:
                        raise ApiClientError(
                            f"Hybrid search failed with status {resp.status_code}",
                            status_code=resp.status_code,
                            detail=resp.text,
                        )
                    return ComposedSearchResponse.model_validate(resp.json())
            except httpx.HTTPError as exc:
                raise ApiClientError(f"HTTP connection error: {exc}") from exc
