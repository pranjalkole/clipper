"""
OpenTelemetry distributed tracing setup and in-memory real-time metrics tracking.
Provides distributed tracing across router, embedding, Qdrant, and LLM nodes,
as well as latency percentiles, throughput, token counts, and cost simulation.
"""

import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode

from src.models.schemas import MetricsResponse

logger = logging.getLogger(__name__)

# Module-level singletons
_tracer_provider: TracerProvider | None = None
_telemetry_initialized: bool = False
_init_lock = threading.Lock()


def init_telemetry(
    service_name: str = "clipper-multimodal-search",
    service_version: str = "0.1.0",
) -> TracerProvider:
    """
    Initializes and sets the global OpenTelemetry TracerProvider if not already configured.
    """
    global _tracer_provider, _telemetry_initialized
    with _init_lock:
        if not _telemetry_initialized:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": service_version,
                }
            )
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            _tracer_provider = provider
            _telemetry_initialized = True
            logger.info("OpenTelemetry TracerProvider initialized for service: %s", service_name)
    return trace.get_tracer_provider()  # type: ignore[return-value]


def get_tracer(name: str = "clipper") -> trace.Tracer:
    """Returns an OpenTelemetry Tracer instance."""
    if not _telemetry_initialized:
        init_telemetry()
    return trace.get_tracer(name)


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "clipper",
) -> Generator[trace.Span, None, None]:
    """
    Context manager that starts an OpenTelemetry span, sets attributes,
    and records exceptions automatically.
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(str(k), v if isinstance(v, (bool, str, int, float)) else str(v))
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def instrumented(span_name: str | None = None, tracer_name: str = "clipper") -> Callable:
    """
    Decorator to wrap a function execution in an OpenTelemetry span.
    """
    def decorator(fn: Callable) -> Callable:
        name = span_name or f"{fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(name, tracer_name=tracer_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


class MetricsTracker:
    """
    Thread-safe in-memory performance and cost metrics accumulator.
    Tracks query latencies, token consumption, and computes P50/P90/P95/P99 percentiles
    along with cost-per-query simulation.
    """

    # Gemini 1.5 Flash Pricing (Standard Tier):
    # $0.075 per 1,000,000 input tokens
    # $0.30 per 1,000,000 output tokens
    COST_PER_MILLION_INPUT_TOKENS = 0.075
    COST_PER_MILLION_OUTPUT_TOKENS = 0.30

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._latencies: list[float] = []
        self._total_requests = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Records a completed request's latency and token consumption."""
        with self._lock:
            self._total_requests += 1
            self._latencies.append(latency_ms)
            # Cap stored raw latencies to prevent memory unbounded growth in long runs
            if len(self._latencies) > 10_000:
                self._latencies = self._latencies[-10_000:]
            self._total_input_tokens += max(0, input_tokens)
            self._total_output_tokens += max(0, output_tokens)

    def _percentile(self, sorted_data: list[float], p: float) -> float:
        """Calculates p-th percentile from sorted list."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_data[int(k)], 2)
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return round(d0 + d1, 2)

    def get_metrics_summary(self) -> MetricsResponse:
        """Computes and returns real-time metrics summary."""
        with self._lock:
            now = time.time()
            uptime = max(0.001, now - self._start_time)
            total_req = self._total_requests
            latencies = list(self._latencies)
            in_tokens = self._total_input_tokens
            out_tokens = self._total_output_tokens

        sorted_latencies = sorted(latencies)
        p50 = self._percentile(sorted_latencies, 50.0)
        p90 = self._percentile(sorted_latencies, 90.0)
        p95 = self._percentile(sorted_latencies, 95.0)
        p99 = self._percentile(sorted_latencies, 99.0)
        avg = round(sum(sorted_latencies) / len(sorted_latencies), 2) if sorted_latencies else 0.0

        total_tokens = in_tokens + out_tokens
        tokens_per_sec = round(total_tokens / uptime, 2)

        # Cost calculation
        input_cost = (in_tokens / 1_000_000.0) * self.COST_PER_MILLION_INPUT_TOKENS
        output_cost = (out_tokens / 1_000_000.0) * self.COST_PER_MILLION_OUTPUT_TOKENS
        total_cost = round(input_cost + output_cost, 6)
        cost_per_query = round(total_cost / max(total_req, 1), 6)

        return MetricsResponse(
            total_requests=total_req,
            p50_latency_ms=p50,
            p90_latency_ms=p90,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_latency_ms=avg,
            tokens_per_sec=tokens_per_sec,
            total_tokens_consumed=total_tokens,
            simulated_cost_usd=total_cost,
            cost_per_query_usd=cost_per_query,
            uptime_seconds=round(uptime, 2),
        )

    def reset(self) -> None:
        """Resets metrics counters (useful for isolated unit testing)."""
        with self._lock:
            self._start_time = time.time()
            self._latencies.clear()
            self._total_requests = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0


_global_metrics_tracker = MetricsTracker()


def get_metrics_tracker() -> MetricsTracker:
    """Returns the global MetricsTracker singleton."""
    return _global_metrics_tracker
