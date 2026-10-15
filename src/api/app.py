"""
FastAPI Application Entrypoint & Middleware Configuration.
Configures CORS, OpenTelemetry distributed tracing instrumentation,
lifespan startup/shutdown routines, and custom error handlers.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.api.routes import router
from src.core.config import settings
from src.core.telemetry import init_telemetry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager handling resource initialization
    and clean shutdown.
    """
    # Startup
    logger.info("Starting up %s (env=%s)...", settings.app_name, settings.app_env)
    init_telemetry()
    yield
    # Shutdown
    logger.info("Shutting down %s...", settings.app_name)


def create_app() -> FastAPI:
    """
    Application factory instantiating and configuring the production FastAPI instance.
    """
    application = FastAPI(
        title="Enterprise Multimodal Semantic Search & Agentic Intelligence Platform API",
        description=(
            "High-performance REST API powered by Apple's MobileCLIP2-S0, Qdrant Vector DB, "
            "LangGraph Multi-Agent Orchestration, and Google Gemini."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. OpenTelemetry Auto-Instrumentation
    FastAPIInstrumentor.instrument_app(application)

    # 3. Global Exception Handlers
    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Validation / Value error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "detail": str(exc)},
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal Server Error", "detail": "An unexpected error occurred."},
        )

    # 4. Route Registration (Both root paths and /api/v1 prefix)
    application.include_router(router)
    application.include_router(router, prefix="/api/v1")

    return application


# Global application instance for ASGI servers (Uvicorn / Gunicorn)
app = create_app()
