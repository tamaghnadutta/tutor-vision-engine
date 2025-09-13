"""
Error Detection API - Main FastAPI application
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.api.routes import error_detection
from src.api.middleware import LoggingMiddleware, SecurityMiddleware
from src.config.settings import get_settings
from src.utils.logging import setup_logging
from src.utils.metrics import setup_metrics


# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    setup_metrics()
    logger.info("Error Detection API starting up")

    yield

    # Shutdown
    logger.info("Error Detection API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title="Error Detection API",
        description="AI-powered educational platform API for error detection in handwritten math solutions",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(SecurityMiddleware)

    # Routes
    app.include_router(error_detection.router, prefix="/api/v1")

    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "error-detection-api"}

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type="text/plain")

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception at {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "An unexpected error occurred"}
        )

    return app


app = create_app()


def main():
    """Main entry point for running the API"""
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_config=None,  # We handle logging ourselves
    )


if __name__ == "__main__":
    main()