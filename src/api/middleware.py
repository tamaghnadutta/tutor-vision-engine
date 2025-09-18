"""
FastAPI middleware components
"""

import time
import uuid
from typing import Callable

import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"[ID: {request_id}, IP: {request.client.host if request.client else 'unknown'}, "
            f"Query: {request.query_params}]"
        )

        # Process request
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Request completed: {response.status_code} [ID: {request_id}, Duration: {duration:.3f}s]"
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for basic security headers and rate limiting"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add security headers
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff" # Stop MIME-type tricks; Prevents attacks where malicious scripts are disguised as harmless files
        response.headers["X-Frame-Options"] = "DENY" # Stop clickjacking; prevents attackers trying to trick users into clicking hidden buttons by overlaying this site inside their own page
        response.headers["X-XSS-Protection"] = "1; mode=block" # Help older browsers block reflected XSS
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains" # Force HTTPS always

        return response


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit concurrent requests and track metrics"""

    def __init__(self, app, max_concurrent_requests: int = 10):
        super().__init__(app)
        self.max_concurrent_requests = max_concurrent_requests
        self.current_requests = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        from src.utils.metrics import CONCURRENT_REQUESTS

        # Check concurrency limit
        if self.current_requests >= self.max_concurrent_requests:
            logger.warning(
                f"Request rejected due to concurrency limit: {self.current_requests}/{self.max_concurrent_requests}"
            )
            return Response(
                content='{"error": "Server busy", "detail": "Too many concurrent requests"}',
                status_code=503,
                media_type="application/json"
            )

        # Process request
        self.current_requests += 1
        CONCURRENT_REQUESTS.set(self.current_requests)  # Update Prometheus metric

        try:
            response = await call_next(request)
            return response
        finally:
            self.current_requests -= 1
            CONCURRENT_REQUESTS.set(self.current_requests)  # Update Prometheus metric