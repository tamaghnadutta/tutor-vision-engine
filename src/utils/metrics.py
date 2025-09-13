"""
Metrics and monitoring utilities
"""

import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Create custom registry to avoid conflicts
registry = CollectorRegistry()

# API Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# Error Detection Metrics
ERROR_DETECTION_COUNT = Counter(
    'error_detection_requests_total',
    'Total error detection requests',
    ['approach', 'has_error'],
    registry=registry
)

ERROR_DETECTION_DURATION = Histogram(
    'error_detection_duration_seconds',
    'Error detection processing duration',
    ['approach'],
    registry=registry
)

MODEL_USAGE_COUNT = Counter(
    'model_usage_total',
    'Model usage count',
    ['provider', 'model'],
    registry=registry
)

CONCURRENT_REQUESTS = Gauge(
    'concurrent_requests',
    'Number of concurrent requests being processed',
    registry=registry
)

# OCR Metrics
OCR_PROCESSING_DURATION = Histogram(
    'ocr_processing_duration_seconds',
    'OCR processing duration',
    ['backend'],
    registry=registry
)

OCR_SUCCESS_RATE = Counter(
    'ocr_success_total',
    'OCR processing success/failure count',
    ['backend', 'status'],
    registry=registry
)


def setup_metrics():
    """Initialize metrics system"""
    pass  # Metrics are initialized on import


class MetricsCollector:
    """Context manager for collecting request metrics"""

    def __init__(self, method: str, endpoint: str):
        self.method = method
        self.endpoint = endpoint
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        CONCURRENT_REQUESTS.inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status = "500" if exc_type else "200"

        REQUEST_COUNT.labels(
            method=self.method,
            endpoint=self.endpoint,
            status=status
        ).inc()

        REQUEST_DURATION.labels(
            method=self.method,
            endpoint=self.endpoint
        ).observe(duration)

        CONCURRENT_REQUESTS.dec()


class ErrorDetectionMetrics:
    """Context manager for error detection metrics"""

    def __init__(self, approach: str):
        self.approach = approach
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        ERROR_DETECTION_DURATION.labels(approach=self.approach).observe(duration)

    def record_result(self, has_error: bool):
        """Record error detection result"""
        ERROR_DETECTION_COUNT.labels(
            approach=self.approach,
            has_error=str(has_error).lower()
        ).inc()

    def record_model_usage(self, provider: str, model: str):
        """Record model usage"""
        MODEL_USAGE_COUNT.labels(provider=provider, model=model).inc()


def record_ocr_metrics(backend: str, duration: float, success: bool):
    """Record OCR processing metrics"""
    OCR_PROCESSING_DURATION.labels(backend=backend).observe(duration)
    OCR_SUCCESS_RATE.labels(
        backend=backend,
        status="success" if success else "failure"
    ).inc()