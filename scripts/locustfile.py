#!/usr/bin/env python3
"""
Locust load testing script for Error Detection API

Usage:
    # Basic load test
    locust -f scripts/locustfile.py --host=http://localhost:8000

    # With specific user count and spawn rate
    locust -f scripts/locustfile.py --host=http://localhost:8000 -u 10 -r 2

    # Headless mode with specific duration
    locust -f scripts/locustfile.py --host=http://localhost:8000 -u 10 -r 2 --run-time 60s --headless

    # Test specific approach
    ERROR_DETECTION_APPROACH=ocr_llm locust -f scripts/locustfile.py --host=http://localhost:8000

Environment Variables:
    ERROR_DETECTION_APPROACH: ocr_llm, vlm_direct, or hybrid (default: hybrid)
    API_KEY: API key for authentication (default: test-api-key-123)
"""

import os
import random
import time
import logging
from typing import Dict, List, Any
from locust import HttpUser, task, between, events
from locust.env import Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY", "test-api-key-123")
ERROR_DETECTION_APPROACH = os.getenv("ERROR_DETECTION_APPROACH", "hybrid")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data - multiple sample combinations for variety
TEST_SAMPLES = [
    {
        "name": "Q1_Attempt1",
        "question_url": "http://localhost:8080/data/sample_images/questions/Q1.jpeg",
        "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
        "bounding_box": {"minX": 0.1, "maxX": 0.9, "minY": 0.1, "maxY": 0.9}
    },
    {
        "name": "Q2_Attempt2",
        "question_url": "http://localhost:8080/data/sample_images/questions/Q2.jpeg",
        "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt2.jpeg",
        "bounding_box": {"minX": 0.15, "maxX": 0.85, "minY": 0.15, "maxY": 0.85}
    },
    {
        "name": "Q3_Attempt3",
        "question_url": "http://localhost:8080/data/sample_images/questions/Q3.jpeg",
        "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt3.jpeg",
        "bounding_box": {"minX": 0.2, "maxX": 0.8, "minY": 0.2, "maxY": 0.8}
    },
    {
        "name": "Q4_Attempt4",
        "question_url": "http://localhost:8080/data/sample_images/questions/Q4.jpeg",
        "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt4.jpeg",
        "bounding_box": {"minX": 0.05, "maxX": 0.95, "minY": 0.05, "maxY": 0.95}
    },
]

# Custom metrics tracking
class MetricsCollector:
    def __init__(self):
        self.successful_detections = 0
        self.errors_found = 0
        self.confidence_scores = []
        self.approach_counts = {}

    def record_detection(self, has_error: bool, confidence: float, approach: str):
        self.successful_detections += 1
        if has_error:
            self.errors_found += 1
        self.confidence_scores.append(confidence)
        self.approach_counts[approach] = self.approach_counts.get(approach, 0) + 1

    def get_stats(self):
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        error_rate = (self.errors_found / self.successful_detections * 100) if self.successful_detections > 0 else 0

        return {
            "successful_detections": self.successful_detections,
            "errors_found": self.errors_found,
            "error_detection_rate": error_rate,
            "avg_confidence": avg_confidence,
            "approach_counts": self.approach_counts
        }

# Global metrics collector
metrics_collector = MetricsCollector()

class ErrorDetectionUser(HttpUser):
    """
    Simulates a user making error detection requests
    """

    # Wait time between requests (simulates real user behavior)
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts"""
        self.approach = ERROR_DETECTION_APPROACH
        self.session_id = f"locust_session_{int(time.time())}_{random.randint(1000, 9999)}"

        # Perform health check
        self.health_check()

        logger.info(f"User started - approach: {self.approach}, session: {self.session_id}")

    def health_check(self):
        """Check API health before starting tests"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                logger.info("API health check passed")
            else:
                logger.error(f"API health check failed: {response.status_code}")
                response.failure(f"Health check failed: {response.status_code}")

    @task(10)
    def detect_error_random_sample(self):
        """Main task: Detect errors using random sample data"""
        sample = random.choice(TEST_SAMPLES)
        self._make_error_detection_request(sample, "random_sample")

    @task(3)
    def detect_error_q1_attempt1(self):
        """Focused task: Always test Q1-Attempt1 combination"""
        self._make_error_detection_request(TEST_SAMPLES[0], "q1_attempt1")

    @task(2)
    def detect_error_with_small_bbox(self):
        """Task: Test with smaller bounding box"""
        sample = random.choice(TEST_SAMPLES)
        small_bbox_sample = sample.copy()
        small_bbox_sample["bounding_box"] = {"minX": 0.3, "maxX": 0.7, "minY": 0.3, "maxY": 0.7}
        self._make_error_detection_request(small_bbox_sample, "small_bbox")

    @task(1)
    def detect_error_full_image(self):
        """Task: Test with full image (no cropping)"""
        sample = random.choice(TEST_SAMPLES)
        full_image_sample = sample.copy()
        full_image_sample["bounding_box"] = {"minX": 0.0, "maxX": 1.0, "minY": 0.0, "maxY": 1.0}
        self._make_error_detection_request(full_image_sample, "full_image")

    def _make_error_detection_request(self, sample: Dict[str, Any], task_name: str):
        """Make an error detection API request"""

        # Prepare request data
        request_data = {
            "question_url": sample["question_url"],
            "solution_url": sample["solution_url"],
            "bounding_box": sample["bounding_box"],
            "user_id": f"locust_user_{self.session_id}",
            "session_id": self.session_id,
            "question_id": f"locust_{sample['name']}_{int(time.time())}"
        }

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-Error-Detection-Approach": self.approach
        }

        # Make the request
        start_time = time.time()

        with self.client.post(
            "/api/v1/detect-error",
            json=request_data,
            headers=headers,
            catch_response=True,
            name=f"detect_error_{task_name}_{self.approach}"
        ) as response:

            duration = time.time() - start_time

            if response.status_code == 200:
                try:
                    # Parse response
                    data = response.json()

                    # Handle new nested structure
                    error_analysis = data.get('error_analysis', {})
                    solution_analysis = data.get('solution_analysis', {})

                    # Extract information (new structure first, fallback to old)
                    has_error = error_analysis.get('has_error', bool(data.get('error')))
                    confidence = error_analysis.get('confidence', data.get('confidence', 0.0))
                    solution_complete = solution_analysis.get('solution_complete', data.get('solution_complete', False))
                    job_id = data.get('job_id', 'unknown')

                    # Record metrics
                    metrics_collector.record_detection(has_error, confidence, self.approach)

                    # Log successful request
                    logger.debug(
                        f"Success: {task_name} | Duration: {duration:.2f}s | "
                        f"Error: {has_error} | Confidence: {confidence:.3f} | "
                        f"Job: {job_id} | Approach: {self.approach}"
                    )

                    # Mark as success
                    response.success()

                except Exception as e:
                    logger.error(f"Failed to parse response: {e}")
                    response.failure(f"Response parsing failed: {e}")

            elif response.status_code == 401:
                response.failure("Authentication failed - check API key")

            elif response.status_code == 422:
                response.failure("Invalid request data")

            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")

            else:
                response.failure(f"Unexpected status code: {response.status_code}")

class OCRLLMUser(ErrorDetectionUser):
    """User specifically testing OCR→LLM approach"""

    def on_start(self):
        self.approach = "ocr_llm"
        super().on_start()

class DirectVLMUser(ErrorDetectionUser):
    """User specifically testing Direct VLM approach"""

    def on_start(self):
        self.approach = "vlm_direct"
        super().on_start()

class HybridUser(ErrorDetectionUser):
    """User specifically testing Hybrid approach"""

    def on_start(self):
        self.approach = "hybrid"
        super().on_start()

# Event listeners for better reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    logger.info("=" * 60)
    logger.info("LOCUST LOAD TEST STARTING")
    logger.info(f"Target URL: {environment.host}")
    logger.info(f"Approach: {ERROR_DETECTION_APPROACH}")
    logger.info(f"API Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else API_KEY)
    logger.info("=" * 60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    stats = metrics_collector.get_stats()

    logger.info("=" * 60)
    logger.info("LOCUST LOAD TEST COMPLETED")
    logger.info("=" * 60)
    logger.info("Error Detection Metrics:")
    logger.info(f"  Successful detections: {stats['successful_detections']}")
    logger.info(f"  Errors found: {stats['errors_found']}")
    logger.info(f"  Error detection rate: {stats['error_detection_rate']:.1f}%")
    logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
    logger.info("Approach breakdown:")
    for approach, count in stats['approach_counts'].items():
        logger.info(f"  {approach}: {count} requests")
    logger.info("=" * 60)

# Custom Locust events for monitoring integration
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response=None, context=None, exception=None, **kwargs):
    """Called for each request"""
    if "detect_error" in name:
        if exception:
            logger.warning(f"Request failed: {name} - {exception}")
        # Could send custom metrics to monitoring systems here
        pass

# Configuration for different test scenarios
if __name__ == "__main__":
    # This allows running the script directly for quick testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)

    # Quick test mode
    print("Running quick Locust test...")
    print("For full testing, use: locust -f scripts/locustfile.py --host=http://localhost:8000")