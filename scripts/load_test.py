#!/usr/bin/env python3
"""
Load testing script for error detection API
Usage: python scripts/load_test.py
"""

import asyncio
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import aiohttp
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Result from load testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    duration: float
    requests_per_second: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    error_breakdown: Dict[str, int]


class LoadTester:
    """Load tester for the API"""

    def __init__(self, base_url: str, api_key: str, approach: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.approach = approach or os.getenv("ERROR_DETECTION_APPROACH", "hybrid")
        self.results: List[Dict[str, Any]] = []

    async def run_load_test(self,
                          total_requests: int = 50,
                          concurrent_requests: int = 5,
                          duration_seconds: int = 60) -> LoadTestResult:
        """Run load test with specified parameters"""

        # Health check first
        if not await self._health_check():
            logger.error("Health check failed, aborting load test")
            return LoadTestResult(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"Health check failed": 1})

        logger.info(f"Starting load test: {total_requests} requests, {concurrent_requests} concurrent, approach: {self.approach}")

        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrent_requests)

        # Create sample request using actual test images
        sample_request = {
            "question_url": "http://localhost:8080/data/sample_images/questions/Q1.jpeg",
            "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
            "bounding_box": {
                "minX": 0.1,
                "maxX": 0.9,
                "minY": 0.1,
                "maxY": 0.9
            },
            "user_id": "load_test_user",
            "session_id": f"load_test_session_{int(time.time())}",
            "question_id": "load_test_question"
        }

        # Run requests with time limit
        tasks = []
        for i in range(total_requests):
            task = asyncio.create_task(
                self._make_request(semaphore, sample_request, i)
            )
            tasks.append(task)

        # Wait for completion or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=duration_seconds
            )
        except asyncio.TimeoutError:
            logger.warning("Load test timed out")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

        end_time = time.time()
        duration = end_time - start_time

        # Analyze results
        return self._analyze_results(duration)

    async def _health_check(self) -> bool:
        """Check if the API is healthy before starting load test"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        logger.info(f"API health check passed: {health_data}")
                        return True
                    else:
                        logger.error(f"API health check failed with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False

    async def _make_request(self, semaphore: asyncio.Semaphore,
                          request_data: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Make a single API request"""
        async with semaphore:
            start_time = time.time()

            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Error-Detection-Approach": self.approach
                    }

                    async with session.post(
                        f"{self.base_url}/api/v1/detect-error",
                        json=request_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:

                        end_time = time.time()
                        latency = end_time - start_time

                        result = {
                            'request_id': request_id,
                            'status_code': response.status,
                            'success': 200 <= response.status < 300,
                            'latency': latency,
                            'timestamp': start_time
                        }

                        if result['success']:
                            try:
                                response_data = await response.json()
                                result['job_id'] = response_data.get('job_id')

                                # Handle new nested structure
                                error_analysis = response_data.get('error_analysis', {})
                                solution_analysis = response_data.get('solution_analysis', {})

                                # Extract error information (new structure first, fallback to old)
                                has_error = error_analysis.get('has_error', bool(response_data.get('error')))
                                confidence = error_analysis.get('confidence', response_data.get('confidence', 0.0))
                                solution_complete = solution_analysis.get('solution_complete', response_data.get('solution_complete', False))

                                result['has_error'] = has_error
                                result['confidence'] = confidence
                                result['solution_complete'] = solution_complete
                                result['approach'] = self.approach
                            except Exception as e:
                                logger.warning(f"Failed to parse response data: {e}")
                                pass
                        else:
                            result['error'] = f"HTTP {response.status}"
                            try:
                                error_data = await response.text()
                                result['error_detail'] = error_data[:200]
                            except:
                                pass

                        self.results.append(result)
                        return result

            except asyncio.TimeoutError:
                end_time = time.time()
                result = {
                    'request_id': request_id,
                    'status_code': 408,
                    'success': False,
                    'latency': end_time - start_time,
                    'timestamp': start_time,
                    'error': 'Request timeout'
                }
                self.results.append(result)
                return result

            except Exception as e:
                end_time = time.time()
                result = {
                    'request_id': request_id,
                    'status_code': 0,
                    'success': False,
                    'latency': end_time - start_time,
                    'timestamp': start_time,
                    'error': str(e)[:100]
                }
                self.results.append(result)
                return result

    def _analyze_results(self, duration: float) -> LoadTestResult:
        """Analyze load test results"""
        if not self.results:
            return LoadTestResult(0, 0, 0, 0.0, duration, 0.0, 0.0, 0.0, 0.0, {})

        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['success'])
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
        requests_per_second = total_requests / duration if duration > 0 else 0.0

        # Latency analysis
        successful_latencies = [r['latency'] for r in self.results if r['success']]
        if successful_latencies:
            successful_latencies.sort()
            latency_p50 = self._percentile(successful_latencies, 50)
            latency_p90 = self._percentile(successful_latencies, 90)
            latency_p95 = self._percentile(successful_latencies, 95)
        else:
            latency_p50 = latency_p90 = latency_p95 = 0.0

        # Error breakdown
        error_breakdown = {}
        for result in self.results:
            if not result['success']:
                error = result.get('error', 'Unknown error')
                error_breakdown[error] = error_breakdown.get(error, 0) + 1

        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            duration=duration,
            requests_per_second=requests_per_second,
            latency_p50=latency_p50,
            latency_p90=latency_p90,
            latency_p95=latency_p95,
            error_breakdown=error_breakdown
        )

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of sorted data"""
        if not data:
            return 0.0
        index = int((percentile / 100) * len(data))
        return data[min(index, len(data) - 1)]

    def save_results(self, output_path: str):
        """Save detailed results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {output_path}")


def print_results(result: LoadTestResult, approach: str = None):
    """Print formatted load test results"""
    print("\n" + "="*60)
    print("LOAD TEST RESULTS")
    if approach:
        print(f"Approach: {approach.upper().replace('_', '→')}")
    print("="*60)
    print(f"Total Requests:       {result.total_requests}")
    print(f"Successful:           {result.successful_requests}")
    print(f"Failed:               {result.failed_requests}")
    print(f"Success Rate:         {result.success_rate:.1%}")
    print(f"Duration:             {result.duration:.1f}s")
    print(f"Requests/Second:      {result.requests_per_second:.1f}")
    print(f"")
    print(f"Latency (successful requests only):")
    print(f"  p50:                {result.latency_p50:.2f}s")
    print(f"  p90:                {result.latency_p90:.2f}s")
    print(f"  p95:                {result.latency_p95:.2f}s")

    if result.error_breakdown:
        print(f"\nError Breakdown:")
        for error, count in result.error_breakdown.items():
            print(f"  {error}: {count}")

    print("="*60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run load test on error detection API")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Base URL of the API")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "test-api-key-123"),
                       help="API key for authentication")
    parser.add_argument("--requests", type=int, default=25,
                       help="Total number of requests")
    parser.add_argument("--concurrent", type=int, default=5,
                       help="Number of concurrent requests")
    parser.add_argument("--duration", type=int, default=60,
                       help="Maximum duration in seconds")
    parser.add_argument("--approach",
                       choices=["ocr_llm", "vlm_direct", "hybrid"],
                       default=os.getenv("ERROR_DETECTION_APPROACH", "hybrid"),
                       help="Error detection approach to test")
    parser.add_argument("--output", help="Output file for detailed results")

    args = parser.parse_args()

    # Run load test
    tester = LoadTester(args.url, args.api_key, args.approach)
    result = await tester.run_load_test(
        total_requests=args.requests,
        concurrent_requests=args.concurrent,
        duration_seconds=args.duration
    )

    # Display results
    print_results(result, args.approach)

    # Save detailed results
    if args.output:
        tester.save_results(args.output)

    # Exit with appropriate code
    if result.success_rate < 0.9:  # Less than 90% success rate
        print("\nWARNING: Success rate below 90%")
        sys.exit(1)
    elif result.latency_p95 > 10.0:  # p95 latency above 10s
        print("\nWARNING: p95 latency above 10 seconds")
        sys.exit(1)
    else:
        print("\nLoad test PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())