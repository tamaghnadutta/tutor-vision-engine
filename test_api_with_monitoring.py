#!/usr/bin/env python3
"""
Quick test script to generate API traffic for monitoring dashboard demonstration
"""

import asyncio
import aiohttp
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "test-api-key-123")

# Test different approaches
APPROACHES = ["ocr_llm", "vlm_direct", "hybrid"]

async def make_test_request(session, approach, test_num):
    """Make a test request to the error detection API"""

    # Sample test data
    payload = {
        "question_url": "http://localhost:8080/data/sample_images/questions/Q1.jpeg",
        "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
        "bounding_box": {
            "minX": 0.1,
            "maxX": 0.9,
            "minY": 0.1,
            "maxY": 0.9
        },
        "user_id": f"monitoring_test_user_{approach}",
        "session_id": f"monitoring_session_{int(time.time())}",
        "question_id": f"monitoring_test_{approach}_{test_num}"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "X-Error-Detection-Approach": approach
    }

    print(f"🚀 Making request {test_num} with {approach} approach...")

    start_time = time.time()

    try:
        async with session.post(
            f"{API_BASE_URL}/api/v1/detect-error",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:

            duration = time.time() - start_time

            if response.status == 200:
                result = await response.json()

                # Extract results with new format support
                error_analysis = result.get('error_analysis', {})
                has_error = error_analysis.get('has_error', False)
                confidence = error_analysis.get('confidence', 0.0)

                print(f"   ✅ Success ({approach}): {duration:.2f}s, error={has_error}, conf={confidence:.3f}")
                return {"success": True, "approach": approach, "duration": duration}
            else:
                print(f"   ❌ Failed ({approach}): HTTP {response.status}")
                return {"success": False, "approach": approach, "duration": duration}

    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Exception ({approach}): {e}")
        return {"success": False, "approach": approach, "duration": duration, "error": str(e)}

async def test_health_endpoint(session):
    """Test the health endpoint"""
    try:
        async with session.get(f"{API_BASE_URL}/health") as response:
            if response.status == 200:
                health_data = await response.json()
                print(f"✅ API Health: {health_data}")
                return True
            else:
                print(f"❌ API Health failed: {response.status}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

async def test_metrics_endpoint(session):
    """Test the metrics endpoint"""
    try:
        async with session.get(f"{API_BASE_URL}/metrics") as response:
            if response.status == 200:
                metrics_data = await response.text()
                print(f"📊 Metrics endpoint working ({len(metrics_data)} bytes)")
                return True
            else:
                print(f"❌ Metrics endpoint failed: {response.status}")
                return False
    except Exception as e:
        print(f"❌ Cannot access metrics: {e}")
        return False

async def run_monitoring_demo():
    """Run a demo to generate traffic for monitoring dashboard"""

    print("🎭 API MONITORING DASHBOARD DEMO")
    print("=" * 60)
    print("📊 Open Grafana at http://localhost:3000 to see live metrics!")
    print("🔍 Dashboard: 'Error Detection API - Live Monitoring'")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Check API health
        if not await test_health_endpoint(session):
            print("❌ API is not healthy, exiting...")
            return

        # Check metrics endpoint
        await test_metrics_endpoint(session)

        print("\n🚀 Starting API traffic generation...")
        print("💡 Watch the Grafana dashboard for real-time metrics!")

        # Run multiple rounds of tests with different approaches
        for round_num in range(1, 4):
            print(f"\n🔄 Round {round_num}/3")
            print("-" * 40)

            # Test each approach
            for approach in APPROACHES:
                await make_test_request(session, approach, round_num)

                # Small delay between requests
                await asyncio.sleep(2)

            print(f"   ⏱️  Round {round_num} completed, waiting 10s...")
            await asyncio.sleep(10)

        print("\n🎉 Demo completed!")
        print("\n📊 Check your Grafana dashboard for the following metrics:")
        print("   • Request rate and response times")
        print("   • Error detection requests by approach")
        print("   • Processing time comparisons")
        print("   • Concurrent request handling")
        print("   • Model usage statistics")

        print("\n💡 Tips:")
        print("   • Refresh the dashboard to see latest data")
        print("   • Try different time ranges (5m, 15m, 1h)")
        print("   • Use the zoom feature on charts")

if __name__ == "__main__":
    asyncio.run(run_monitoring_demo())