#!/usr/bin/env python3
"""
Quick test to check if confidence is properly returned by the API
"""

import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv

load_dotenv()

async def test_api_confidence():
    """Test API confidence value"""

    print("🔍 TESTING API CONFIDENCE")
    print("=" * 40)

    API_BASE_URL = "http://localhost:8000"
    API_KEY = os.getenv("API_KEY", "test-api-key-123")

    payload = {
        "question_url": "http://localhost:8080/data/sample_images/questions/Q1.jpeg",
        "solution_url": "http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
        "bounding_box": {
            "minX": 0.05, "maxX": 0.65,
            "minY": 0.05, "maxY": 0.55
        },
        "user_id": "confidence_test"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE_URL}/api/v1/detect-error",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:

            if response.status == 200:
                result = await response.json()

                print(f"✅ API Status: {response.status}")
                print(f"📊 Confidence: {result.get('confidence', 'MISSING')}")
                print(f"📊 Processing Approach: {result.get('processing_approach', 'MISSING')}")
                print(f"📊 Processing Time: {result.get('processing_time', 'MISSING')}")
                print(f"📊 Has Error: {result.get('error') is not None}")

                print(f"\n🔍 All response keys:")
                for key, value in result.items():
                    print(f"   {key}: {type(value)} = {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")

            else:
                error_text = await response.text()
                print(f"❌ API Error {response.status}: {error_text}")

if __name__ == "__main__":
    asyncio.run(test_api_confidence())