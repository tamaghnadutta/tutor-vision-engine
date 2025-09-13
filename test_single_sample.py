#!/usr/bin/env python3
"""
Quick test of a single sample with Gemini robust provider
"""

import asyncio
import time
import sys
sys.path.append('.')

from src.models.error_detector import ErrorDetector
from src.data.dataset import ErrorDetectionDataset

async def test_single_sample():
    """Test a single sample with Gemini"""

    print("🔍 TESTING SINGLE SAMPLE WITH GEMINI ROBUST PROVIDER")
    print("=" * 60)

    # Load dataset
    dataset = ErrorDetectionDataset("./data/real_eval_dataset.json")
    samples = dataset.get_samples()

    # Test first sample with Gemini
    detector = ErrorDetector(approach="gemini")
    sample = samples[0]  # Just test the first one

    print(f"🧪 Testing Sample: {sample.id}")
    print(f"   Question: {sample.question_url}")
    print(f"   Solution: {sample.solution_url}")

    start_time = time.time()

    try:
        result = await asyncio.wait_for(
            detector.detect_errors(
                question_url=sample.question_url,
                solution_url=sample.solution_url,
                context={'sample_id': sample.id}
            ),
            timeout=60.0  # 1 minute timeout
        )

        duration = time.time() - start_time
        has_error = bool(result.get('error'))
        confidence = result.get('confidence', 0.0)

        print(f"   ✅ Success: {duration:.2f}s")
        print(f"   Has error: {has_error}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Provider: {result.get('processing_approach', 'unknown')}")

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        print(f"   ⏰ TIMEOUT: {duration:.2f}s")

    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ ERROR: {duration:.2f}s - {str(e)}")

    print("\n🏁 Single sample test completed")

if __name__ == "__main__":
    asyncio.run(test_single_sample())