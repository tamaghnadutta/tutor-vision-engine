#!/usr/bin/env python3
"""
Debug script to check why evaluation gets stuck and test with better error handling
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.error_detector import ErrorDetector
from src.data.dataset import ErrorDetectionDataset

async def test_individual_samples():
    """Test each sample individually to isolate the issue"""

    print("🔍 DEBUGGING STUCK EVALUATION")
    print("=" * 50)

    # Load dataset
    dataset = ErrorDetectionDataset("./data/real_eval_dataset.json")
    samples = dataset.get_samples()

    print(f"Found {len(samples)} samples to test")
    print()

    approach="gemini"  # Test Gemini provider specifically

    # Test each sample individually
    detector = ErrorDetector(approach=approach)

    for i, sample in enumerate(samples, 1):
        print(f"🧪 Testing Sample {i}/{len(samples)}: {sample.id}")
        print(f"   Question: {sample.question_url}")
        print(f"   Solution: {sample.solution_url}")

        start_time = time.time()

        try:
            # Add timeout to detect hanging
            result = await asyncio.wait_for(
                detector.detect_errors(
                    question_url=sample.question_url,
                    solution_url=sample.solution_url,
                    context={'sample_id': sample.id}
                ),
                timeout=120.0  # 2 minute timeout (robust router handles internal retries)
            )

            duration = time.time() - start_time
            has_error = bool(result.get('error'))
            confidence = result.get('confidence', 0.0)

            print(f"   ✅ Success: {duration:.2f}s ({approach})")
            print(f"   Has error: {has_error}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Provider: {result.get('processing_approach', 'unknown')}")

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            print(f"   ⏰ TIMEOUT: {duration:.2f}s - Sample is hanging")
            print("   This is likely the cause of the stuck evaluation")
            break

        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ ERROR: {duration:.2f}s - {str(e)}")
            print(f"   Error type: {type(e).__name__}")

        print()

        # Add delay between samples to avoid rate limiting
        print("   💤 Waiting 2s to avoid rate limits...")
        await asyncio.sleep(2)

    print("🏁 Individual sample testing completed")

async def main():
    await test_individual_samples()

if __name__ == "__main__":
    asyncio.run(main())