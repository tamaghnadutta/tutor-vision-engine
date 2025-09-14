#!/usr/bin/env python3
"""
Test script for the multi-model router supporting both Gemini and OpenAI
"""

import asyncio
import time
from PIL import Image
from src.models.model_router import MultiModelProcessor, create_processor
from src.models.error_detector import ErrorDetector
from src.config.settings import get_settings

async def test_model_providers():
    """Test both Gemini and OpenAI providers (if available)"""
    print("🧪 MULTI-MODEL ROUTER TESTS")
    print("=" * 50)

    settings = get_settings()

    # Load test images
    question_image = Image.open("/Users/tamaghnadutta/Downloads/meraki_labs/data/sample_images/questions/Q1.jpeg")
    solution_image = Image.open("/Users/tamaghnadutta/Downloads/meraki_labs/data/sample_images/attempts/Attempt1.jpeg")

    print(f"Test images loaded: Q={question_image.size}, S={solution_image.size}")
    print()

    # Test 1: Gemini Provider
    print("🔮 Testing Gemini Provider")
    print("-" * 30)

    start_time = time.time()
    try:
        gemini_processor = create_processor("gemini")
        result = await gemini_processor.analyze_images_directly(question_image, solution_image)

        duration = time.time() - start_time
        print(f"✅ Gemini Success: {duration:.2f}s")
        print(f"   Has error: {bool(result.error_analysis.error_description)}")
        print(f"   Confidence: {result.error_analysis.confidence:.2f}")
        print(f"   Processing notes: {result.processing_notes}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Gemini Failed: {duration:.2f}s - {e}")

    print()

    # Test 2: OpenAI Provider (if API key available)
    if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
        print("🤖 Testing OpenAI Provider")
        print("-" * 30)

        start_time = time.time()
        try:
            openai_processor = create_processor("openai")
            result = await openai_processor.analyze_images_directly(question_image, solution_image)

            duration = time.time() - start_time
            print(f"✅ OpenAI Success: {duration:.2f}s")
            print(f"   Has error: {bool(result.error_analysis.error_description)}")
            print(f"   Confidence: {result.error_analysis.confidence:.2f}")
            print(f"   Processing notes: {result.processing_notes}")

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ OpenAI Failed: {duration:.2f}s - {e}")
    else:
        print("🤖 OpenAI Provider: SKIPPED (no API key found)")

    print()

    # Test 3: Auto-detection
    print("🎯 Testing Auto Provider Detection")
    print("-" * 30)

    start_time = time.time()
    try:
        auto_processor = create_processor("auto")
        result = await auto_processor.analyze_images_directly(question_image, solution_image)

        duration = time.time() - start_time
        print(f"✅ Auto Success: {duration:.2f}s")
        print(f"   Selected provider: {auto_processor.provider_name}")
        print(f"   Has error: {bool(result.error_analysis.error_description)}")
        print(f"   Confidence: {result.error_analysis.confidence:.2f}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Auto Failed: {duration:.2f}s - {e}")

    print()

async def test_error_detector_integration():
    """Test the updated ErrorDetector with multi-model support"""
    print("🔧 ERROR DETECTOR INTEGRATION TESTS")
    print("=" * 50)

    # Test with different approaches
    approaches = ["gemini", "auto"]

    for approach in approaches:
        print(f"🧪 Testing ErrorDetector with approach='{approach}'")
        print("-" * 30)

        start_time = time.time()
        try:
            detector = ErrorDetector(approach=approach)
            result = await detector.detect_errors(
                question_url="http://localhost:8080/data/sample_images/questions/Q1.jpeg",
                solution_url="http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
                context={'test': 'integration'}
            )

            duration = time.time() - start_time
            print(f"✅ {approach.title()} Success: {duration:.2f}s")
            print(f"   Has error: {bool(result.get('error'))}")
            print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"   Processing approach: {result.get('processing_approach')}")

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {approach.title()} Failed: {duration:.2f}s - {e}")

        print()

async def main():
    """Run all tests"""
    print("🚀 MULTI-MODEL ROUTER TEST SUITE")
    print("="*60)
    print()

    await test_model_providers()
    await test_error_detector_integration()

    print("🏁 All tests completed!")
    print()
    print("💡 Configuration Tips:")
    print("   - Set MODEL_PROVIDER=gemini to force Gemini")
    print("   - Set MODEL_PROVIDER=openai to force OpenAI (requires OPENAI_API_KEY)")
    print("   - Set MODEL_PROVIDER=auto for automatic detection")
    print("   - Add OPENAI_API_KEY to .env to enable OpenAI support")

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    asyncio.run(main())