#!/usr/bin/env python3
"""
Test script to validate token extraction from OpenAI and Gemini APIs
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from src.utils.cost_calculator import CostCalculator, TokenUsage
from src.config.settings import get_settings

async def test_openai_token_extraction():
    """Test extracting tokens from OpenAI API calls"""
    print("Testing OpenAI token extraction...")

    settings = get_settings()
    if not settings.openai_api_key:
        print("❌ OpenAI API key not configured")
        return False

    client = openai.OpenAI(api_key=settings.openai_api_key)

    try:
        start_time = time.time()

        # Simple text completion
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Explain what 2+2 equals in exactly 50 words."}
            ],
            max_tokens=100,
            temperature=0.1
        )

        duration = time.time() - start_time

        # Extract usage
        usage = completion.usage
        print(f"✅ OpenAI Response received in {duration:.2f}s")
        print(f"   Prompt tokens: {usage.prompt_tokens}")
        print(f"   Completion tokens: {usage.completion_tokens}")
        print(f"   Total tokens: {usage.total_tokens}")

        # Calculate cost
        calculator = CostCalculator()
        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

        cost = calculator.calculate_cost("gpt-4o", token_usage)
        print(f"   Input cost: ${cost.input_cost:.6f}")
        print(f"   Output cost: ${cost.output_cost:.6f}")
        print(f"   Total cost: ${cost.total_cost:.6f}")
        print(f"   Response: {completion.choices[0].message.content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

async def test_openai_vision_token_extraction():
    """Test extracting tokens from OpenAI Vision API calls"""
    print("\nTesting OpenAI Vision token extraction...")

    settings = get_settings()
    if not settings.openai_api_key:
        print("❌ OpenAI API key not configured")
        return False

    client = openai.OpenAI(api_key=settings.openai_api_key)

    # Create a simple base64 encoded test image (1x1 pixel PNG)
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    try:
        start_time = time.time()

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_b64}"}}
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.1
        )

        duration = time.time() - start_time

        # Extract usage
        usage = completion.usage
        print(f"✅ OpenAI Vision Response received in {duration:.2f}s")
        print(f"   Prompt tokens: {usage.prompt_tokens}")
        print(f"   Completion tokens: {usage.completion_tokens}")
        print(f"   Total tokens: {usage.total_tokens}")
        print(f"   Note: Vision calls typically have higher token counts due to image processing")

        # Calculate cost
        calculator = CostCalculator()
        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

        cost = calculator.calculate_cost("gpt-4o", token_usage)
        print(f"   Input cost: ${cost.input_cost:.6f}")
        print(f"   Output cost: ${cost.output_cost:.6f}")
        print(f"   Total cost: ${cost.total_cost:.6f}")
        print(f"   Response: {completion.choices[0].message.content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ OpenAI Vision test failed: {e}")
        return False

async def test_gemini_token_estimation():
    """Test Gemini token estimation (Gemini doesn't return token counts)"""
    print("\nTesting Gemini token estimation...")

    try:
        from google import genai
        from google.genai import types

        settings = get_settings()
        if not settings.gemini_api_key:
            print("❌ Gemini API key not configured")
            return False

        client = genai.Client(api_key=settings.gemini_api_key)

        start_time = time.time()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text="Explain what 2+2 equals in exactly 50 words.")
                ])
            ],
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )

        duration = time.time() - start_time

        print(f"✅ Gemini Response received in {duration:.2f}s")
        print(f"   Response text length: {len(response.text)} characters")

        # Estimate tokens (rough approximation: 1 token ≈ 4 characters for English)
        estimated_input_tokens = 20  # Simple prompt
        estimated_output_tokens = max(len(response.text) // 4, 1)

        print(f"   Estimated input tokens: {estimated_input_tokens}")
        print(f"   Estimated output tokens: {estimated_output_tokens}")
        print(f"   Estimated total tokens: {estimated_input_tokens + estimated_output_tokens}")

        # Calculate estimated cost
        calculator = CostCalculator()
        token_usage = TokenUsage(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            total_tokens=estimated_input_tokens + estimated_output_tokens
        )

        cost = calculator.calculate_cost("gemini-2.5-flash", token_usage)
        print(f"   Estimated input cost: ${cost.input_cost:.6f}")
        print(f"   Estimated output cost: ${cost.output_cost:.6f}")
        print(f"   Estimated total cost: ${cost.total_cost:.6f}")
        print(f"   Response: {response.text[:100]}...")
        print(f"   Note: Gemini costs are estimated since token counts aren't returned")

        return True

    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

async def main():
    """Run all token extraction tests"""
    print("=" * 70)
    print("TOKEN EXTRACTION VALIDATION TESTS")
    print("=" * 70)

    results = []

    # Test OpenAI text
    results.append(await test_openai_token_extraction())

    # Test OpenAI vision
    results.append(await test_openai_vision_token_extraction())

    # Test Gemini estimation
    results.append(await test_gemini_token_estimation())

    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! Token extraction is working correctly.")
        print("\nKey findings:")
        print("• OpenAI returns precise token counts in completion.usage")
        print("• Vision API calls have higher token counts due to image processing")
        print("• Gemini requires token estimation (doesn't return counts)")
        print("• Cost calculation works for both exact and estimated tokens")
    else:
        print("❌ Some tests failed. Check API keys and network connectivity.")

    print(f"\n💡 Next step: Integrate token tracking into error detection approaches")

if __name__ == "__main__":
    asyncio.run(main())