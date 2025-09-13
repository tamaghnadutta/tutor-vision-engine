#!/usr/bin/env python3
"""
Test image size optimization for further latency reduction
"""

import asyncio
import time
import base64
import io
from PIL import Image
from google import genai
from google.genai import types
from src.config.settings import get_settings

class ImageOptimizationTester:
    def __init__(self):
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def _resize_image(self, image: Image.Image, max_dimension: int) -> Image.Image:
        """Resize image to max dimension"""
        if image.width > max_dimension or image.height > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def _image_to_base64(self, image: Image.Image, quality: int = 85) -> str:
        """Convert PIL Image to base64 with quality control"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    async def analyze_question_simple(self, image: Image.Image) -> str:
        """Simple question analysis"""
        image_b64 = self._image_to_base64(image)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    types.Content(parts=[
                        types.Part.from_text(text="Extract the question text from this image."),
                        types.Part.from_bytes(
                            data=base64.b64decode(image_b64),
                            mime_type="image/jpeg"
                        )
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512
                )
            )
        )
        return response.text or "Question analysis failed"

    async def extract_text_simple(self, image: Image.Image) -> str:
        """Simple text extraction"""
        image_b64 = self._image_to_base64(image)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    types.Content(parts=[
                        types.Part.from_text(text="Extract all text and mathematical expressions from this solution image."),
                        types.Part.from_bytes(
                            data=base64.b64decode(image_b64),
                            mime_type="image/jpeg"
                        )
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512
                )
            )
        )
        return response.text or "Text extraction failed"

    async def test_parallel_call_with_different_sizes(self):
        """Test PARALLEL API calls with different image sizes"""

        # Load original images
        question_image = Image.open("/Users/tamaghnadutta/Downloads/meraki_labs/data/sample_images/questions/Q1.jpeg")
        solution_image = Image.open("/Users/tamaghnadutta/Downloads/meraki_labs/data/sample_images/attempts/Attempt1.jpeg")

        print(f"Original sizes: Q={question_image.size}, S={solution_image.size}")

        test_configs = [
            {"max_dim": 2048, "quality": 85, "name": "High Quality (current)"},
            {"max_dim": 1024, "quality": 85, "name": "Medium Size"},
            {"max_dim": 1024, "quality": 70, "name": "Medium Size + Lower Quality"},
            {"max_dim": 768, "quality": 70, "name": "Small Size"},
        ]

        results = []

        for config in test_configs:
            print(f"\n🧪 Testing {config['name']} (max_dim={config['max_dim']}, quality={config['quality']})")

            # Resize images
            q_resized = self._resize_image(question_image.copy(), config['max_dim'])
            s_resized = self._resize_image(solution_image.copy(), config['max_dim'])

            print(f"   Resized to: Q={q_resized.size}, S={s_resized.size}")

            # Convert to base64
            q_b64 = self._image_to_base64(q_resized, config['quality'])
            s_b64 = self._image_to_base64(s_resized, config['quality'])

            print(f"   Base64 sizes: Q={len(q_b64)} chars, S={len(s_b64)} chars")

            # Test PARALLEL API calls timing (proven approach)
            start_time = time.time()

            try:
                # Use parallel approach like in successful test
                question_task = asyncio.create_task(self.analyze_question_simple(q_resized))
                solution_task = asyncio.create_task(self.extract_text_simple(s_resized))

                # Wait for both tasks to complete
                question_result, solution_result = await asyncio.gather(question_task, solution_task)

                # Simple text analysis
                response_text = f"Question: {question_result[:100]}... Solution: {solution_result[:100]}..."

                duration = time.time() - start_time
                print(f"   ✅ Success: {duration:.2f}s")
                print(f"   Response: {response_text[:100]}...")

                results.append({
                    'config': config['name'],
                    'duration': duration,
                    'success': True,
                    'q_size': q_resized.size,
                    's_size': s_resized.size,
                    'data_size': len(q_b64) + len(s_b64)
                })

            except Exception as e:
                duration = time.time() - start_time
                print(f"   ❌ Failed: {duration:.2f}s - {e}")
                results.append({
                    'config': config['name'],
                    'duration': duration,
                    'success': False
                })

        return results

async def main():
    print("🖼️ IMAGE SIZE OPTIMIZATION TESTS")
    print("=" * 50)
    print("Goal: Reduce image processing time to reach <10s target")
    print()

    tester = ImageOptimizationTester()
    results = await tester.test_parallel_call_with_different_sizes()

    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)

    successful_results = [r for r in results if r['success']]

    if successful_results:
        fastest = min(successful_results, key=lambda x: x['duration'])
        baseline = next((r for r in successful_results if 'current' in r['config'].lower()), successful_results[0])

        print("Timing Results:")
        for result in successful_results:
            improvement = ""
            if result != baseline:
                pct = ((baseline['duration'] - result['duration']) / baseline['duration']) * 100
                improvement = f" ({pct:+.1f}%)"
            print(f"  {result['config']:25} {result['duration']:6.2f}s{improvement}")

        print(f"\n🏆 BEST CONFIGURATION: {fastest['config']}")
        print(f"   Time: {fastest['duration']:.2f}s")
        print(f"   Image sizes: Q={fastest['q_size']}, S={fastest['s_size']}")
        print(f"   Data size: {fastest['data_size']:,} chars")

        if fastest['duration'] <= 10:
            print("\n🎯 ✅ TARGET ACHIEVED! <10s latency reached")
        else:
            remaining = fastest['duration'] - 10
            print(f"\n🎯 🟡 Close! Need {remaining:.1f}s more reduction")
            print("   Additional optimizations:")
            print("   - Connection pooling")
            print("   - Response caching")
            print("   - Concurrent request batching")

    else:
        print("❌ All tests failed")

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    asyncio.run(main())