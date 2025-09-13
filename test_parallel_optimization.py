#!/usr/bin/env python3
"""
Test script to validate parallel processing optimizations for Gemini API calls
Tests different approaches to achieve <10s p95 latency target
"""

import asyncio
import time
import base64
import io
from typing import Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types
from src.config.settings import get_settings
from src.models.gemini_schemas import OCRResult, QuestionAnalysis, VisionAnalysisResult

class ParallelOptimizationTester:
    """Test different optimization strategies"""

    def __init__(self):
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        if image.width > 2048 or image.height > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    async def load_test_images(self):
        """Load test images"""
        question_image = Image.open("/Users/tamaghnadutta/Downloads/meraki_labs/data/sample_images/questions/Q1.jpeg")
        solution_image = Image.open("/Users/tamaghnadutta/Downloads/meraki_labs/data/sample_images/attempts/Attempt1.jpeg")
        return question_image, solution_image

    async def test_current_sequential_approach(self, question_image: Image.Image, solution_image: Image.Image):
        """Test current sequential approach (baseline)"""
        print("🔄 Testing CURRENT SEQUENTIAL approach...")
        start_time = time.time()

        # Step 1: Analyze question
        step1_start = time.time()
        question_analysis = await self.analyze_question(question_image)
        step1_time = time.time() - step1_start
        print(f"   Question analysis: {step1_time:.2f}s")

        # Step 2: Extract solution text
        step2_start = time.time()
        solution_ocr = await self.extract_text_from_image(solution_image)
        step2_time = time.time() - step2_start
        print(f"   Solution OCR: {step2_time:.2f}s")

        # Step 3: Analyze problem
        step3_start = time.time()
        # Simulate text analysis (simplified)
        await asyncio.sleep(0.5)  # Simulate processing time
        step3_time = time.time() - step3_start
        print(f"   Text analysis: {step3_time:.2f}s")

        total_time = time.time() - start_time
        print(f"   ✅ SEQUENTIAL TOTAL: {total_time:.2f}s")
        return total_time

    async def test_parallel_approach(self, question_image: Image.Image, solution_image: Image.Image):
        """Test parallel processing of question and solution"""
        print("🚀 Testing PARALLEL approach...")
        start_time = time.time()

        # Steps 1&2: Parallel processing
        parallel_start = time.time()
        question_task = asyncio.create_task(self.analyze_question(question_image))
        solution_task = asyncio.create_task(self.extract_text_from_image(solution_image))

        question_analysis, solution_ocr = await asyncio.gather(question_task, solution_task)
        parallel_time = time.time() - parallel_start
        print(f"   Parallel processing: {parallel_time:.2f}s")

        # Step 3: Text analysis
        step3_start = time.time()
        await asyncio.sleep(0.5)  # Simulate processing time
        step3_time = time.time() - step3_start
        print(f"   Text analysis: {step3_time:.2f}s")

        total_time = time.time() - start_time
        print(f"   ✅ PARALLEL TOTAL: {total_time:.2f}s")
        return total_time

    async def test_combined_single_call_approach(self, question_image: Image.Image, solution_image: Image.Image):
        """Test single API call with both images"""
        print("⚡ Testing COMBINED SINGLE CALL approach...")
        start_time = time.time()

        try:
            question_b64 = self._image_to_base64(question_image)
            solution_b64 = self._image_to_base64(solution_image)

            prompt = '''
            Analyze these two images:
            - First image: Mathematical question/problem
            - Second image: Student's handwritten solution

            Extract question text, solution steps, identify errors, and provide feedback.
            '''

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=[
                        types.Content(parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                data=base64.b64decode(question_b64),
                                mime_type="image/jpeg"
                            ),
                            types.Part.from_bytes(
                                data=base64.b64decode(solution_b64),
                                mime_type="image/jpeg"
                            )
                        ])
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048
                    )
                )
            )

            total_time = time.time() - start_time
            print(f"   ✅ COMBINED TOTAL: {total_time:.2f}s")
            print(f"   Response preview: {response.text[:100]}...")
            return total_time

        except Exception as e:
            total_time = time.time() - start_time
            print(f"   ❌ COMBINED FAILED: {total_time:.2f}s - {e}")
            return total_time

    async def analyze_question(self, image: Image.Image) -> QuestionAnalysis:
        """Analyze question image"""
        image_b64 = self._image_to_base64(image)

        prompt = "Analyze this mathematical question image and extract the question text and key information."

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    types.Content(parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=base64.b64decode(image_b64),
                            mime_type="image/jpeg"
                        )
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuestionAnalysis,
                    temperature=0.1
                )
            )
        )

        return response.parsed if response.parsed else QuestionAnalysis(
            question_text="Failed to analyze",
            problem_type="unknown",
            expected_approach="unknown",
            key_concepts=[],
            difficulty_level="medium",
            has_multiple_choice=False
        )

    async def extract_text_from_image(self, image: Image.Image) -> OCRResult:
        """Extract text from solution image"""
        image_b64 = self._image_to_base64(image)

        prompt = '''
        Extract all text and mathematical expressions from this handwritten solution image.
        Pay attention to mathematical symbols, equations, and step-by-step work.
        '''

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    types.Content(parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=base64.b64decode(image_b64),
                            mime_type="image/jpeg"
                        )
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=OCRResult,
                    temperature=0.1
                )
            )
        )

        return response.parsed if response.parsed else OCRResult(
            extracted_text="Failed to extract",
            mathematical_expressions=[],
            steps=[],
            has_diagrams=False,
            confidence=0.0
        )

async def main():
    """Run optimization tests"""
    print("🧪 LATENCY OPTIMIZATION TESTS")
    print("=" * 50)
    print("Target: p95 ≤ 10s for ~5 concurrent requests")
    print("Current: p95 ≈ 43s (needs 4x improvement)")
    print()

    tester = ParallelOptimizationTester()

    # Load test images
    print("📂 Loading test images...")
    question_image, solution_image = await tester.load_test_images()
    print(f"   Question: {question_image.size}")
    print(f"   Solution: {solution_image.size}")
    print()

    # Test 1: Current sequential approach
    sequential_time = await tester.test_current_sequential_approach(question_image, solution_image)
    print()

    # Test 2: Parallel approach
    parallel_time = await tester.test_parallel_approach(question_image, solution_image)
    print()

    # Test 3: Combined single call
    combined_time = await tester.test_combined_single_call_approach(question_image, solution_image)
    print()

    # Analysis
    print("📊 OPTIMIZATION ANALYSIS")
    print("=" * 50)
    print(f"Sequential approach:     {sequential_time:.2f}s")
    print(f"Parallel approach:       {parallel_time:.2f}s ({((sequential_time - parallel_time) / sequential_time * 100):+.1f}%)")
    print(f"Combined single call:    {combined_time:.2f}s ({((sequential_time - combined_time) / sequential_time * 100):+.1f}%)")
    print()

    # Recommendations
    best_time = min(parallel_time, combined_time)
    improvement_needed = best_time / 10.0  # Factor needed to reach 10s target

    print("🎯 RECOMMENDATIONS")
    print("=" * 50)
    print(f"Best optimization:       {best_time:.2f}s")
    print(f"Target (10s):            {improvement_needed:.1f}x further improvement needed")

    if best_time <= 10:
        print("✅ TARGET ACHIEVED!")
    elif best_time <= 15:
        print("🟡 Close to target - additional optimizations needed")
    else:
        print("🔴 Significant optimizations still required")

    print()
    print("Next steps:")
    if combined_time < parallel_time:
        print("- ✅ Use combined single API call approach")
    else:
        print("- ✅ Use parallel processing approach")
    print("- 🔍 Consider image size reduction")
    print("- 🔍 Consider response caching")
    print("- 🔍 Consider connection pooling")

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    asyncio.run(main())