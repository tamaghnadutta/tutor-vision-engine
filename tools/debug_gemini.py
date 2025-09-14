#!/usr/bin/env python3
"""
Debug script to test Gemini API calls
"""

import asyncio
import base64
import io
import sys
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings

async def test_basic_gemini():
    """Test basic Gemini functionality"""

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    # Test 1: Simple text-only call
    print("Test 1: Simple text call...")
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=settings.gemini_model,
                contents="Hello, can you respond with a simple greeting?"
            )
        )
        print(f"✅ Text response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ Text call failed: {e}")

    # Test 2: Download and process image
    print("\nTest 2: Image processing...")
    try:
        import httpx
        async with httpx.AsyncClient() as client_http:
            response = await client_http.get("http://localhost:8080/data/sample_images/questions/Q1.jpeg")
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                print(f"✅ Image loaded: {image.size}")

                # Convert to base64
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                print(f"✅ Image converted to base64: {len(image_b64)} chars")

                # Test 3: Simple image + text call (no structured output)
                print("\nTest 3: Simple image analysis...")
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=settings.gemini_model,
                        contents=[
                            types.Content(parts=[
                                types.Part.from_text(text="What do you see in this image?"),
                                types.Part.from_bytes(
                                    data=base64.b64decode(image_b64),
                                    mime_type="image/jpeg"
                                )
                            ])
                        ]
                    )
                )
                print(f"✅ Simple image analysis: {response.text[:100]}...")

                # Test 4: Structured output with single image
                print("\nTest 4: Structured output with single image...")
                try:
                    from src.models.gemini_schemas import OCRResult
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text="Extract text from this image and provide structured analysis."),
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
                    print(f"✅ Structured output: {response.parsed}")
                except Exception as e:
                    print(f"❌ Structured output failed: {e}")

                # Test 5: Multiple images with structured output
                print("\nTest 5: Multiple images with structured output...")
                try:
                    # Load second image
                    response2 = await client_http.get("http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg")
                    if response2.status_code == 200:
                        image2 = Image.open(io.BytesIO(response2.content))
                        buffer2 = io.BytesIO()
                        image2.save(buffer2, format='JPEG', quality=85)
                        image2_bytes = buffer2.getvalue()
                        image2_b64 = base64.b64encode(image2_bytes).decode('utf-8')

                        from src.models.gemini_schemas import VisionAnalysisResult
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: client.models.generate_content(
                                model=settings.gemini_model,
                                contents=[
                                    types.Content(parts=[
                                        types.Part.from_text(text="Analyze these two images: first is a question, second is a solution."),
                                        types.Part.from_bytes(
                                            data=base64.b64decode(image_b64),
                                            mime_type="image/jpeg"
                                        ),
                                        types.Part.from_bytes(
                                            data=base64.b64decode(image2_b64),
                                            mime_type="image/jpeg"
                                        )
                                    ])
                                ],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=VisionAnalysisResult,
                                    temperature=0.1
                                )
                            )
                        )
                        print(f"✅ Multiple images with structured output: Success!")
                    else:
                        print(f"❌ Failed to load second image: {response2.status_code}")
                except Exception as e:
                    print(f"❌ Multiple images failed: {e}")

                # Test 6: QuestionAnalysis schema (the one that's failing)
                print("\nTest 6: QuestionAnalysis schema...")
                try:
                    from src.models.gemini_schemas import QuestionAnalysis
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text="Analyze this question image and extract information."),
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
                    print(f"✅ QuestionAnalysis: {response.parsed}")
                except Exception as e:
                    print(f"❌ QuestionAnalysis failed: {e}")

                # Test 7: MathematicsProblemAnalysis schema (text-based analysis)
                print("\nTest 7: MathematicsProblemAnalysis schema...")
                try:
                    from src.models.gemini_schemas import MathematicsProblemAnalysis
                    test_prompt = """
                    You are an expert mathematics tutor. Analyze this student's solution to identify errors.

                    QUESTION: Find the probability that the student knows the answer.

                    STUDENT'S SOLUTION: Using Bayes theorem, P(K|C) = P(C|K) * P(K) / P(C)

                    Provide detailed step-by-step analysis, identify any errors, and give educational feedback.
                    Focus on finding the first significant error and providing helpful guidance.
                    """

                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text=test_prompt)
                                ])
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=MathematicsProblemAnalysis,
                                temperature=0.1
                            )
                        )
                    )
                    print(f"✅ MathematicsProblemAnalysis: Success!")
                    print(f"   Error analysis: {response.parsed.error_analysis.has_error}")
                except Exception as e:
                    print(f"❌ MathematicsProblemAnalysis failed: {e}")

                # Test 8: Test individual sub-schemas to isolate the issue
                print("\nTest 8: Testing individual schemas...")
                try:
                    from src.models.gemini_schemas import ErrorAnalysis
                    simple_prompt = "Analyze this: 2 + 2 = 5. Is there an error?"

                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text=simple_prompt)
                                ])
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=ErrorAnalysis,
                                temperature=0.1
                            )
                        )
                    )
                    print(f"✅ ErrorAnalysis schema: Success!")
                except Exception as e:
                    print(f"❌ ErrorAnalysis schema failed: {e}")

                # Test 9: Test SolutionAnalysis schema
                print("\nTest 9: SolutionAnalysis schema...")
                try:
                    from src.models.gemini_schemas import SolutionAnalysis
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text="Analyze this solution quality: 2 + 2 = 4")
                                ])
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SolutionAnalysis,
                                temperature=0.1
                            )
                        )
                    )
                    print(f"✅ SolutionAnalysis schema: Success!")
                except Exception as e:
                    print(f"❌ SolutionAnalysis schema failed: {e}")

                # Test 10: Test MathStep schema
                print("\nTest 10: MathStep schema...")
                try:
                    from src.models.gemini_schemas import MathStep
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text="Analyze this step: Step 1: 2 + 2 = 4")
                                ])
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=MathStep,
                                temperature=0.1
                            )
                        )
                    )
                    print(f"✅ MathStep schema: Success!")
                except Exception as e:
                    print(f"❌ MathStep schema failed: {e}")

                # Test 11: Simplified MathematicsProblemAnalysis
                print("\nTest 11: Simplified MathematicsProblemAnalysis...")
                try:
                    from pydantic import BaseModel
                    from typing import List, Optional

                    class SimplifiedMathAnalysis(BaseModel):
                        problem_type: str
                        student_approach: str
                        has_error: bool
                        error_description: Optional[str] = None
                        recommendations: List[str]

                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text="""Analyze this problem: QUESTION: Find probability SOLUTION: Using Bayes theorem""")
                                ])
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SimplifiedMathAnalysis,
                                temperature=0.1
                            )
                        )
                    )
                    print(f"✅ Simplified MathAnalysis: Success!")
                except Exception as e:
                    print(f"❌ Simplified MathAnalysis failed: {e}")

                # Test 12: Fixed MathematicsProblemAnalysis (should work now!)
                print("\nTest 12: Fixed MathematicsProblemAnalysis...")
                try:
                    from src.models.gemini_schemas import MathematicsProblemAnalysis
                    test_prompt = """
                    You are an expert mathematics tutor. Analyze this student's solution to identify errors.

                    QUESTION: Find the probability that the student knows the answer.

                    STUDENT'S SOLUTION: Using Bayes theorem, P(K|C) = P(C|K) * P(K) / P(C)

                    Provide detailed step-by-step analysis, identify any errors, and give educational feedback.
                    Focus on finding the first significant error and providing helpful guidance.
                    """

                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=settings.gemini_model,
                            contents=[
                                types.Content(parts=[
                                    types.Part.from_text(text=test_prompt)
                                ])
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=MathematicsProblemAnalysis,
                                temperature=0.1
                            )
                        )
                    )
                    print(f"✅ Fixed MathematicsProblemAnalysis: SUCCESS!")
                    print(f"   Has error: {response.parsed.has_error}")
                    print(f"   Error description: {response.parsed.error_description}")
                    print(f"   Compatibility - error_analysis.has_error: {response.parsed.error_analysis.has_error}")
                except Exception as e:
                    print(f"❌ Fixed MathematicsProblemAnalysis failed: {e}")

            else:
                print(f"❌ Failed to download image: {response.status_code}")
    except Exception as e:
        print(f"❌ Image processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_basic_gemini())