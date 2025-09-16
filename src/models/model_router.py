"""
Multi-model router supporting both Gemini and OpenAI GPT-4o with Pydantic structured outputs
"""

import asyncio
import time
import base64
import io
from typing import Dict, Any, Optional, Type, TypeVar, Union
import logging
from PIL import Image
from pydantic import BaseModel

from src.config.settings import get_settings
from src.models.gemini_schemas import (
    OCRResult, ErrorAnalysis, VisionAnalysisResult,
    GeminiErrorDetectionResult, QuestionAnalysis,
    MathematicsProblemAnalysis
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class ModelProvider:
    """Base class for model providers"""

    async def analyze_question(self, image: Image.Image, response_model: Type[T]) -> T:
        raise NotImplementedError

    async def extract_text_from_image(self, image: Image.Image, response_model: Type[T]) -> T:
        raise NotImplementedError

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 response_model: Type[T], context: Optional[Dict] = None) -> T:
        raise NotImplementedError


class GeminiProvider(ModelProvider):
    """Gemini 2.5 Flash provider with structured outputs"""

    def __init__(self):
        self.settings = get_settings()
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.types = types

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    async def analyze_question(self, image: Image.Image, response_model: Type[T]) -> T:
        """Analyze question image using Gemini"""
        image_b64 = self._image_to_base64(image)

        prompt = "Analyze this mathematical question image and extract the question text and key information."

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    self.types.Content(parts=[
                        self.types.Part.from_text(text=prompt),
                        self.types.Part.from_bytes(
                            data=base64.b64decode(image_b64),
                            mime_type="image/jpeg"
                        )
                    ])
                ],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_model,
                    temperature=0.1
                )
            )
        )

        return response.parsed if response.parsed else self._create_fallback(response_model)

    async def extract_text_from_image(self, image: Image.Image, response_model: Type[T]) -> T:
        """Extract text from solution image using Gemini"""
        image_b64 = self._image_to_base64(image)

        prompt = """
        Extract all text and mathematical expressions from this handwritten solution image.
        Pay attention to mathematical symbols, equations, and step-by-step work.
        Identify each mathematical step and whether it contains any errors.
        """

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    self.types.Content(parts=[
                        self.types.Part.from_text(text=prompt),
                        self.types.Part.from_bytes(
                            data=base64.b64decode(image_b64),
                            mime_type="image/jpeg"
                        )
                    ])
                ],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_model,
                    temperature=0.1
                )
            )
        )

        return response.parsed if response.parsed else self._create_fallback(response_model)

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 response_model: Type[T], context: Optional[Dict] = None) -> T:
        """Analyze mathematical problem using Gemini"""
        prompt = f"""
        You are an expert mathematics tutor. Analyze this student's solution to identify errors.

        QUESTION: {question_text}

        STUDENT'S SOLUTION: {solution_text}

        Provide detailed step-by-step analysis, identify any errors, and give educational feedback.
        Focus on finding the first significant error and providing helpful guidance.
        """

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[
                    self.types.Content(parts=[
                        self.types.Part.from_text(text=prompt)
                    ])
                ],
                config=self.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_model,
                    temperature=0.1
                )
            )
        )

        return response.parsed if response.parsed else self._create_fallback(response_model)

    def _create_fallback(self, response_model: Type[T]) -> T:
        """Create fallback response for failed parsing"""
        if response_model == QuestionAnalysis:
            return QuestionAnalysis(
                question_text="Failed to analyze question",
                problem_type="unknown",
                expected_approach="unknown",
                key_concepts=[],
                difficulty_level="medium",
                has_multiple_choice=False
            )
        elif response_model == OCRResult:
            return OCRResult(
                extracted_text="Failed to extract text",
                mathematical_expressions=[],
                steps=[],
                has_diagrams=False,
                confidence=0.0
            )
        else:
            # Generic fallback
            return response_model()


class OpenAIProvider(ModelProvider):
    """OpenAI GPT-4o provider with structured outputs"""

    def __init__(self):
        self.settings = get_settings()
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL for OpenAI"""
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"

    async def analyze_question(self, image: Image.Image, response_model: Type[T]) -> T:
        """Analyze question image using OpenAI GPT-4o with structured outputs"""
        image_url = self._image_to_base64_url(image)

        prompt = "Analyze this mathematical question image and extract the question text and key information."

        try:
            # Using OpenAI's structured outputs with .parse()
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",  # Model that supports structured outputs
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    response_format=response_model,
                    temperature=0.1
                )
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._create_fallback(response_model)

    async def extract_text_from_image(self, image: Image.Image, response_model: Type[T]) -> T:
        """Extract text from solution image using OpenAI GPT-4o"""
        image_url = self._image_to_base64_url(image)

        prompt = """
        Extract all text and mathematical expressions from this handwritten solution image.
        Pay attention to mathematical symbols, equations, and step-by-step work.
        Identify each mathematical step and whether it contains any errors.
        """

        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    response_format=response_model,
                    temperature=0.1
                )
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            logger.error(f"OpenAI text extraction failed: {e}")
            return self._create_fallback(response_model)

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 response_model: Type[T], context: Optional[Dict] = None) -> T:
        """Analyze mathematical problem using OpenAI GPT-4o"""
        prompt = f"""
        You are an expert mathematics tutor. Analyze this student's solution to identify errors.

        QUESTION: {question_text}

        STUDENT'S SOLUTION: {solution_text}

        Provide detailed step-by-step analysis, identify any errors, and give educational feedback.
        Focus on finding the first significant error and providing helpful guidance.
        """

        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format=response_model,
                    temperature=0.1
                )
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            logger.error(f"OpenAI math analysis failed: {e}")
            return self._create_fallback(response_model)

    def _create_fallback(self, response_model: Type[T]) -> T:
        """Create fallback response for failed parsing"""
        # Same fallback logic as Gemini
        if response_model == QuestionAnalysis:
            return QuestionAnalysis(
                question_text="Failed to analyze question",
                problem_type="unknown",
                expected_approach="unknown",
                key_concepts=[],
                difficulty_level="medium",
                has_multiple_choice=False
            )
        elif response_model == OCRResult:
            return OCRResult(
                extracted_text="Failed to extract text",
                mathematical_expressions=[],
                steps=[],
                has_diagrams=False,
                confidence=0.0
            )
        else:
            return response_model()


class MultiModelProcessor:
    """Router that can use either Gemini or OpenAI based on configuration"""

    def __init__(self, provider: str = "auto"):
        """
        Initialize multi-model processor

        Args:
            provider: "gemini", "openai", or "auto" (auto-detect from settings)
        """
        self.settings = get_settings()

        if provider == "auto":
            # Auto-detect based on available API keys
            if hasattr(self.settings, 'openai_api_key') and self.settings.openai_api_key:
                provider = "openai"
            else:
                provider = "gemini"

        self.provider_name = provider

        if provider == "gemini":
            self.provider = GeminiProvider()
        elif provider == "openai":
            self.provider = OpenAIProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"Initialized MultiModelProcessor with {provider} provider")

    async def analyze_images_directly(self, question_image: Image.Image,
                                    solution_image: Image.Image,
                                    context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """
        Direct image analysis with parallel processing and provider routing

        This method maintains compatibility with the existing API while allowing
        provider switching through configuration.
        """
        start_time = time.time()

        try:
            # TRY PARALLEL FIRST (faster for most cases)
            try:
                logger.info(f"Starting parallel image processing with {self.provider_name}...")

                # Create parallel tasks
                question_task = asyncio.create_task(
                    self.provider.analyze_question(question_image, QuestionAnalysis)
                )
                solution_task = asyncio.create_task(
                    self.provider.extract_text_from_image(solution_image, OCRResult)
                )

                # Wait for both tasks with timeout
                question_analysis, solution_ocr = await asyncio.wait_for(
                    asyncio.gather(question_task, solution_task),
                    timeout=30.0
                )

                logger.info(f"Parallel processing completed successfully with {self.provider_name}")

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Parallel processing failed ({e}), falling back to sequential")

                # FALLBACK TO SEQUENTIAL
                logger.info(f"Starting sequential image processing with {self.provider_name}...")
                question_analysis = await self.provider.analyze_question(question_image, QuestionAnalysis)
                solution_ocr = await self.provider.extract_text_from_image(solution_image, OCRResult)
                logger.info(f"Sequential processing completed with {self.provider_name}")

            # Use text-based analysis for the actual error detection
            math_analysis = await self.provider.analyze_math_problem(
                question_analysis.question_text,
                solution_ocr.extracted_text,
                MathematicsProblemAnalysis,
                context
            )

            # Combine results into VisionAnalysisResult format
            vision_result = VisionAnalysisResult(
                ocr_result=solution_ocr,
                error_analysis=math_analysis.error_analysis,
                solution_analysis=math_analysis.solution_quality,
                question_has_diagram=question_analysis.has_multiple_choice,
                solution_has_diagram=solution_ocr.has_diagrams,
                processing_notes=f"Processed with {self.provider_name} provider"
            )

            processing_time = time.time() - start_time
            logger.info(f"Image analysis completed in {processing_time:.2f}s using {self.provider_name}")

            return vision_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Image analysis failed with {self.provider_name}: {e}")
            return self._create_fallback_vision_result()

    def _create_fallback_vision_result(self) -> VisionAnalysisResult:
        """Create fallback vision result when analysis fails"""
        from src.models.gemini_schemas import ErrorAnalysis, SolutionAnalysis

        return VisionAnalysisResult(
            ocr_result=OCRResult(
                extracted_text="Could not extract text",
                mathematical_expressions=[],
                steps=[],
                has_diagrams=False,
                confidence=0.0
            ),
            error_analysis=ErrorAnalysis(
                has_error=False,
                confidence=0.0
            ),
            solution_analysis=SolutionAnalysis(
                solution_complete=False,
                total_steps=0,
                steps_with_errors=0,
                overall_approach_correct=False
            ),
            question_has_diagram=False,
            solution_has_diagram=False,
            processing_notes=f"Analysis failed with {self.provider_name}, fallback result provided"
        )


# Factory function for easy provider switching
def create_processor(provider: str = "auto") -> MultiModelProcessor:
    """
    Create a multi-model processor instance

    Args:
        provider: "gemini", "openai", or "auto"

    Returns:
        MultiModelProcessor instance
    """
    return MultiModelProcessor(provider)