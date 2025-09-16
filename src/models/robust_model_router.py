"""
Robust multi-model router with retry, timeout, and partial output handling
"""

import asyncio
import time
import base64
import io
from typing import Dict, Any, Optional, Type, TypeVar, Union
import logging
from PIL import Image
from pydantic import BaseModel

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    before_sleep_log, RetryError, AsyncRetrying
)

from src.config.settings import get_settings
from src.models.gemini_schemas import (
    OCRResult, ErrorAnalysis, VisionAnalysisResult,
    GeminiErrorDetectionResult, QuestionAnalysis,
    MathematicsProblemAnalysis
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class PartialOutputError(Exception):
    """Exception raised when partial output is received"""
    def __init__(self, message: str, partial_result: Optional[Dict] = None):
        super().__init__(message)
        self.partial_result = partial_result


class RobustModelProvider:
    """Base class for robust model providers with retry and timeout"""

    def __init__(self):
        self.settings = get_settings()
        self.max_retries = getattr(self.settings, 'max_retries', 3)
        self.timeout_seconds = getattr(self.settings, 'timeout_seconds', 30)

    async def analyze_question_with_retry(self, image: Image.Image, response_model: Type[T]) -> T:
        """Analyze question with retry and timeout handling"""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((Exception,)),
                before_sleep=before_sleep_log(logger, logging.WARNING)
            ):
                with attempt:
                    return await asyncio.wait_for(
                        self.analyze_question(image, response_model),
                        timeout=self.timeout_seconds
                    )
        except RetryError as e:
            logger.error(f"All retry attempts failed for question analysis: {e}")
            return self._create_fallback(response_model)
        except asyncio.TimeoutError:
            logger.error(f"Question analysis timed out after {self.timeout_seconds}s")
            return self._create_fallback(response_model)

    async def extract_text_with_retry(self, image: Image.Image, response_model: Type[T]) -> T:
        """Extract text with retry and timeout handling"""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((Exception,)),
                before_sleep=before_sleep_log(logger, logging.WARNING)
            ):
                with attempt:
                    return await asyncio.wait_for(
                        self.extract_text_from_image(image, response_model),
                        timeout=self.timeout_seconds
                    )
        except RetryError as e:
            logger.error(f"All retry attempts failed for text extraction: {e}")
            return self._create_fallback(response_model)
        except asyncio.TimeoutError:
            logger.error(f"Text extraction timed out after {self.timeout_seconds}s")
            return self._create_fallback(response_model)

    async def analyze_math_with_retry(self, question_text: str, solution_text: str,
                                    response_model: Type[T], context: Optional[Dict] = None) -> T:
        """Analyze math problem with retry and timeout handling"""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((Exception,)),
                before_sleep=before_sleep_log(logger, logging.WARNING)
            ):
                with attempt:
                    return await asyncio.wait_for(
                        self.analyze_math_problem(question_text, solution_text, response_model, context),
                        timeout=self.timeout_seconds
                    )
        except RetryError as e:
            logger.error(f"All retry attempts failed for math analysis: {e}")
            return self._create_fallback(response_model)
        except asyncio.TimeoutError:
            logger.error(f"Math analysis timed out after {self.timeout_seconds}s")
            return self._create_fallback(response_model)

    # Abstract methods to be implemented by subclasses
    async def analyze_question(self, image: Image.Image, response_model: Type[T]) -> T:
        raise NotImplementedError

    async def extract_text_from_image(self, image: Image.Image, response_model: Type[T]) -> T:
        raise NotImplementedError

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 response_model: Type[T], context: Optional[Dict] = None) -> T:
        raise NotImplementedError

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
        elif response_model == MathematicsProblemAnalysis:
            return MathematicsProblemAnalysis(
                problem_type="unknown",
                student_approach="Failed to analyze approach",
                has_error=False,
                error_description=None,
                error_location_y=None,
                correction=None,
                hint="Please try again with clearer images",
                solution_complete=False,
                confidence=0.0,
                recommendations=["Analysis failed, please retry"]
            )
        else:
            # Generic fallback
            return response_model()


class RobustGeminiProvider(RobustModelProvider):
    """Robust Gemini provider with retry, timeout, and partial output handling"""

    def __init__(self):
        super().__init__()
        from google import genai
        from google.genai import types

        # Configure Gemini client with timeout
        self.client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000)  # milliseconds
        )
        self.types = types

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string with optimization"""
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    async def analyze_question(self, image: Image.Image, response_model: Type[T]) -> T:
        """Analyze question image using Gemini with error handling"""
        image_b64 = self._image_to_base64(image)

        prompt = "Analyze this mathematical question image and extract the question text and key information."

        try:
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

            if response.parsed:
                return response.parsed
            elif response.text:
                # Handle partial output - try to extract what we can
                logger.warning("Gemini returned text but failed parsing, attempting fallback")
                raise PartialOutputError("Parsing failed but got text response", {"text": response.text})
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            logger.error(f"Gemini question analysis failed: {e}")
            raise

    async def extract_text_from_image(self, image: Image.Image, response_model: Type[T]) -> T:
        """Extract text from solution image using Gemini with error handling"""
        image_b64 = self._image_to_base64(image)

        prompt = """
        Extract all text and mathematical expressions from this handwritten solution image.
        Pay attention to mathematical symbols, equations, and step-by-step work.
        Identify each mathematical step and whether it contains any errors.
        """

        try:
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

            if response.parsed:
                return response.parsed
            elif response.text:
                logger.warning("Gemini returned text but failed parsing, attempting fallback")
                raise PartialOutputError("Parsing failed but got text response", {"text": response.text})
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            logger.error(f"Gemini text extraction failed: {e}")
            raise

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 response_model: Type[T], context: Optional[Dict] = None) -> T:
        """Analyze mathematical problem using Gemini with error handling"""
        prompt = f"""
        You are an expert mathematics tutor. Analyze this student's solution to identify errors.

        QUESTION: {question_text}

        STUDENT'S SOLUTION: {solution_text}

        Provide detailed step-by-step analysis, identify any errors, and give educational feedback.
        Focus on finding the first significant error and providing helpful guidance.
        """

        try:
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

            if response.parsed:
                return response.parsed
            elif response.text:
                logger.warning("Gemini returned text but failed parsing, attempting fallback")
                raise PartialOutputError("Parsing failed but got text response", {"text": response.text})
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            logger.error(f"Gemini math analysis failed: {e}")
            raise


class RobustOpenAIProvider(RobustModelProvider):
    """Robust OpenAI provider with retry, timeout, and partial output handling"""

    def __init__(self):
        super().__init__()
        try:
            import openai
            # Configure OpenAI client with timeout and retry settings
            self.client = openai.OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.timeout_seconds,
                max_retries=0  # We handle retries ourselves with tenacity
            )
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
        """Analyze question image using OpenAI with error handling"""
        image_url = self._image_to_base64_url(image)

        prompt = "Analyze this mathematical question image and extract the question text and key information."

        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.beta.chat.completions.parse(
                    model=self.settings.openai_model,
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

            if completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed
            elif completion.choices[0].message.content:
                logger.warning("OpenAI returned content but failed parsing, attempting fallback")
                raise PartialOutputError("Parsing failed but got content", {"content": completion.choices[0].message.content})
            else:
                raise Exception("Empty response from OpenAI")

        except Exception as e:
            logger.error(f"OpenAI question analysis failed: {e}")
            raise

    async def extract_text_from_image(self, image: Image.Image, response_model: Type[T]) -> T:
        """Extract text from solution image using OpenAI with error handling"""
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
                    model=self.settings.openai_model,
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

            if completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed
            elif completion.choices[0].message.content:
                logger.warning("OpenAI returned content but failed parsing, attempting fallback")
                raise PartialOutputError("Parsing failed but got content", {"content": completion.choices[0].message.content})
            else:
                raise Exception("Empty response from OpenAI")

        except Exception as e:
            logger.error(f"OpenAI text extraction failed: {e}")
            raise

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 response_model: Type[T], context: Optional[Dict] = None) -> T:
        """Analyze mathematical problem using OpenAI with error handling"""
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
                    model=self.settings.openai_model,
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

            if completion.choices[0].message.parsed:
                return completion.choices[0].message.parsed
            elif completion.choices[0].message.content:
                logger.warning("OpenAI returned content but failed parsing, attempting fallback")
                raise PartialOutputError("Parsing failed but got content", {"content": completion.choices[0].message.content})
            else:
                raise Exception("Empty response from OpenAI")

        except Exception as e:
            logger.error(f"OpenAI math analysis failed: {e}")
            raise


class RobustMultiModelProcessor:
    """Robust router with retry, timeout, and partial output handling"""

    def __init__(self, provider: str = "auto"):
        """
        Initialize robust multi-model processor

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
            self.provider = RobustGeminiProvider()
        elif provider == "openai":
            self.provider = RobustOpenAIProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"Initialized RobustMultiModelProcessor with {provider} provider")

    async def analyze_images_directly(self, question_image: Image.Image,
                                    solution_image: Image.Image,
                                    context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """
        Direct image analysis with robust error handling, retry, and timeout

        This method provides comprehensive error handling and recovery mechanisms.
        """
        start_time = time.time()

        try:
            # TRY PARALLEL FIRST with robust error handling
            try:
                logger.info(f"Starting robust parallel image processing with {self.provider_name}...")

                # Create parallel tasks with retry mechanisms
                question_task = asyncio.create_task(
                    self.provider.analyze_question_with_retry(question_image, QuestionAnalysis)
                )
                solution_task = asyncio.create_task(
                    self.provider.extract_text_with_retry(solution_image, OCRResult)
                )

                # Wait for both tasks with overall timeout
                question_analysis, solution_ocr = await asyncio.wait_for(
                    asyncio.gather(question_task, solution_task),
                    timeout=60.0  # Overall timeout for parallel processing
                )

                logger.info(f"Robust parallel processing completed successfully with {self.provider_name}")

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Robust parallel processing failed ({e}), falling back to sequential")

                # FALLBACK TO SEQUENTIAL with robust error handling
                logger.info(f"Starting robust sequential image processing with {self.provider_name}...")
                question_analysis = await self.provider.analyze_question_with_retry(question_image, QuestionAnalysis)
                solution_ocr = await self.provider.extract_text_with_retry(solution_image, OCRResult)
                logger.info(f"Robust sequential processing completed with {self.provider_name}")

            # Use text-based analysis for the actual error detection with robust handling
            math_analysis = await self.provider.analyze_math_with_retry(
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
                processing_notes=f"Robust processing with {self.provider_name} provider"
            )

            processing_time = time.time() - start_time
            logger.info(f"Robust image analysis completed in {processing_time:.2f}s using {self.provider_name}")

            return vision_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Robust image analysis failed with {self.provider_name}: {e}")
            return self._create_fallback_vision_result()

    def _create_fallback_vision_result(self) -> VisionAnalysisResult:
        """Create fallback vision result when all attempts fail"""
        from src.models.gemini_schemas import ErrorAnalysis, SolutionAnalysis

        return VisionAnalysisResult(
            ocr_result=OCRResult(
                extracted_text="Could not extract text - all attempts failed",
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
            processing_notes=f"Robust analysis failed with {self.provider_name}, fallback result provided"
        )


# Factory function for easy provider switching
def create_robust_processor(provider: str = "auto") -> RobustMultiModelProcessor:
    """
    Create a robust multi-model processor instance

    Args:
        provider: "gemini", "openai", or "auto"

    Returns:
        RobustMultiModelProcessor instance
    """
    return RobustMultiModelProcessor(provider)