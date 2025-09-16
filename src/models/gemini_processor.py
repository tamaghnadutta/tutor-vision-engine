"""
Gemini-based processor for OCR, LLM reasoning, and VLM analysis
"""

import asyncio
import time
import base64
import io
from typing import Optional, Dict, Any

import logging
from PIL import Image
from google import genai
from google.genai import types

from src.config.settings import get_settings
from src.models.gemini_schemas import (
    OCRResult, ErrorAnalysis, VisionAnalysisResult,
    GeminiErrorDetectionResult, QuestionAnalysis,
    MathematicsProblemAnalysis
)

logger = logging.getLogger(__name__)


class GeminiProcessor:
    """Unified Gemini processor for all AI tasks"""

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            self.client = genai.Client(api_key=self.settings.gemini_api_key)
            logger.info("Initialized Gemini client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        # Resize if too large (Gemini has size limits)
        if image.width > 2048 or image.height > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    async def extract_text_from_image(self, image: Image.Image) -> OCRResult:
        """Extract text from image using Gemini vision (replaces OCR)"""
        start_time = time.time()

        try:
            image_b64 = self._image_to_base64(image)

            prompt = """
            You are an expert OCR system specialized in mathematical content.
            Analyze this image and extract all text, mathematical expressions, and structural information.

            Pay special attention to:
            - Mathematical symbols, equations, and formulas
            - Step-by-step solutions
            - Handwritten text (may be unclear)
            - Any diagrams or visual elements

            Identify each mathematical step and whether it contains any errors.
            """

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

            processing_time = time.time() - start_time

            if response.parsed:
                result = response.parsed
                logger.info(f"OCR completed successfully in {processing_time:.2f}s")
                return result
            else:
                # Fallback if parsing fails
                logger.warning("Structured parsing failed, using fallback")
                return OCRResult(
                    extracted_text=response.text or "Could not extract text",
                    mathematical_expressions=[],
                    steps=[],
                    has_diagrams=False,
                    confidence=0.5
                )

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(
                extracted_text="Error during OCR processing",
                mathematical_expressions=[],
                steps=[],
                has_diagrams=False,
                confidence=0.0
            )

    async def analyze_question(self, question_image: Image.Image) -> QuestionAnalysis:
        """Analyze the question/problem image"""
        try:
            image_b64 = self._image_to_base64(question_image)

            prompt = """
            Analyze this mathematical question/problem image. Extract the complete question text
            and provide detailed analysis about the problem type, expected approach, and key concepts.

            If there are multiple choice options, extract them as well.
            """

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
                question_text="Could not analyze question",
                problem_type="unknown",
                expected_approach="unknown",
                key_concepts=[],
                difficulty_level="medium",
                has_multiple_choice=False
            )

        except Exception as e:
            logger.error(f"Question analysis failed: {e}")
            return QuestionAnalysis(
                question_text="Error analyzing question",
                problem_type="unknown",
                expected_approach="unknown",
                key_concepts=[],
                difficulty_level="medium",
                has_multiple_choice=False
            )

    async def analyze_math_problem(self, question_text: str, solution_text: str,
                                 context: Optional[Dict[str, Any]] = None) -> MathematicsProblemAnalysis:
        """Analyze mathematical problem and solution using text (replaces LLM reasoner)"""
        try:
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
                        types.Content(parts=[
                            types.Part.from_text(text=prompt)
                        ])
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MathematicsProblemAnalysis,
                        temperature=0.1
                    )
                )
            )

            return response.parsed if response.parsed else self._create_fallback_analysis()

        except Exception as e:
            logger.error(f"Math problem analysis failed: {e}")
            return self._create_fallback_analysis()

    async def analyze_images_directly(self, question_image: Image.Image,
                                    solution_image: Image.Image,
                                    context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """Direct image analysis with parallel processing and sequential fallback"""
        try:
            # TRY PARALLEL FIRST (faster for most cases)
            try:
                logger.info("Starting parallel image processing...")

                # Add timeout to parallel processing to prevent hanging
                question_task = asyncio.create_task(self.analyze_question(question_image))
                solution_task = asyncio.create_task(self.extract_text_from_image(solution_image))

                # Wait for both tasks with timeout
                question_analysis, solution_ocr = await asyncio.wait_for(
                    asyncio.gather(question_task, solution_task),
                    timeout=30.0  # 30 second timeout for parallel processing
                )

                logger.info("Parallel processing completed successfully")

            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Parallel processing failed ({e}), falling back to sequential")

                # FALLBACK TO SEQUENTIAL (original approach)
                logger.info("Starting sequential image processing...")
                question_analysis = await self.analyze_question(question_image)
                solution_ocr = await self.extract_text_from_image(solution_image)
                logger.info("Sequential processing completed")

            # Use text-based analysis for the actual error detection
            math_analysis = await self.analyze_math_problem(
                question_analysis.question_text,
                solution_ocr.extracted_text,
                context
            )

            # Combine results into VisionAnalysisResult format
            from src.models.gemini_schemas import ErrorAnalysis, SolutionAnalysis

            vision_result = VisionAnalysisResult(
                ocr_result=solution_ocr,
                error_analysis=math_analysis.error_analysis,
                solution_analysis=math_analysis.solution_quality,
                question_has_diagram=question_analysis.has_multiple_choice,  # Approximate
                solution_has_diagram=solution_ocr.has_diagrams,
                processing_notes="Processed images separately due to API limitations"
            )

            return vision_result

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return self._create_fallback_vision_result()

    async def detect_errors_comprehensive(self, question_image: Image.Image,
                                        solution_image: Image.Image,
                                        bounding_box: Optional[Dict[str, float]] = None,
                                        context: Optional[Dict[str, Any]] = None) -> GeminiErrorDetectionResult:
        """Comprehensive error detection using Gemini (main API method)"""
        start_time = time.time()

        try:
            # Apply bounding box if provided
            if bounding_box:
                solution_image = self._apply_bounding_box(solution_image, bounding_box)

            # Use direct vision analysis for best results
            vision_result = await self.analyze_images_directly(question_image, solution_image, context)

            # Convert to API response format
            result = GeminiErrorDetectionResult(
                y=vision_result.error_analysis.error_location_y,
                error=vision_result.error_analysis.error_description,
                correction=vision_result.error_analysis.correction,
                hint=vision_result.error_analysis.hint,
                solution_complete=vision_result.solution_analysis.solution_complete,
                contains_diagram=(vision_result.question_has_diagram or vision_result.solution_has_diagram),
                question_has_diagram=vision_result.question_has_diagram,
                solution_has_diagram=vision_result.solution_has_diagram,
                llm_used=True,
                solution_lines=vision_result.ocr_result.extracted_text.split('\n'),
                llm_ocr_lines=[step.text for step in vision_result.ocr_result.steps],
                confidence=min(vision_result.ocr_result.confidence, vision_result.error_analysis.confidence),
                processing_approach="gemini_2.5_flash"
            )

            processing_time = time.time() - start_time
            logger.info(f"Comprehensive error detection completed in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Comprehensive error detection failed: {e}")
            return self._create_fallback_error_result()

    def _apply_bounding_box(self, image: Image.Image, bounding_box: Dict[str, float]) -> Image.Image:
        """Apply bounding box to crop image"""
        return image.crop((
            bounding_box["minX"],
            bounding_box["minY"],
            bounding_box["maxX"],
            bounding_box["maxY"]
        ))

    def _create_fallback_analysis(self) -> MathematicsProblemAnalysis:
        """Create fallback analysis when parsing fails"""
        from src.models.gemini_schemas import MathStep, ErrorAnalysis, SolutionAnalysis

        return MathematicsProblemAnalysis(
            problem_type="unknown",
            student_approach="Could not analyze approach",
            steps=[],
            first_error_step=None,
            error_analysis=ErrorAnalysis(
                has_error=False,
                confidence=0.0
            ),
            solution_quality=SolutionAnalysis(
                solution_complete=False,
                total_steps=0,
                steps_with_errors=0,
                overall_approach_correct=False
            ),
            recommendations=["Please try again with a clearer image"]
        )

    def _create_fallback_vision_result(self) -> VisionAnalysisResult:
        """Create fallback vision result when parsing fails"""
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
            processing_notes="Analysis failed, fallback result provided"
        )

    def _create_fallback_error_result(self) -> GeminiErrorDetectionResult:
        """Create fallback error detection result"""
        return GeminiErrorDetectionResult(
            y=None,
            error=None,
            correction=None,
            hint="Unable to analyze solution. Please try again with a clearer image.",
            solution_complete=False,
            contains_diagram=False,
            question_has_diagram=False,
            solution_has_diagram=False,
            llm_used=True,
            solution_lines=["Could not extract solution steps"],
            llm_ocr_lines=["Processing failed"],
            confidence=0.0,
            processing_approach="gemini_2.5_flash_fallback"
        )