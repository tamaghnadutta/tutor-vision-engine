"""
Error Detection Approaches as per assignment requirements:
1. OCR→LLM: GPT-4V (OCR) → GPT-4o/Gemini-2.5-Flash (text reasoning)
2. Direct VLM: GPT-4V or Gemini-2.5-Flash (single call)
3. Hybrid: Run both approaches, compare confidence scores, ensemble results
"""

import asyncio
import time
import base64
import io
from typing import Dict, Any, Optional, Tuple
import logging
from PIL import Image
from abc import ABC, abstractmethod

from src.config.settings import get_settings
from src.models.gemini_schemas import (
    VisionAnalysisResult, OCRResult, ErrorAnalysis, SolutionAnalysis,
    QuestionAnalysis, MathStep, OCRLLMReasoningResult, DirectVLMResult
)
from src.utils.api_tracker import api_tracker

logger = logging.getLogger(__name__)


class BaseErrorDetectionApproach(ABC):
    """Base class for error detection approaches"""

    def __init__(self):
        self.settings = get_settings()

    @abstractmethod
    async def detect_errors(self, question_image: Image.Image, solution_image: Image.Image,
                          context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """Detect errors using this approach"""
        pass

    @abstractmethod
    def get_approach_name(self) -> str:
        """Get the name of this approach"""
        pass


class OCRToLLMApproach(BaseErrorDetectionApproach):
    """OCR→LLM: GPT-4V (OCR) → GPT-4o/Gemini-2.5-Flash (text reasoning)"""

    def __init__(self):
        super().__init__()
        # Import OpenAI for GPT-4V OCR
        try:
            import openai
            self.ocr_client = openai.OpenAI(api_key=self.settings.openai_api_key)
        except ImportError:
            raise ImportError("OpenAI library required for OCR→LLM approach. Run: pip install openai")

        # Set up reasoning provider (GPT-4o or Gemini-2.5-Flash)
        reasoning_provider = self.settings.reasoning_provider
        if reasoning_provider == "auto":
            reasoning_provider = "gemini" if self.settings.gemini_api_key else "openai"

        if reasoning_provider == "gemini":
            from google import genai
            from google.genai import types
            self.reasoning_client = genai.Client(api_key=self.settings.gemini_api_key)
            self.genai_types = types
            self.reasoning_model = self.settings.gemini_model
        else:
            self.reasoning_client = self.ocr_client  # Same OpenAI client
            self.reasoning_model = self.settings.openai_model

        self.reasoning_provider = reasoning_provider
        logger.info(f"Initialized OCR→LLM: GPT-4V (OCR) → {reasoning_provider} (reasoning)")

    def get_approach_name(self) -> str:
        return f"ocr_llm_gpt4v_to_{self.reasoning_provider}"

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

    async def _extract_text_with_gpt4v(self, image: Image.Image, image_type: str) -> str:
        """Extract text from image using GPT-4V"""
        image_url = self._image_to_base64_url(image)

        if image_type == "question":
            prompt = """Extract the mathematical question or problem statement from this image.
            Focus on getting the exact text and mathematical expressions. Be precise and complete."""
        else:  # solution
            prompt = """Extract ALL text and mathematical expressions from this handwritten solution image.
            Include every step, equation, calculation, and written work. Pay attention to:
            - Mathematical symbols and equations
            - Step numbers or labels
            - All calculations and intermediate steps
            - Any annotations or corrections

            Present the extracted text exactly as written, preserving the step-by-step structure."""

        try:
            start_time = time.time()
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ocr_client.chat.completions.create(
                    model=self.settings.openai_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
            )

            # Track the API call
            duration = time.time() - start_time
            api_tracker.track_openai_call(
                completion,
                self.settings.openai_vision_model,
                f"OCR extraction from {image_type} image",
                duration
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"GPT-4V OCR failed for {image_type}: {e}")
            return f"OCR extraction failed for {image_type}"

    async def _analyze_with_reasoning_model(self, question_text: str, solution_text: str,
                                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze extracted text using the reasoning model (GPT-4o or Gemini)"""

        prompt = f"""You are an expert mathematics tutor. Analyze this student's solution to identify errors.

QUESTION: {question_text}

STUDENT'S SOLUTION: {solution_text}

Your task:
1. Identify if there are any mathematical errors in the solution
2. If errors exist, find the FIRST significant error
3. Provide educational feedback with corrections and hints
4. Determine if the solution is complete

Respond in JSON format with these exact fields:
{{
    "has_error": true/false,
    "error_description": "string or null",
    "error_location_y": 123.45 (numeric pixel location) or null,
    "correction": "string or null",
    "hint": "helpful hint for student",
    "solution_complete": true/false,
    "confidence": 0.85 (number between 0.0-1.0),
    "step_analysis": ["step1 analysis", "step2 analysis", ...]
}}

IMPORTANT: error_location_y must be a NUMBER (e.g. 150.0) or null, NOT a string with words."""

        try:
            if self.reasoning_provider == "gemini":
                # Use Gemini for reasoning
                start_time = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.reasoning_client.models.generate_content(
                        model=self.reasoning_model,
                        contents=[
                            self.genai_types.Content(parts=[
                                self.genai_types.Part.from_text(text=prompt)
                            ])
                        ],
                        config=self.genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=OCRLLMReasoningResult,
                            temperature=0.1
                        )
                    )
                )

                # Track the API call (estimated tokens for Gemini)
                duration = time.time() - start_time
                api_tracker.track_gemini_call(
                    response,
                    self.reasoning_model,
                    "OCR→LLM Gemini reasoning analysis",
                    duration,
                    estimated_input_tokens=len(prompt.split()) * 1.3,
                    estimated_output_tokens=len(response.text.split()) * 1.3
                )

                result_obj = OCRLLMReasoningResult.model_validate_json(response.text)
                result = result_obj.model_dump()

            else:
                # Use OpenAI GPT-4o for reasoning
                start_time = time.time()
                completion = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.reasoning_client.chat.completions.create(
                        model=self.reasoning_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                )

                # Track the API call
                duration = time.time() - start_time
                api_tracker.track_openai_call(
                    completion,
                    self.reasoning_model,
                    "OCR→LLM text reasoning analysis",
                    duration
                )

                import json
                result = json.loads(completion.choices[0].message.content)

            return result

        except Exception as e:
            logger.error(f"Reasoning model analysis failed: {e}")
            return {
                "has_error": False,
                "error_description": None,
                "error_location_y": None,
                "correction": None,
                "hint": "Analysis failed, please try again",
                "solution_complete": False,
                "confidence": 0.0,
                "step_analysis": ["Analysis failed"]
            }

    async def detect_errors(self, question_image: Image.Image, solution_image: Image.Image,
                          context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """OCR→LLM approach: GPT-4V extracts text, then reasoning model analyzes"""
        start_time = time.time()

        try:
            # Step 1: Extract text from both images using GPT-4V in parallel
            logger.info("Starting parallel OCR extraction with GPT-4V...")
            question_task = asyncio.create_task(
                self._extract_text_with_gpt4v(question_image, "question")
            )
            solution_task = asyncio.create_task(
                self._extract_text_with_gpt4v(solution_image, "solution")
            )

            question_text, solution_text = await asyncio.gather(question_task, solution_task)
            logger.info("OCR extraction completed")

            # Step 2: Analyze the extracted text with reasoning model
            logger.info(f"Starting reasoning analysis with {self.reasoning_provider}")
            analysis_result = await self._analyze_with_reasoning_model(
                question_text, solution_text, context
            )

            # Step 3: Convert to VisionAnalysisResult format
            steps_data = analysis_result.get("step_analysis", [])
            math_steps = self._convert_steps_to_mathsteps(steps_data)

            vision_result = VisionAnalysisResult(
                ocr_result=OCRResult(
                    extracted_text=solution_text,
                    mathematical_expressions=[],  # Could parse these from text
                    steps=math_steps,
                    has_diagrams=False,  # GPT-4V could detect this
                    confidence=analysis_result.get("confidence", 0.8)
                ),
                error_analysis=ErrorAnalysis(
                    has_error=analysis_result.get("has_error", False),
                    error_description=analysis_result.get("error_description"),
                    error_location_y=analysis_result.get("error_location_y"),
                    correction=analysis_result.get("correction"),
                    hint=analysis_result.get("hint", "Keep working!"),
                    confidence=analysis_result.get("confidence", 0.8)
                ),
                solution_analysis=SolutionAnalysis(
                    solution_complete=analysis_result.get("solution_complete", False),
                    total_steps=len(analysis_result.get("step_analysis", [])),
                    steps_with_errors=1 if analysis_result.get("has_error") else 0,
                    overall_approach_correct=not analysis_result.get("has_error", False)
                ),
                question_has_diagram=False,  # Could enhance to detect
                solution_has_diagram=False,  # Could enhance to detect
                processing_notes=f"OCR→LLM: GPT-4V → {self.reasoning_provider}, processing_time: {time.time() - start_time:.2f}s"
            )

            logger.info(f"OCR→LLM approach completed in {time.time() - start_time:.2f}s")
            return vision_result

        except Exception as e:
            logger.error(f"OCR→LLM approach failed: {e}")
            return self._create_fallback_result()

    def _create_mathstep_from_string(self, text: str, step_number: int) -> MathStep:
        """Convert string to MathStep object"""
        return MathStep(
            step_number=step_number,
            text=text,
            has_error=False,
            error_type=None
        )

    def _convert_steps_to_mathsteps(self, steps: list) -> list[MathStep]:
        """Convert list of strings to list of MathStep objects"""
        if not steps:
            return []

        mathsteps = []
        for i, step in enumerate(steps, 1):
            if isinstance(step, str):
                mathsteps.append(self._create_mathstep_from_string(step, i))
            elif isinstance(step, dict):
                # If it's already a dict, convert to MathStep
                mathsteps.append(MathStep(
                    step_number=step.get('step_number', i),
                    text=step.get('text', str(step)),
                    has_error=step.get('has_error', False),
                    error_type=step.get('error_type')
                ))
            else:
                # Fallback for other types
                mathsteps.append(self._create_mathstep_from_string(str(step), i))
        return mathsteps

    def _create_fallback_result(self) -> VisionAnalysisResult:
        """Create fallback result when approach fails"""
        return VisionAnalysisResult(
            ocr_result=OCRResult(
                extracted_text="OCR extraction failed",
                mathematical_expressions=[],
                steps=[self._create_mathstep_from_string("Failed to extract text", 1)],
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
            processing_notes="OCR→LLM approach failed"
        )


class DirectVLMApproach(BaseErrorDetectionApproach):
    """Direct VLM: GPT-4V or Gemini-2.5-Flash (single call)"""

    def __init__(self):
        super().__init__()

        # Choose VLM provider based on configuration
        vlm_provider = self.settings.model_provider
        if vlm_provider == "auto":
            vlm_provider = "openai" if self.settings.openai_api_key else "gemini"

        self.vlm_provider = vlm_provider

        if vlm_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
                self.model = self.settings.openai_vision_model
            except ImportError:
                raise ImportError("OpenAI library required for Direct VLM approach")
        else:  # gemini
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.settings.gemini_api_key)
            self.genai_types = types
            self.model = self.settings.gemini_model

        logger.info(f"Initialized Direct VLM approach with {vlm_provider}")

    def get_approach_name(self) -> str:
        return f"direct_vlm_{self.vlm_provider}"

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL"""
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for Gemini"""
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def _create_mathstep_from_string(self, text: str, step_number: int) -> MathStep:
        """Convert string to MathStep object"""
        return MathStep(
            step_number=step_number,
            text=text,
            has_error=False,
            error_type=None
        )

    def _convert_steps_to_mathsteps(self, steps: list) -> list[MathStep]:
        """Convert list of strings to list of MathStep objects"""
        if not steps:
            return []

        mathsteps = []
        for i, step in enumerate(steps, 1):
            if isinstance(step, str):
                mathsteps.append(self._create_mathstep_from_string(step, i))
            elif isinstance(step, dict):
                # If it's already a dict, convert to MathStep
                mathsteps.append(MathStep(
                    step_number=step.get('step_number', i),
                    text=step.get('text', str(step)),
                    has_error=step.get('has_error', False),
                    error_type=step.get('error_type')
                ))
            else:
                # Fallback for other types
                mathsteps.append(self._create_mathstep_from_string(str(step), i))
        return mathsteps

    async def detect_errors(self, question_image: Image.Image, solution_image: Image.Image,
                          context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """Direct VLM approach: Single call to vision-language model"""
        start_time = time.time()

        prompt = """You are an expert mathematics tutor. Analyze the student's handwritten solution by looking at both the question and solution images.

Your task:
1. Extract the question from the question image
2. Extract the student's work from the solution image
3. Identify any mathematical errors in the solution
4. If errors exist, find the FIRST significant error
5. Provide educational feedback with corrections and hints
6. Determine if the solution is complete

Respond in JSON format with these exact fields:
{
    "question_text": "extracted question text",
    "solution_text": "extracted solution text",
    "has_error": true/false,
    "error_description": "string or null",
    "error_location_y": 123.45 (numeric pixel location) or null,
    "correction": "string or null",
    "hint": "helpful hint for student",
    "solution_complete": true/false,
    "question_has_diagram": true/false,
    "solution_has_diagram": true/false,
    "confidence": 0.85 (number between 0.0-1.0),
    "steps": ["step1", "step2", ...]
}

IMPORTANT: error_location_y must be a NUMBER (e.g. 150.0) or null, NOT a string with words."""

        try:
            if self.vlm_provider == "openai":
                # OpenAI GPT-4V approach
                question_url = self._image_to_base64_url(question_image)
                solution_url = self._image_to_base64_url(solution_image)

                start_time = time.time()
                completion = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "text", "text": "Question Image:"},
                                    {"type": "image_url", "image_url": {"url": question_url}},
                                    {"type": "text", "text": "Student's Solution Image:"},
                                    {"type": "image_url", "image_url": {"url": solution_url}}
                                ]
                            }
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                )

                # Track the API call
                duration = time.time() - start_time
                api_tracker.track_openai_call(
                    completion,
                    self.model,
                    "Direct VLM vision analysis",
                    duration
                )

                import json
                result = json.loads(completion.choices[0].message.content)

            else:
                # Gemini approach
                question_b64 = self._image_to_base64(question_image)
                solution_b64 = self._image_to_base64(solution_image)

                start_time = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=[
                            self.genai_types.Content(parts=[
                                self.genai_types.Part.from_text(text=prompt),
                                self.genai_types.Part.from_text(text="Question Image:"),
                                self.genai_types.Part.from_bytes(
                                    data=base64.b64decode(question_b64),
                                    mime_type="image/jpeg"
                                ),
                                self.genai_types.Part.from_text(text="Student's Solution Image:"),
                                self.genai_types.Part.from_bytes(
                                    data=base64.b64decode(solution_b64),
                                    mime_type="image/jpeg"
                                )
                            ])
                        ],
                        config=self.genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=DirectVLMResult,
                            temperature=0.1
                        )
                    )
                )

                # Track the API call (estimated tokens for Gemini with vision)
                duration = time.time() - start_time
                api_tracker.track_gemini_call(
                    response,
                    self.model,
                    "Direct VLM Gemini vision analysis",
                    duration,
                    estimated_input_tokens=len(prompt.split()) * 1.3 + 2000,  # +2000 for images
                    estimated_output_tokens=len(response.text.split()) * 1.3
                )

                result_obj = DirectVLMResult.model_validate_json(response.text)
                result = result_obj.model_dump()

            # Convert to VisionAnalysisResult format
            steps_data = result.get("steps", [])
            math_steps = self._convert_steps_to_mathsteps(steps_data)

            vision_result = VisionAnalysisResult(
                ocr_result=OCRResult(
                    extracted_text=result.get("solution_text", ""),
                    mathematical_expressions=[],
                    steps=math_steps,
                    has_diagrams=result.get("solution_has_diagram", False),
                    confidence=result.get("confidence", 0.8)
                ),
                error_analysis=ErrorAnalysis(
                    has_error=result.get("has_error", False),
                    error_description=result.get("error_description"),
                    error_location_y=result.get("error_location_y"),
                    correction=result.get("correction"),
                    hint=result.get("hint", "Keep working!"),
                    confidence=result.get("confidence", 0.8)
                ),
                solution_analysis=SolutionAnalysis(
                    solution_complete=result.get("solution_complete", False),
                    total_steps=len(result.get("steps", [])),
                    steps_with_errors=1 if result.get("has_error") else 0,
                    overall_approach_correct=not result.get("has_error", False)
                ),
                question_has_diagram=result.get("question_has_diagram", False),
                solution_has_diagram=result.get("solution_has_diagram", False),
                processing_notes=f"Direct VLM: {self.vlm_provider}, processing_time: {time.time() - start_time:.2f}s"
            )

            logger.info(f"Direct VLM approach completed in {time.time() - start_time:.2f}s")
            return vision_result

        except Exception as e:
            logger.error(f"Direct VLM approach failed: {e}")
            return self._create_fallback_result()

    def _create_fallback_result(self) -> VisionAnalysisResult:
        """Create fallback result when approach fails"""
        return VisionAnalysisResult(
            ocr_result=OCRResult(
                extracted_text="Direct VLM analysis failed",
                mathematical_expressions=[],
                steps=[self._create_mathstep_from_string("Failed to analyze images", 1)],
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
            processing_notes=f"Direct VLM ({self.vlm_provider}) approach failed"
        )


class HybridApproach(BaseErrorDetectionApproach):
    """Hybrid: Run both OCR→LLM and Direct VLM, ensemble results"""

    def __init__(self):
        super().__init__()
        self.ocr_llm_approach = OCRToLLMApproach()
        self.direct_vlm_approach = DirectVLMApproach()
        logger.info("Initialized Hybrid approach (OCR→LLM + Direct VLM)")

    def get_approach_name(self) -> str:
        return "hybrid_ocr_llm_plus_direct_vlm"

    def _create_mathstep_from_string(self, text: str, step_number: int) -> MathStep:
        """Convert string to MathStep object"""
        return MathStep(
            step_number=step_number,
            text=text,
            has_error=False,
            error_type=None
        )

    def _convert_steps_to_mathsteps(self, steps: list) -> list[MathStep]:
        """Convert list of strings to list of MathStep objects"""
        if not steps:
            return []

        mathsteps = []
        for i, step in enumerate(steps, 1):
            if isinstance(step, str):
                mathsteps.append(self._create_mathstep_from_string(step, i))
            elif isinstance(step, dict):
                # If it's already a dict, convert to MathStep
                mathsteps.append(MathStep(
                    step_number=step.get('step_number', i),
                    text=step.get('text', str(step)),
                    has_error=step.get('has_error', False),
                    error_type=step.get('error_type')
                ))
            else:
                # Fallback for other types
                mathsteps.append(self._create_mathstep_from_string(str(step), i))
        return mathsteps

    def _ensemble_results(self, ocr_result: VisionAnalysisResult,
                         vlm_result: VisionAnalysisResult) -> VisionAnalysisResult:
        """Ensemble results from OCR→LLM and Direct VLM approaches"""

        # Get confidence scores
        ocr_confidence = ocr_result.error_analysis.confidence
        vlm_confidence = vlm_result.error_analysis.confidence

        logger.info(f"Ensembling results: OCR→LLM confidence={ocr_confidence:.3f}, Direct VLM confidence={vlm_confidence:.3f}")

        # Choose primary result based on higher confidence
        if ocr_confidence >= vlm_confidence:
            primary_result = ocr_result
            secondary_result = vlm_result
            primary_name = "OCR→LLM"
            secondary_name = "Direct VLM"
        else:
            primary_result = vlm_result
            secondary_result = ocr_result
            primary_name = "Direct VLM"
            secondary_name = "OCR→LLM"

        # If both approaches agree on error detection, use the higher confidence result
        both_detect_error = (ocr_result.error_analysis.has_error and vlm_result.error_analysis.has_error)
        neither_detect_error = (not ocr_result.error_analysis.has_error and not vlm_result.error_analysis.has_error)

        if both_detect_error or neither_detect_error:
            logger.info(f"Both approaches agree (error={both_detect_error}), using {primary_name}")
            ensemble_result = primary_result
        else:
            # Approaches disagree - use higher confidence, but note the disagreement
            logger.info(f"Approaches disagree on error detection, using {primary_name} (higher confidence)")
            ensemble_result = primary_result

        # Combine the best aspects from both
        # Use the more detailed OCR text if available
        if len(ocr_result.ocr_result.extracted_text) > len(vlm_result.ocr_result.extracted_text):
            ensemble_result.ocr_result.extracted_text = ocr_result.ocr_result.extracted_text

        # Average confidence scores for final confidence
        final_confidence = (ocr_confidence + vlm_confidence) / 2
        ensemble_result.error_analysis.confidence = final_confidence
        ensemble_result.ocr_result.confidence = final_confidence

        # Update processing notes
        ensemble_result.processing_notes = (
            f"Hybrid ensemble: {primary_name}({ocr_confidence:.3f}) + {secondary_name}({vlm_confidence:.3f}) "
            f"→ final_confidence={final_confidence:.3f}"
        )

        return ensemble_result

    async def detect_errors(self, question_image: Image.Image, solution_image: Image.Image,
                          context: Optional[Dict[str, Any]] = None) -> VisionAnalysisResult:
        """Hybrid approach: Run both approaches and ensemble results"""
        start_time = time.time()

        try:
            logger.info("Starting parallel execution of OCR→LLM and Direct VLM approaches...")

            # Run both approaches in parallel
            ocr_task = asyncio.create_task(
                self.ocr_llm_approach.detect_errors(question_image, solution_image, context)
            )
            vlm_task = asyncio.create_task(
                self.direct_vlm_approach.detect_errors(question_image, solution_image, context)
            )

            # Wait for both with timeout
            ocr_result, vlm_result = await asyncio.wait_for(
                asyncio.gather(ocr_task, vlm_task),
                timeout=90.0  # Generous timeout for both approaches
            )

            logger.info("Both approaches completed, starting ensemble...")

            # Ensemble the results
            ensemble_result = self._ensemble_results(ocr_result, vlm_result)

            processing_time = time.time() - start_time
            ensemble_result.processing_notes += f", total_time: {processing_time:.2f}s"

            logger.info(f"Hybrid approach completed in {processing_time:.2f}s")
            return ensemble_result

        except asyncio.TimeoutError:
            logger.error("Hybrid approach timed out")
            return self._create_fallback_result()
        except Exception as e:
            logger.error(f"Hybrid approach failed: {e}")
            return self._create_fallback_result()

    def _create_fallback_result(self) -> VisionAnalysisResult:
        """Create fallback result when hybrid approach fails"""
        return VisionAnalysisResult(
            ocr_result=OCRResult(
                extracted_text="Hybrid approach failed",
                mathematical_expressions=[],
                steps=[self._create_mathstep_from_string("Both approaches failed", 1)],
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
            processing_notes="Hybrid approach failed"
        )


def create_approach(approach_name: str) -> BaseErrorDetectionApproach:
    """Factory function to create the appropriate approach"""
    if approach_name == "ocr_llm":
        return OCRToLLMApproach()
    elif approach_name == "vlm_direct":
        return DirectVLMApproach()
    elif approach_name == "hybrid":
        return HybridApproach()
    else:
        raise ValueError(f"Unknown approach: {approach_name}")