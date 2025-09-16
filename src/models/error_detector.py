"""
Error detection model using Gemini 2.5 Flash for all processing
"""

import asyncio
import time
import io
from typing import Dict, Any, Optional
from dataclasses import dataclass

import logging
from PIL import Image
import httpx

from src.models.gemini_processor import GeminiProcessor
from src.models.model_router import MultiModelProcessor
from src.models.robust_model_router import create_robust_processor
from src.models.error_detection_approaches import create_approach
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ErrorDetectionResult:
    """Result from error detection process"""
    y: Optional[float] = None
    error: Optional[str] = None
    correction: Optional[str] = None
    hint: Optional[str] = None
    solution_complete: bool = False
    contains_diagram: bool = False
    question_has_diagram: bool = False
    solution_has_diagram: bool = False
    llm_used: bool = False
    solution_lines: Optional[list] = None
    llm_ocr_lines: Optional[list] = None


class ErrorDetector:
    """Error detection model supporting three distinct approaches as per assignment requirements"""

    def __init__(self, approach: str = "auto"):
        """
        Initialize error detector with the specified approach.

        Args:
            approach: "ocr_llm", "vlm_direct", "hybrid", or "auto" (uses settings)
        """
        self.settings = get_settings()

        # Use settings-based approach selection if approach is "auto"
        if approach == "auto":
            approach = self.settings.error_detection_approach

        # Validate approach
        valid_approaches = ["ocr_llm", "vlm_direct", "hybrid"]
        if approach not in valid_approaches:
            logger.warning(f"Invalid approach '{approach}', defaulting to 'hybrid'")
            approach = "hybrid"

        self.approach = approach

        # Initialize the approach-specific processor
        self.approach_processor = create_approach(approach)

        # Keep legacy processors for compatibility methods only
        self.gemini_processor = GeminiProcessor()
        self.robust_processor = create_robust_processor("auto")  # Fallback

        logger.info(f"Initialized ErrorDetector with '{approach}' approach")

    async def detect_errors(self,
                          question_url: str,
                          solution_url: str,
                          bounding_box: Optional[Dict[str, float]] = None,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect errors in student's solution using Gemini 2.5 Flash.

        Args:
            question_url: URL to question image
            solution_url: URL to solution image
            bounding_box: Optional bounding box coordinates (x, y, width, height) in normalized coordinates [0-1]
            context: Additional context (user_id, session_id, etc.)

        Returns:
            Dictionary with error detection results
        """
        start_time = time.time()

        try:
            # Download images
            question_image, solution_image = await self._download_images(question_url, solution_url)

            # Crop solution image to bounding box if provided
            if bounding_box:
                solution_image = self._crop_to_bounding_box(solution_image, bounding_box)
                logger.info(f"Cropped solution image to bounding box: {bounding_box}")

            # Use the selected approach for error detection
            vision_result = await self.approach_processor.detect_errors(
                question_image=question_image,
                solution_image=solution_image,
                context=context
            )

            # Convert VisionAnalysisResult Pydantic model to dict using built-in method
            vision_dict = vision_result.model_dump()

            # Convert to expected API format
            result_dict = {
                "y": vision_dict["error_analysis"].get("error_location_y"),
                "error": vision_dict["error_analysis"].get("error_description"),
                "correction": vision_dict["error_analysis"].get("correction"),
                "hint": vision_dict["error_analysis"].get("hint"),
                "solution_complete": vision_dict["solution_analysis"].get("solution_complete", False),
                "contains_diagram": (vision_dict.get("question_has_diagram", False) or vision_dict.get("solution_has_diagram", False)),
                "question_has_diagram": vision_dict.get("question_has_diagram", False),
                "solution_has_diagram": vision_dict.get("solution_has_diagram", False),
                "llm_used": True,
                "solution_lines": vision_dict["ocr_result"].get("extracted_text", "").split('\n'),
                "llm_ocr_lines": [step.get("text", "") for step in vision_dict["ocr_result"].get("steps", [])],
                "confidence": min(
                    vision_dict["ocr_result"].get("confidence", 0.8),
                    vision_dict["error_analysis"].get("confidence", 0.8)
                ),
                "processing_approach": self.approach_processor.get_approach_name()
            }

            # Add processing time
            result_dict["processing_time"] = time.time() - start_time

            has_error = bool(result_dict.get("error"))
            confidence = result_dict.get("confidence", 0.0)
            logger.info(f"Error detection completed using '{self.approach}': has_error={has_error}, confidence={confidence:.3f}, duration={result_dict['processing_time']:.2f}s")

            return result_dict

        except Exception as e:
            logger.error(f"Error detection failed: {str(e)}", exc_info=True)
            return self._create_fallback_result(str(e))

    def detect_errors_sync(self,
                          question_url: str,
                          solution_url: str,
                          bounding_box: Optional[Dict[str, float]] = None,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous wrapper for detect_errors"""
        return asyncio.run(self.detect_errors(question_url, solution_url, bounding_box, context))

    async def _download_images(self, question_url: str, solution_url: str) -> tuple[Image.Image, Image.Image]:
        """Download and validate images with parallel downloading"""

        async def download_single_image(url: str) -> Image.Image:
            """Download a single image"""
            if url.startswith("file://"):
                path = url.replace("file://", "")
                return Image.open(path)
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url)
                    if response.status_code != 200:
                        raise ValueError(f"Failed to download image {url}: {response.status_code}")
                    return Image.open(io.BytesIO(response.content))

        # PARALLEL OPTIMIZATION: Download both images concurrently
        question_task = asyncio.create_task(download_single_image(question_url))
        solution_task = asyncio.create_task(download_single_image(solution_url))

        # Wait for both downloads to complete
        question_image, solution_image = await asyncio.gather(question_task, solution_task)

        # Validate and resize if needed
        question_image = self._validate_and_resize_image(question_image)
        solution_image = self._validate_and_resize_image(solution_image)

        return question_image, solution_image

    def _validate_and_resize_image(self, image: Image.Image) -> Image.Image:
        """Validate and resize image if needed"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large (Gemini limits)
        max_dimension = 2048
        if image.width > max_dimension or image.height > max_dimension:
            logger.info(f"Resizing image from {image.width}x{image.height}")
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        return image

    def _crop_to_bounding_box(self, image: Image.Image, bounding_box: Dict[str, float]) -> Image.Image:
        """
        Crop image to the specified bounding box.

        Args:
            image: PIL Image to crop
            bounding_box: Dict with keys 'minX', 'maxX', 'minY', 'maxY' (from BoundingBox schema)
                         These are expected to be in pixel coordinates relative to original image

        Returns:
            Cropped PIL Image
        """
        img_width, img_height = image.size

        # Extract coordinates from bounding box
        min_x = bounding_box.get('minX', 0)
        max_x = bounding_box.get('maxX', img_width)
        min_y = bounding_box.get('minY', 0)
        max_y = bounding_box.get('maxY', img_height)

        # Convert to integers and ensure they're within image bounds
        left = max(0, min(int(min_x), img_width))
        top = max(0, min(int(min_y), img_height))
        right = max(left, min(int(max_x), img_width))
        bottom = max(top, min(int(max_y), img_height))

        # Validate crop area
        if right <= left or bottom <= top:
            logger.warning(f"Invalid crop area: left={left}, top={top}, right={right}, bottom={bottom}")
            return image

        logger.info(f"Cropping image from {img_width}x{img_height} to region ({left},{top})->({right},{bottom})")

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

    def _create_fallback_result(self, error_message: str) -> Dict[str, Any]:
        """Create fallback result when processing fails"""
        return {
            "y": None,
            "error": None,
            "correction": None,
            "hint": "Unable to process solution. Please try again with clearer images.",
            "solution_complete": False,
            "contains_diagram": False,
            "question_has_diagram": False,
            "solution_has_diagram": False,
            "llm_used": True,
            "solution_lines": ["Processing failed"],
            "llm_ocr_lines": ["Error during processing"],
            "processing_time": 0.0,
            "confidence": 0.0,
            "processing_approach": f"{self.approach}_fallback",
            "_error": error_message
        }

    # Compatibility methods (for evaluation framework)
    async def extract_text(self, image: Image.Image) -> str:
        """Extract text from image (compatibility method)"""
        try:
            ocr_result = await self.gemini_processor.extract_text_from_image(image)
            return ocr_result.extracted_text
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return "Text extraction failed"

    async def analyze_solution(self, question_text: str, solution_text: str,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze solution using text (compatibility method)"""
        try:
            analysis = await self.gemini_processor.analyze_math_problem(
                question_text, solution_text, context
            )

            return {
                "y": analysis.first_error_step * 25.0 if analysis.first_error_step else None,  # Rough estimate
                "error": analysis.error_analysis.error_description,
                "correction": analysis.error_analysis.correction,
                "hint": analysis.error_analysis.hint,
                "solution_complete": analysis.solution_quality.solution_complete
            }
        except Exception as e:
            logger.error(f"Solution analysis failed: {e}")
            return {
                "y": None,
                "error": None,
                "correction": None,
                "hint": "Analysis failed",
                "solution_complete": False
            }