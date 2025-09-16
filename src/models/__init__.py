"""
Models module for error detection using Gemini 2.5 Flash
"""

from .error_detector import ErrorDetector
from .gemini_processor import GeminiProcessor
from .gemini_schemas import (
    GeminiErrorDetectionResult, OCRResult, ErrorAnalysis,
    VisionAnalysisResult, QuestionAnalysis, MathematicsProblemAnalysis
)

__all__ = [
    'ErrorDetector',
    'GeminiProcessor',
    'GeminiErrorDetectionResult',
    'OCRResult',
    'ErrorAnalysis',
    'VisionAnalysisResult',
    'QuestionAnalysis',
    'MathematicsProblemAnalysis'
]