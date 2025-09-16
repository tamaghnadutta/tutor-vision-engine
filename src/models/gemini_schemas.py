"""
Pydantic schemas for structured outputs from Gemini models
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class MathStep(BaseModel):
    """Individual mathematical step"""
    step_number: int = Field(..., description="Step number (1-based)")
    text: str = Field(..., description="The mathematical expression or text in this step")
    has_error: bool = Field(..., description="Whether this step contains an error")
    error_type: Optional[str] = Field(None, description="Type of error (e.g., 'arithmetic', 'algebraic', 'conceptual')")


class OCRResult(BaseModel):
    """OCR extraction result from Gemini Vision"""
    extracted_text: str = Field(..., description="Full extracted text from the image")
    mathematical_expressions: List[str] = Field(..., description="List of mathematical expressions found")
    steps: List[MathStep] = Field(..., description="Parsed mathematical steps")
    has_diagrams: bool = Field(..., description="Whether the image contains diagrams or graphs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in OCR extraction (0-1)")


class ErrorAnalysis(BaseModel):
    """Error analysis result from Gemini"""
    has_error: bool = Field(..., description="Whether an error was detected")
    error_description: Optional[str] = Field(None, description="Description of the error found")
    error_location_y: Optional[float] = Field(None, description="Y-coordinate of error location in image")
    correction: Optional[str] = Field(None, description="Suggested correction for the error")
    hint: Optional[str] = Field(None, description="Educational hint for the student")
    error_type: Optional[Literal["arithmetic", "algebraic", "conceptual", "incomplete", "unclear"]] = Field(
        None, description="Category of error"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in error detection (0-1)")


class SolutionAnalysis(BaseModel):
    """Complete solution analysis"""
    solution_complete: bool = Field(..., description="Whether the solution is complete")
    total_steps: int = Field(..., description="Total number of steps in the solution")
    steps_with_errors: int = Field(..., description="Number of steps containing errors")
    overall_approach_correct: bool = Field(..., description="Whether the overall approach is correct")
    topic: Optional[str] = Field(None, description="Mathematical topic (e.g., 'algebra', 'trigonometry')")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(None, description="Assessed difficulty level")


class VisionAnalysisResult(BaseModel):
    """Complete vision analysis result combining OCR and error detection"""
    ocr_result: OCRResult = Field(..., description="OCR extraction results")
    error_analysis: ErrorAnalysis = Field(..., description="Error detection analysis")
    solution_analysis: SolutionAnalysis = Field(..., description="Overall solution analysis")
    question_has_diagram: bool = Field(..., description="Whether question image has diagrams")
    solution_has_diagram: bool = Field(..., description="Whether solution image has diagrams")
    processing_notes: Optional[str] = Field(None, description="Any additional processing notes or warnings")


class MathematicsProblemAnalysis(BaseModel):
    """Simplified analysis of a mathematics problem and student solution"""
    problem_type: str = Field(..., description="Type of mathematical problem")
    student_approach: str = Field(..., description="Description of student's approach")
    has_error: bool = Field(..., description="Whether an error was detected")
    error_description: Optional[str] = Field(None, description="Description of the error found")
    error_location_y: Optional[float] = Field(None, description="Y-coordinate of error location in image")
    correction: Optional[str] = Field(None, description="Suggested correction for the error")
    hint: Optional[str] = Field(None, description="Educational hint for the student")
    solution_complete: bool = Field(..., description="Whether the solution is complete")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis (0-1)")
    recommendations: List[str] = Field(..., description="Specific recommendations for improvement")

    # Create compatibility properties for the old nested structure
    @property
    def error_analysis(self) -> "ErrorAnalysis":
        """Compatibility property to mimic old nested structure"""
        return ErrorAnalysis(
            has_error=self.has_error,
            error_description=self.error_description,
            error_location_y=self.error_location_y,
            correction=self.correction,
            hint=self.hint,
            confidence=self.confidence
        )

    @property
    def solution_quality(self) -> "SolutionAnalysis":
        """Compatibility property to mimic old nested structure"""
        return SolutionAnalysis(
            solution_complete=self.solution_complete,
            total_steps=1,  # Simplified
            steps_with_errors=1 if self.has_error else 0,
            overall_approach_correct=not self.has_error
        )


class DiagramAnalysis(BaseModel):
    """Analysis of diagrams or visual elements in mathematical content"""
    has_diagram: bool = Field(..., description="Whether a diagram is present")
    diagram_type: Optional[str] = Field(None, description="Type of diagram (e.g., 'graph', 'geometric figure', 'tree')")
    diagram_relevance: Optional[Literal["essential", "helpful", "decorative"]] = Field(
        None, description="How relevant the diagram is to solving the problem"
    )
    diagram_quality: Optional[Literal["clear", "unclear", "incomplete"]] = Field(
        None, description="Quality of the diagram"
    )


class GeminiErrorDetectionResult(BaseModel):
    """Final structured result for error detection API"""
    y: Optional[float] = Field(None, description="Y-coordinate of error location")
    error: Optional[str] = Field(None, description="Error description")
    correction: Optional[str] = Field(None, description="Suggested correction")
    hint: Optional[str] = Field(None, description="Educational hint")
    solution_complete: bool = Field(..., description="Whether solution is complete")
    contains_diagram: bool = Field(..., description="Whether any diagram is present")
    question_has_diagram: bool = Field(..., description="Whether question has diagram")
    solution_has_diagram: bool = Field(..., description="Whether solution has diagram")
    llm_used: bool = Field(True, description="Whether LLM was used (always true for Gemini)")
    solution_lines: List[str] = Field(..., description="Extracted solution lines")
    llm_ocr_lines: List[str] = Field(..., description="LLM-processed OCR lines")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in analysis")
    processing_approach: str = Field("gemini_2.5_flash", description="Processing approach used")


class QuestionAnalysis(BaseModel):
    """Analysis of the question/problem image"""
    question_text: str = Field(..., description="Extracted question text")
    problem_type: str = Field(..., description="Type of mathematical problem")
    expected_approach: str = Field(..., description="Expected solution approach")
    key_concepts: List[str] = Field(..., description="Key mathematical concepts involved")
    difficulty_level: Literal["easy", "medium", "hard"] = Field(..., description="Assessed difficulty")
    has_multiple_choice: bool = Field(..., description="Whether question has multiple choice options")
    choice_options: List[str] = Field(default_factory=list, description="Multiple choice options if present")


class OCRLLMReasoningResult(BaseModel):
    """Result from OCR→LLM reasoning analysis"""
    has_error: bool = Field(..., description="Whether an error was detected")
    error_description: Optional[str] = Field(None, description="Description of the error found")
    error_location_y: Optional[float] = Field(None, description="Y-coordinate of error location in image (numeric pixels)")
    correction: Optional[str] = Field(None, description="Suggested correction for the error")
    hint: str = Field(..., description="Educational hint for the student")
    solution_complete: bool = Field(..., description="Whether the solution is complete")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis (0-1)")
    step_analysis: List[str] = Field(default_factory=list, description="Analysis of each step")


class DirectVLMResult(BaseModel):
    """Result from Direct VLM analysis"""
    question_text: str = Field(..., description="Extracted question text")
    solution_text: str = Field(..., description="Extracted solution text")
    has_error: bool = Field(..., description="Whether an error was detected")
    error_description: Optional[str] = Field(None, description="Description of the error found")
    error_location_y: Optional[float] = Field(None, description="Y-coordinate of error location in image (numeric pixels)")
    correction: Optional[str] = Field(None, description="Suggested correction for the error")
    hint: str = Field(..., description="Educational hint for the student")
    solution_complete: bool = Field(..., description="Whether the solution is complete")
    question_has_diagram: bool = Field(..., description="Whether question image has diagrams")
    solution_has_diagram: bool = Field(..., description="Whether solution image has diagrams")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis (0-1)")
    steps: List[str] = Field(default_factory=list, description="List of solution steps")