"""
Pydantic schemas for API request/response models
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl, Field, field_validator


class BoundingBox(BaseModel):
    """Bounding box coordinates for the edited area"""
    minX: float = Field(..., description="Minimum X coordinate")
    maxX: float = Field(..., description="Maximum X coordinate")
    minY: float = Field(..., description="Minimum Y coordinate")
    maxY: float = Field(..., description="Maximum Y coordinate")

    @field_validator('maxX')
    @classmethod
    def validate_x_coords(cls, v, info):
        if info.data.get('minX') is not None and v <= info.data['minX']:
            raise ValueError('maxX must be greater than minX')
        return v

    @field_validator('maxY')
    @classmethod
    def validate_y_coords(cls, v, info):
        if info.data.get('minY') is not None and v <= info.data['minY']:
            raise ValueError('maxY must be greater than minY')
        return v


class ErrorDetectionRequest(BaseModel):
    """Request schema for error detection endpoint"""
    question_url: HttpUrl = Field(..., description="URL to the question image")
    solution_url: HttpUrl = Field(..., description="URL to the student's solution image")
    bounding_box: BoundingBox = Field(..., description="Coordinates of the edited area")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    question_id: Optional[str] = Field(None, description="Optional question identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "question_url": "https://example.com/question_image.png",
                "solution_url": "https://example.com/solution_image.png",
                "bounding_box": {
                    "minX": 316,
                    "maxX": 635,
                    "minY": 48.140625,
                    "maxY": 79.140625
                },
                "user_id": "optional_user_identifier",
                "session_id": "optional_session_identifier",
                "question_id": "optional_question_identifier"
            }
        }


class ErrorDetectionResponse(BaseModel):
    """Response schema for error detection endpoint"""
    job_id: str = Field(..., description="Unique job identifier")
    y: Optional[float] = Field(None, description="Y-coordinate of the error location")
    error: Optional[str] = Field(None, description="Description of the identified error")
    correction: Optional[str] = Field(None, description="Suggested correction for the error")
    hint: Optional[str] = Field(None, description="Helpful hint for the student")
    solution_complete: bool = Field(..., description="Whether the solution is complete")
    contains_diagram: bool = Field(..., description="Whether the solution contains a diagram")
    question_has_diagram: bool = Field(..., description="Whether the question has a diagram")
    solution_has_diagram: bool = Field(..., description="Whether the solution has a diagram")
    llm_used: bool = Field(..., description="Whether an LLM was used in processing")
    solution_lines: Optional[List[str]] = Field(None, description="Extracted solution lines")
    llm_ocr_lines: Optional[List[str]] = Field(None, description="LLM-processed OCR lines")
    confidence: Optional[float] = Field(None, description="Confidence score for the analysis (0-1)", ge=0.0, le=1.0)
    processing_approach: Optional[str] = Field(None, description="Processing approach used (e.g., robust_model_openai)")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "unique_job_identifier",
                "y": 150.5,
                "error": "Incorrect application of quadratic formula",
                "correction": "The discriminant should be b² - 4ac, not b² + 4ac",
                "hint": "Remember: discriminant = b² - 4ac for quadratic equations",
                "solution_complete": False,
                "contains_diagram": True,
                "question_has_diagram": True,
                "solution_has_diagram": False,
                "llm_used": True,
                "solution_lines": ["Step 1: ...", "Step 2: ..."],
                "llm_ocr_lines": ["x = (-b ± √(b² + 4ac)) / 2a"]
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    job_id: Optional[str] = Field(None, description="Job ID if applicable")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: Optional[str] = Field(None, description="Response timestamp")