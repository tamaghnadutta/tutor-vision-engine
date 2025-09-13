"""
Configuration settings for the Error Detection API
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_key: str = Field(..., env="API_KEY")
    environment: str = Field(default="development", env="ENVIRONMENT")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")

    # Gemini Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    gemini_thinking_budget: int = Field(default=0, env="GEMINI_THINKING_BUDGET")

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-2024-08-06", env="OPENAI_MODEL")
    openai_vision_model: str = Field(default="gpt-4o", env="OPENAI_VISION_MODEL")  # For OCR tasks

    # Model Provider Selection
    model_provider: str = Field(default="auto", env="MODEL_PROVIDER")  # "gemini", "openai", or "auto"

    # Database Configuration
    database_url: str = Field(default="sqlite:///./data/requests.db", env="DATABASE_URL")

    # Observability
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    structured_logs: bool = Field(default=True, env="STRUCTURED_LOGS")

    # Performance Settings
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    max_image_size_mb: int = Field(default=5, env="MAX_IMAGE_SIZE_MB")

    # Evaluation Settings
    eval_dataset_path: str = Field(default="./data/eval_dataset.json", env="EVAL_DATASET_PATH")
    eval_results_path: str = Field(default="./data/eval_results/", env="EVAL_RESULTS_PATH")

    # Approach Selection (as per assignment requirements)
    error_detection_approach: str = Field(default="hybrid", env="ERROR_DETECTION_APPROACH")  # ocr_llm, vlm_direct, hybrid

    # OCR→LLM Configuration
    ocr_provider: str = Field(default="gpt4v", env="OCR_PROVIDER")  # Which model to use for OCR
    reasoning_provider: str = Field(default="auto", env="REASONING_PROVIDER")  # Which model to use for reasoning from OCR text

    model_config = {
        'protected_namespaces': ('settings_',),
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()