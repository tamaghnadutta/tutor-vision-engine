"""
Authentication utilities
"""

import hmac
import hashlib
from typing import Optional

import logging

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def verify_api_key(provided_key: str) -> bool:
    """Verify API key using constant-time comparison"""
    settings = get_settings()
    expected_key = settings.api_key

    if not expected_key:
        logger.warning("No API key configured")
        return False

    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(provided_key.encode(), expected_key.encode())


def hash_user_id(user_id: str) -> str:
    """Hash user ID for privacy-preserving logging"""
    if not user_id:
        return "anonymous"

    # Simple hash for user ID anonymization
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]