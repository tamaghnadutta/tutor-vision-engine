"""
Error Detection API Routes
"""

import time
import uuid
from typing import Optional

import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.api.schemas import ErrorDetectionRequest, ErrorDetectionResponse
from src.config.settings import get_settings
from src.models.error_detector import ErrorDetector
from src.utils.auth import verify_api_key
from src.utils.persistence import save_request_response
from src.utils.metrics import REQUEST_COUNT, REQUEST_DURATION

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key"""
    if not verify_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@router.post("/detect-error", response_model=ErrorDetectionResponse)
async def detect_error(
    request: ErrorDetectionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
) -> ErrorDetectionResponse:
    """
    Detect errors in student's handwritten mathematical solution.

    Takes a question image and student's handwritten attempt, analyzes the work,
    and returns step-level error detection with corrections and hints.
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())

    logger.info(
        f"Processing error detection request - job_id={job_id}, question_url={request.question_url}, "
        f"solution_url={request.solution_url}, user_id={request.user_id}, session_id={request.session_id}, "
        f"question_id={request.question_id}"
    )

    try:
        # Initialize error detector
        detector = ErrorDetector()

        # Process the request
        result = await detector.detect_errors(
            question_url=str(request.question_url),
            solution_url=str(request.solution_url),
            bounding_box=request.bounding_box.model_dump() if request.bounding_box else None,
            context={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "question_id": request.question_id,
            }
        )

        # Create response
        response = ErrorDetectionResponse(
            job_id=job_id,
            **result
        )

        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.labels(method="POST", endpoint="/detect-error").observe(duration)
        REQUEST_COUNT.labels(method="POST", endpoint="/detect-error", status="200").inc()

        # Save request/response for auditing (background task)
        background_tasks.add_task(
            save_request_response,
            job_id,
            request.model_dump(mode='json'),
            response.model_dump(mode='json'),
            duration
        )

        logger.info(
            f"Error detection completed successfully - job_id={job_id}, duration={duration:.2f}s, "
            f"has_error={bool(result.get('error'))}, llm_used={result.get('llm_used', False)}"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        REQUEST_COUNT.labels(method="POST", endpoint="/detect-error", status="500").inc()

        logger.error(
            f"Error detection failed - job_id={job_id}, duration={duration:.2f}s, error={str(e)}",
            exc_info=True
        )

        # Return partial results if possible
        if hasattr(e, 'partial_result') and e.partial_result:
            response = ErrorDetectionResponse(
                job_id=job_id,
                **e.partial_result
            )
            background_tasks.add_task(
                save_request_response,
                job_id,
                request.model_dump(mode='json'),
                response.model_dump(mode='json'),
                duration,
                error=str(e)
            )
            return response

        raise HTTPException(
            status_code=500,
            detail="Error processing request. Please try again."
        )


@router.get("/detect-error/{job_id}")
async def get_job_status(
    job_id: str,
    api_key: str = Depends(get_api_key)
) -> dict:
    """Get the status of a previous error detection job"""
    # TODO: Implement job status lookup from persistence layer
    # For now, return a simple response
    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Job status lookup not yet implemented"
    }