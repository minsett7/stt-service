import asyncio
import logging

from fastapi import APIRouter, File, Request, UploadFile

from app.core.exceptions import RequestTimeoutError
from app.schemas.transcription import TranscriptionResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["transcription"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: Request, file: UploadFile = File(...)) -> TranscriptionResponse:
    logger.info("transcription_request_started content_type=%s", file.content_type)
    try:
        data = await asyncio.wait_for(
            request.app.state.transcription_service.transcribe(file),
            timeout=request.app.state.settings.request_timeout_seconds,
        )
    except TimeoutError as exc:
        raise RequestTimeoutError from exc
    logger.info("transcription_request_completed duration_ms=%d", data.processing_time_ms)
    return TranscriptionResponse(data=data)
