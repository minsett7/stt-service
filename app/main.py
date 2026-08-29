from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.transcribe import router as transcribe_router
from app.core.config import get_settings
from app.core.exceptions import AudioValidationError, RequestTimeoutError, TranscriptionError
from app.core.logging import configure_logging
from app.schemas.common import ErrorDetail, ErrorResponse
from app.services.audio_service import AudioService
from app.services.audio_preprocessing_service import AudioPreprocessingService
from app.services.audio_quality_service import AudioQualityService
from app.services.asr_service import ASRService
from app.services.correction_service import CorrectionService
from app.services.transcription_service import TranscriptionService
from app.services.validation_service import TranscriptValidationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    audio_preprocessor = AudioPreprocessingService(settings, AudioQualityService(settings))
    app.state.transcription_service = TranscriptionService(
        AudioService(settings), ASRService(settings), CorrectionService(settings), TranscriptValidationService(settings), audio_preprocessor
    )
    logging.getLogger(__name__).info("service_started name=%s ffmpeg_ready=%s", settings.app_name, audio_preprocessor.ffmpeg_ready)
    yield


app = FastAPI(title="Burmese STT Service", version="1.0.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(transcribe_router)


@app.exception_handler(AudioValidationError)
async def handle_audio_validation_error(_: Request, exc: AudioValidationError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump())


@app.exception_handler(TranscriptionError)
async def handle_transcription_error(_: Request, exc: TranscriptionError) -> JSONResponse:
    logging.getLogger(__name__).error("transcription_failed")
    return JSONResponse(status_code=502, content=ErrorResponse(error=ErrorDetail(code="transcription_failed", message="Unable to transcribe the provided audio.")).model_dump())


@app.exception_handler(RequestTimeoutError)
async def handle_request_timeout(_: Request, exc: RequestTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content=ErrorResponse(error=ErrorDetail(code="transcription_timeout", message="Transcription request timed out.")).model_dump())
