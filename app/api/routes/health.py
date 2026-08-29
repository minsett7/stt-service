from fastapi import APIRouter, Request

from app.schemas.transcription import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    service = request.app.state.transcription_service
    return HealthResponse(
        status="healthy", service=settings.app_name, version=settings.app_version,
        asr_ready=service.asr.ready,
        correction_provider=service.correction.provider_name if settings.correction_enabled else None,
        audio_preprocessing_ready=service.preprocessor is not None and service.preprocessor.ffmpeg_ready,
        ffmpeg_ready=service.preprocessor is not None and service.preprocessor.ffmpeg_ready,
        vad_ready=service.preprocessor is not None,
    )
