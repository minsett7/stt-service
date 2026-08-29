from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.audio import AudioProcessingTimings, AudioQualityResult


class ASRResult(BaseModel):
    text: str
    confidence: float | None = None
    language: str | None = "my"


class CorrectionItem(BaseModel):
    from_text: str = Field(alias="from")
    to: str
    type: str
    confidence: float | None = Field(default=None, ge=0, le=1)

    model_config = {"populate_by_name": True}


class CorrectionResult(BaseModel):
    corrected_transcript: str
    corrections: list[CorrectionItem] = []
    overall_confidence: float | None = Field(default=None, ge=0, le=1)
    uncertain_terms: list[str] = []


class ValidationResult(BaseModel):
    passed: bool
    warnings: list[str] = []
    change_ratio: float = 0.0


class TranscriptionData(BaseModel):
    raw_transcript: str
    corrected_transcript: str
    final_transcript: str
    used_correction: bool
    correction_provider: str | None = None
    validation_passed: bool
    correction_confidence: float | None = None
    corrections: list[CorrectionItem] = []
    warnings: list[str] = []
    processing_time_ms: int
    audio_quality: AudioQualityResult | None = None
    timings: AudioProcessingTimings | None = None


class TranscriptionResponse(BaseModel):
    success: bool = True
    data: TranscriptionData


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    asr_ready: bool
    correction_provider: str | None
    audio_preprocessing_ready: bool
    ffmpeg_ready: bool
    vad_ready: bool
