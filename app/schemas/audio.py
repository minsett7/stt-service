from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AudioQualityResult(BaseModel):
    quality: Literal["good", "acceptable", "bad"]
    duration_seconds: float
    speech_duration_seconds: float
    speech_ratio: float
    sample_rate: int
    channels: int
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipping_detected: bool
    clipping_ratio: float
    volume_too_low: bool
    volume_too_high: bool
    speech_detected: bool


class AudioProcessingTimings(BaseModel):
    decode_ms: int = 0
    filter_ms: int = 0
    noise_suppression_ms: int = 0
    normalization_ms: int = 0
    vad_ms: int = 0
    quality_analysis_ms: int = 0
    audio_preprocessing_ms: int = 0
    asr_ms: int = 0
    correction_ms: int = 0
    total_processing_ms: int = 0
