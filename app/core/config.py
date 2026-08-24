from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "burmese-stt-service"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"
    log_transcripts: bool = False

    max_audio_size_mb: int = Field(default=20, ge=1)
    supported_audio_formats: Annotated[list[str], NoDecode] = ["wav", "mp3", "m4a", "webm"]
    normalize_audio: bool = True
    ffmpeg_binary: str = "ffmpeg"
    normalized_sample_rate: int = Field(default=16000, ge=8000)

    asr_model: str = "chuuhtetnaing/whisper-large-v3-myanmar"
    asr_device: str = "auto"
    asr_dtype: str = "auto"

    correction_enabled: bool = True
    correction_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    correction_timeout_seconds: float = Field(default=10, gt=0)
    request_timeout_seconds: float = Field(default=900, gt=0)
    max_correction_change_ratio: float = Field(default=0.45, ge=0, le=1)

    @field_validator("supported_audio_formats", mode="before")
    @classmethod
    def split_formats(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]
        return value

    @field_validator("asr_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("ASR_DEVICE must be auto, cpu, or cuda")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
