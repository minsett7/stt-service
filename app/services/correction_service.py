from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import CorrectionUnavailable
from app.providers import GeminiCorrectionProvider, LocalCorrectionProvider, TranscriptCorrectionProvider
from app.schemas.transcription import CorrectionResult


class CorrectionService:
    def __init__(self, settings: Settings, provider: TranscriptCorrectionProvider | None = None) -> None:
        self.settings = settings
        self.provider = provider or self._create_provider()

    def _create_provider(self) -> TranscriptCorrectionProvider:
        if self.settings.correction_provider == "gemini":
            return GeminiCorrectionProvider(self.settings)
        if self.settings.correction_provider == "local":
            return LocalCorrectionProvider()
        raise ValueError(f"Unsupported correction provider: {self.settings.correction_provider}")

    async def correct(self, transcript: str) -> CorrectionResult:
        if not self.settings.correction_enabled:
            raise CorrectionUnavailable("Transcript correction is disabled")
        return await self.provider.correct(transcript)

    @property
    def provider_name(self) -> str:
        return self.provider.name
