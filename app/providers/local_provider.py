from app.core.exceptions import CorrectionUnavailable
from app.providers.base import TranscriptCorrectionProvider
from app.schemas.transcription import CorrectionResult


class LocalCorrectionProvider(TranscriptCorrectionProvider):
    """Extension point for a local Qwen/SLM correction runtime."""

    name = "local"

    async def correct(self, transcript: str) -> CorrectionResult:
        raise CorrectionUnavailable("Local correction provider is not configured")
