from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.transcription import CorrectionResult


class TranscriptCorrectionProvider(ABC):
    name: str

    @abstractmethod
    async def correct(self, transcript: str) -> CorrectionResult:
        """Return only a constrained correction of `transcript`."""
