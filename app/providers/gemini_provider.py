from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import CorrectionUnavailable
from app.providers.base import TranscriptCorrectionProvider
from app.schemas.transcription import CorrectionResult


class GeminiCorrectionProvider(TranscriptCorrectionProvider):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.prompt = (Path(__file__).parents[1] / "prompts" / "transcript_correction.txt").read_text(encoding="utf-8")

    async def correct(self, transcript: str) -> CorrectionResult:
        if not self.settings.gemini_api_key or not self.settings.gemini_model:
            raise CorrectionUnavailable("Gemini credentials or model are not configured")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._correct_sync, transcript),
                timeout=self.settings.correction_timeout_seconds,
            )
        except CorrectionUnavailable:
            raise
        except Exception as exc:
            raise CorrectionUnavailable("Gemini correction request failed") from exc

    def _correct_sync(self, transcript: str) -> CorrectionResult:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise CorrectionUnavailable("google-genai is not installed") from exc

        client = genai.Client(api_key=self.settings.gemini_api_key)
        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=f"{self.prompt}\n\nRAW TRANSCRIPT:\n{transcript}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CorrectionResult,
                temperature=0,
            ),
        )
        if not response.text:
            raise CorrectionUnavailable("Gemini returned no correction output")
        return self._parse_response(response.text)

    @staticmethod
    def _parse_response(response_text: str) -> CorrectionResult:
        try:
            return CorrectionResult.model_validate(json.loads(response_text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise CorrectionUnavailable("Gemini returned invalid correction output", code="invalid_correction_output") from exc
