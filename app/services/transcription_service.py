from __future__ import annotations

import logging
import time

from fastapi import UploadFile

from app.core.exceptions import CorrectionUnavailable
from app.schemas.transcription import TranscriptionData
from app.services.audio_service import AudioService
from app.services.asr_service import ASRService
from app.services.correction_service import CorrectionService
from app.services.validation_service import TranscriptValidationService

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, audio: AudioService, asr: ASRService, correction: CorrectionService, validation: TranscriptValidationService) -> None:
        self.audio = audio
        self.asr = asr
        self.correction = correction
        self.validation = validation

    async def transcribe(self, upload: UploadFile) -> TranscriptionData:
        started = time.perf_counter()
        prepared = await self.audio.prepare_upload(upload)
        try:
            asr_started = time.perf_counter()
            raw = await self.asr.transcribe(prepared.transcription_path)
            logger.info("asr_completed duration_ms=%d", int((time.perf_counter() - asr_started) * 1000))
            corrected = raw.text
            used_correction = False
            provider_name: str | None = None
            validation_passed = True
            confidence = None
            corrections = []
            warnings: list[str] = []
            try:
                correction_started = time.perf_counter()
                candidate = await self.correction.correct(raw.text)
                logger.info("correction_completed provider=%s duration_ms=%d", self.correction.provider_name, int((time.perf_counter() - correction_started) * 1000))
                validation = self.validation.validate(raw.text, candidate.corrected_transcript)
                validation_passed = validation.passed
                if validation.passed:
                    corrected = candidate.corrected_transcript
                    used_correction = corrected != raw.text
                    provider_name = self.correction.provider_name
                    confidence = candidate.overall_confidence
                    corrections = candidate.corrections
                else:
                    warnings.extend(validation.warnings)
                    logger.warning("correction_validation_failed warnings=%s", validation.warnings)
            except CorrectionUnavailable as exc:
                warnings.append(exc.code)
                if exc.code == "invalid_correction_output":
                    validation_passed = False
                logger.warning("correction_unavailable reason=%s", str(exc))

            return TranscriptionData(
                raw_transcript=raw.text, corrected_transcript=corrected, final_transcript=corrected,
                used_correction=used_correction, correction_provider=provider_name,
                validation_passed=validation_passed, correction_confidence=confidence,
                corrections=corrections, warnings=warnings,
                processing_time_ms=int((time.perf_counter() - started) * 1000),
            )
        finally:
            self.audio.cleanup(prepared)
