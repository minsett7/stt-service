from __future__ import annotations

import logging
import time

from fastapi import UploadFile

from app.core.exceptions import CorrectionUnavailable
from app.schemas.transcription import TranscriptionData
from app.services.audio_service import AudioService
from app.services.audio_preprocessing_service import AudioPreprocessingService
from app.services.asr_service import ASRService
from app.services.correction_service import CorrectionService
from app.services.validation_service import TranscriptValidationService

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, audio: AudioService, asr: ASRService, correction: CorrectionService, validation: TranscriptValidationService, preprocessor: AudioPreprocessingService | None = None) -> None:
        self.audio = audio
        self.asr = asr
        self.correction = correction
        self.validation = validation
        self.preprocessor = preprocessor

    async def transcribe(self, upload: UploadFile) -> TranscriptionData:
        started = time.perf_counter()
        prepared = await self.audio.prepare_upload(upload)
        try:
            preprocessing_warnings: list[str] = []
            audio_quality = None
            timings = None
            transcription_path = prepared.transcription_path
            if self.preprocessor and self.preprocessor.settings.audio_preprocessing_enabled:
                processed = await self.preprocessor.preprocess(prepared.source_path, prepared.temp_dir)
                transcription_path = processed.path
                preprocessing_warnings = processed.warnings
                audio_quality = processed.quality
                timings = processed.timings
            asr_started = time.perf_counter()
            raw = await self.asr.transcribe(transcription_path)
            asr_ms = int((time.perf_counter() - asr_started) * 1000)
            logger.info("asr_completed duration_ms=%d", asr_ms)
            corrected = raw.text
            used_correction = False
            provider_name: str | None = None
            validation_passed = True
            confidence = None
            corrections = []
            warnings: list[str] = list(preprocessing_warnings)
            correction_ms = 0
            try:
                correction_started = time.perf_counter()
                candidate = await self.correction.correct(raw.text)
                logger.info("correction_completed provider=%s", self.correction.provider_name)
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
            finally:
                correction_ms = int((time.perf_counter() - correction_started) * 1000)
                logger.info("correction_finished duration_ms=%d", correction_ms)

            total_ms = int((time.perf_counter() - started) * 1000)
            if timings:
                timings.asr_ms = asr_ms
                timings.correction_ms = correction_ms
                timings.total_processing_ms = total_ms
            return TranscriptionData(
                raw_transcript=raw.text, corrected_transcript=corrected, final_transcript=corrected,
                used_correction=used_correction, correction_provider=provider_name,
                validation_passed=validation_passed, correction_confidence=confidence,
                corrections=corrections, warnings=warnings,
                processing_time_ms=total_ms,
                audio_quality=audio_quality, timings=timings,
            )
        finally:
            self.audio.cleanup(prepared)
