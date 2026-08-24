from app.core.config import Settings
from app.schemas.transcription import ValidationResult
from app.utils.text_utils import correction_change_ratio, protected_values_match


class TranscriptValidationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate(self, raw_transcript: str, corrected_transcript: str) -> ValidationResult:
        warnings: list[str] = []
        if not protected_values_match(raw_transcript, corrected_transcript):
            warnings.append("protected_value_changed")
        ratio = correction_change_ratio(raw_transcript, corrected_transcript)
        if ratio > self.settings.max_correction_change_ratio:
            warnings.append("correction_change_ratio_exceeded")
        return ValidationResult(passed=not warnings, warnings=warnings, change_ratio=ratio)
