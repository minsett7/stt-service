class AudioValidationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class TranscriptionError(Exception):
    """Raised when ASR cannot produce a transcript."""


class CorrectionUnavailable(Exception):
    """Raised for expected correction provider failures."""

    def __init__(self, message: str, code: str = "transcript_correction_unavailable") -> None:
        self.code = code
        super().__init__(message)


class RequestTimeoutError(Exception):
    """Raised when an end-to-end transcription request exceeds its deadline."""
