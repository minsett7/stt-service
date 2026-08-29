import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.schemas.transcription import ASRResult, CorrectionResult
from app.services.audio_service import AudioService
from app.services.correction_service import CorrectionService
from app.services.transcription_service import TranscriptionService
from app.services.validation_service import TranscriptValidationService


class FakeASR:
    ready = True

    def __init__(self, text: str = "အေတီအမ် ကတ်ရဲ့ ပင် နံပါတ် မေ့သွားတယ်"):
        self.text = text

    async def transcribe(self, _):
        return ASRResult(text=self.text)


class FakeProvider:
    name = "gemini"

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    async def correct(self, _):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def settings():
    return Settings(normalize_audio=False, audio_preprocessing_enabled=False, correction_enabled=True)


@pytest.fixture
def client(settings):
    with TestClient(app) as test_client:
        provider = FakeProvider(CorrectionResult(corrected_transcript="ATM ကတ်ရဲ့ PIN နံပါတ် မေ့သွားတယ်", overall_confidence=0.95))
        service = TranscriptionService(
            AudioService(settings), FakeASR(), CorrectionService(settings, provider), TranscriptValidationService(settings)
        )
        app.state.settings = settings
        app.state.transcription_service = service
        yield test_client
