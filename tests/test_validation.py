from app.core.config import Settings
from app.services.validation_service import TranscriptValidationService


def test_audio_formats_accept_comma_separated_environment_value(monkeypatch):
    monkeypatch.setenv("SUPPORTED_AUDIO_FORMATS", "wav,mp3,m4a,webm")
    settings = Settings(_env_file=None)
    assert settings.supported_audio_formats == ["wav", "mp3", "m4a", "webm"]


def test_rejects_changed_arabic_number():
    result = TranscriptValidationService(Settings()).validate("ငွေ 50000 လွှဲချင်တယ်", "ငွေ 5000 လွှဲချင်တယ်")
    assert not result.passed
    assert "protected_value_changed" in result.warnings


def test_rejects_changed_myanmar_number():
    result = TranscriptValidationService(Settings()).validate("ငွေ ၅၀၀၀၀", "ငွေ ၅၀၀၀")
    assert not result.passed
