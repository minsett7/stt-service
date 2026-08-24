from app.core.exceptions import CorrectionUnavailable
from app.schemas.transcription import CorrectionResult
from tests.conftest import FakeProvider


def _audio():
    return {"file": ("sample.wav", b"minimal-audio", "audio/wav")}


def test_successful_transcription(client):
    response = client.post("/api/v1/transcribe", files=_audio())
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["final_transcript"] == "ATM ကတ်ရဲ့ PIN နံပါတ် မေ့သွားတယ်"
    assert data["used_correction"] is True


def test_correction_failure_falls_back_to_raw(client):
    service = client.app.state.transcription_service
    service.correction.provider = FakeProvider(error=CorrectionUnavailable("down"))
    response = client.post("/api/v1/transcribe", files=_audio())
    data = response.json()["data"]
    assert response.status_code == 200
    assert data["final_transcript"] == data["raw_transcript"]
    assert data["used_correction"] is False
    assert "transcript_correction_unavailable" in data["warnings"]


def test_unsupported_format(client):
    response = client.post("/api/v1/transcribe", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_audio_format"


def test_oversized_audio(client):
    client.app.state.settings.max_audio_size_mb = 1
    service = client.app.state.transcription_service
    service.audio.settings.max_audio_size_mb = 1
    response = client.post("/api/v1/transcribe", files={"file": ("large.wav", b"x" * (1024 * 1024 + 1), "audio/wav")})
    assert response.status_code == 413
