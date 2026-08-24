from app.core.config import Settings
from app.core.exceptions import CorrectionUnavailable
from app.providers.gemini_provider import GeminiCorrectionProvider


def test_invalid_correction_json_is_rejected():
    provider = GeminiCorrectionProvider(Settings(gemini_api_key=None, gemini_model=None))
    try:
        provider._parse_response("not json")
    except CorrectionUnavailable:
        pass
    else:
        raise AssertionError("Invalid correction JSON must be rejected")
