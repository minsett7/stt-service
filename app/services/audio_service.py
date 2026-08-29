from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import AudioValidationError


@dataclass
class PreparedAudio:
    source_path: Path
    transcription_path: Path
    temp_dir: Path


class AudioService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def prepare_upload(self, upload: UploadFile) -> PreparedAudio:
        suffix = Path(upload.filename or "").suffix.lower().lstrip(".")
        if suffix not in self.settings.supported_audio_formats:
            raise AudioValidationError("unsupported_audio_format", "Unsupported audio format.")
        if upload.content_type and not upload.content_type.startswith("audio/") and upload.content_type != "video/webm":
            raise AudioValidationError("unsupported_audio_format", "The uploaded file is not an audio file.")

        temp_dir = Path(tempfile.mkdtemp(prefix="stt-"))
        source_path = temp_dir / f"upload.{suffix}"
        size = 0
        max_size = self.settings.max_audio_size_mb * 1024 * 1024
        try:
            with source_path.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_size:
                        raise AudioValidationError("audio_file_too_large", "Audio file exceeds the permitted size.", 413)
                    output.write(chunk)
            if size == 0:
                raise AudioValidationError("empty_audio_file", "Audio file is empty.")
            return PreparedAudio(source_path, source_path, temp_dir)
        except Exception:
            self.cleanup(PreparedAudio(source_path, source_path, temp_dir))
            raise
        finally:
            await upload.close()

    @staticmethod
    def cleanup(audio: PreparedAudio) -> None:
        shutil.rmtree(audio.temp_dir, ignore_errors=True)
