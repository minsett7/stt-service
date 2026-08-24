from __future__ import annotations

import asyncio
import os
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
            transcription_path = source_path
            if self.settings.normalize_audio:
                transcription_path = await self._normalize(source_path, temp_dir)
            return PreparedAudio(source_path, transcription_path, temp_dir)
        except Exception:
            self.cleanup(PreparedAudio(source_path, source_path, temp_dir))
            raise
        finally:
            await upload.close()

    async def _normalize(self, source_path: Path, temp_dir: Path) -> Path:
        output_path = temp_dir / "normalized.wav"
        command = [
            self.settings.ffmpeg_binary, "-y", "-i", str(source_path), "-ac", "1",
            "-ar", str(self.settings.normalized_sample_rate), "-c:a", "pcm_s16le", str(output_path),
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            if await process.wait() != 0 or not output_path.exists():
                raise AudioValidationError("audio_preprocessing_failed", "Audio could not be prepared.")
        except FileNotFoundError as exc:
            raise AudioValidationError("audio_preprocessing_failed", "Audio preprocessing is unavailable.") from exc
        return output_path

    @staticmethod
    def cleanup(audio: PreparedAudio) -> None:
        shutil.rmtree(audio.temp_dir, ignore_errors=True)
