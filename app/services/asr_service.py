from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import TranscriptionError
from app.schemas.transcription import ASRResult

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline = None
        self._load_lock = asyncio.Lock()
        self._load_task: asyncio.Task | None = None

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    async def transcribe(self, audio_path: Path) -> ASRResult:
        try:
            pipeline = await self._get_pipeline()
            result = await asyncio.to_thread(pipeline, str(audio_path), generate_kwargs={"language": "my", "task": "transcribe"})
            text = str(result.get("text", "")).strip()
            if not text:
                raise TranscriptionError("ASR returned an empty transcript")
            return ASRResult(text=text, language="my")
        except TranscriptionError:
            raise
        except Exception as exc:
            logger.exception("asr_transcription_failed")
            raise TranscriptionError("ASR transcription failed") from exc

    async def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        async with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            if self._load_task is None or self._load_task.done():
                self._load_task = asyncio.create_task(asyncio.to_thread(self._load_pipeline))
            load_task = self._load_task
        try:
            # Shield the shared cold-start task: a client timeout must not cancel a
            # model download/load that later requests can reuse.
            self._pipeline = await asyncio.shield(load_task)
            return self._pipeline
        except Exception:
            if load_task.done() and load_task.exception() is not None:
                async with self._load_lock:
                    if self._load_task is load_task:
                        self._load_task = None
            raise

    def _load_pipeline(self):
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise TranscriptionError("ASR dependencies are not installed") from exc

        use_cuda = self.settings.asr_device == "cuda" or (self.settings.asr_device == "auto" and torch.cuda.is_available())
        if self.settings.asr_device == "cuda" and not torch.cuda.is_available():
            raise TranscriptionError("CUDA was requested but is unavailable")
        dtype = torch.float16 if use_cuda and self.settings.asr_dtype in {"auto", "float16"} else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.settings.asr_model, torch_dtype=dtype)
        if use_cuda:
            model.to("cuda")
        processor = AutoProcessor.from_pretrained(self.settings.asr_model)
        return pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, dtype=dtype, device=0 if use_cuda else -1,
        )
