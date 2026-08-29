from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from app.core.config import Settings
from app.core.exceptions import AudioValidationError
from app.schemas.audio import AudioProcessingTimings, AudioQualityResult
from app.services.audio_quality_service import AudioQualityService
from app.services.vad_service import EnergyVADService
from app.utils.audio_utils import read_wav, write_wav

logger = logging.getLogger(__name__)


@dataclass
class ProcessedAudio:
    path: Path
    quality: AudioQualityResult
    warnings: list[str]
    timings: AudioProcessingTimings


class AudioPreprocessingService:
    def __init__(self, settings: Settings, quality_service: AudioQualityService | None = None, vad_service: EnergyVADService | None = None) -> None:
        self.settings = settings
        self.quality_service = quality_service or AudioQualityService(settings)
        self.vad_service = vad_service or EnergyVADService(settings)
        self.ffmpeg_ready = shutil.which(settings.ffmpeg_binary) is not None

    async def preprocess(self, input_path: Path, temp_dir: Path) -> ProcessedAudio:
        started = time.perf_counter()
        timings = AudioProcessingTimings()
        warnings: list[str] = []
        canonical = temp_dir / "canonical.wav"
        decode_started = time.perf_counter()
        await self.convert_to_canonical_format(input_path, canonical)
        timings.decode_ms = _elapsed_ms(decode_started)

        samples, sample_rate = read_wav(canonical)
        original_duration = len(samples) / sample_rate if sample_rate else 0.0
        original_clipping_ratio = float(np.mean(np.abs(samples) >= self.settings.audio_clipping_threshold)) if len(samples) else 0.0
        if original_duration == 0:
            raise AudioValidationError("audio_decode_failed", "Audio contains no samples.")
        if original_duration > self.settings.audio_max_duration_seconds:
            raise AudioValidationError("audio_too_long", "Audio exceeds the permitted duration.", 413)
        current = canonical

        if self.settings.audio_dc_offset_removal_enabled:
            filter_started = time.perf_counter()
            current = self._remove_dc_offset(current, temp_dir / "dc_removed.wav")
            timings.filter_ms += _elapsed_ms(filter_started)
        if self.settings.audio_high_pass_enabled:
            filter_started = time.perf_counter()
            current = self._apply_high_pass(current, temp_dir / "high_pass.wav")
            timings.filter_ms += _elapsed_ms(filter_started)
        if self.settings.audio_noise_suppression_enabled:
            noise_started = time.perf_counter()
            current, noise_warning = await self._suppress_noise(current, temp_dir / "denoised.wav")
            timings.noise_suppression_ms = _elapsed_ms(noise_started)
            if noise_warning:
                warnings.append(noise_warning)
        if self.settings.audio_normalization_enabled:
            normalization_started = time.perf_counter()
            current = self._normalize(current, temp_dir / "normalized.wav")
            timings.normalization_ms = _elapsed_ms(normalization_started)

        vad_started = time.perf_counter()
        vad_result = self.vad_service.trim_to_speech(current, temp_dir / "speech.wav")
        timings.vad_ms = _elapsed_ms(vad_started)
        if not vad_result.speech_detected:
            raise AudioValidationError("no_speech_detected", "No meaningful speech was detected in the audio.", 422)
        current = vad_result.path

        quality_started = time.perf_counter()
        quality = self.quality_service.analyze(current, vad_result.speech_duration_seconds, original_duration, vad_result.speech_detected)
        if original_clipping_ratio > 0:
            severe_clipping = original_clipping_ratio > self.settings.audio_max_clipping_ratio
            quality = quality.model_copy(update={
                "clipping_detected": True,
                "clipping_ratio": round(original_clipping_ratio, 5),
                "volume_too_high": severe_clipping,
                "quality": "bad" if severe_clipping else quality.quality,
            })
        timings.quality_analysis_ms = _elapsed_ms(quality_started)
        if quality.volume_too_low:
            warnings.append("audio_volume_too_low")
        if quality.clipping_detected:
            warnings.append("audio_clipping_detected")
        if quality.volume_too_high:
            raise AudioValidationError("severe_clipping", "The recorded speech is too distorted to transcribe reliably.", 422)
        if quality.quality == "bad":
            code = "audio_volume_too_low" if quality.volume_too_low else "poor_audio_quality"
            raise AudioValidationError(code, "The recorded speech is not clear enough to transcribe reliably.", 422)
        timings.audio_preprocessing_ms = _elapsed_ms(started)
        self._save_debug_audio(temp_dir)
        logger.info("audio_preprocessing_completed duration_ms=%d quality=%s speech_duration_seconds=%.3f", timings.audio_preprocessing_ms, quality.quality, quality.speech_duration_seconds)
        return ProcessedAudio(path=current, quality=quality, warnings=warnings, timings=timings)

    async def convert_to_canonical_format(self, input_path: Path, output_path: Path) -> None:
        if not self.ffmpeg_ready:
            raise AudioValidationError("audio_preprocessing_failed", "Audio preprocessing is unavailable.", 503)
        command = [
            self.settings.ffmpeg_binary, "-v", "error", "-y", "-i", str(input_path), "-vn", "-ac", str(self.settings.audio_target_channels),
            "-ar", str(self.settings.audio_target_sample_rate), "-c:a", "pcm_s16le", str(output_path),
        ]
        try:
            process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            if await process.wait() != 0 or not output_path.exists():
                raise AudioValidationError("audio_decode_failed", "Audio could not be decoded.")
        except FileNotFoundError as exc:
            self.ffmpeg_ready = False
            raise AudioValidationError("audio_preprocessing_failed", "Audio preprocessing is unavailable.", 503) from exc

    def _remove_dc_offset(self, input_path: Path, output_path: Path) -> Path:
        samples, sample_rate = read_wav(input_path)
        if len(samples):
            samples = samples - np.mean(samples, dtype=np.float64)
        write_wav(output_path, samples, sample_rate)
        return output_path

    def _apply_high_pass(self, input_path: Path, output_path: Path) -> Path:
        samples, sample_rate = read_wav(input_path)
        cutoff = min(self.settings.audio_high_pass_hz, sample_rate * 0.45)
        sos = butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, samples) if len(samples) > 32 else sosfilt(sos, samples)
        write_wav(output_path, filtered, sample_rate)
        return output_path

    def _normalize(self, input_path: Path, output_path: Path) -> Path:
        samples, sample_rate = read_wav(input_path)
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak:
            target = 10 ** (self.settings.audio_target_peak_dbfs / 20)
            max_gain = 10 ** (self.settings.audio_max_gain_db / 20)
            samples = samples * min(target / peak, max_gain)
        write_wav(output_path, samples, sample_rate)
        return output_path

    async def _suppress_noise(self, input_path: Path, output_path: Path) -> tuple[Path, str | None]:
        if self.settings.audio_noise_suppression_provider == "none":
            return input_path, None
        if self.settings.audio_noise_suppression_provider != "ffmpeg_afftdn" or not self.ffmpeg_ready:
            return input_path, "noise_suppression_unavailable"
        command = [self.settings.ffmpeg_binary, "-v", "error", "-y", "-i", str(input_path), "-af", "afftdn=nr=12:nf=-25", "-c:a", "pcm_s16le", str(output_path)]
        try:
            process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            if await process.wait() == 0 and output_path.exists():
                return output_path, None
        except FileNotFoundError:
            self.ffmpeg_ready = False
        logger.warning("noise_suppression_unavailable")
        return input_path, "noise_suppression_unavailable"

    def _save_debug_audio(self, temp_dir: Path) -> None:
        if not self.settings.audio_debug_save_intermediate:
            return
        destination = Path(self.settings.audio_debug_directory) / temp_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for audio_file in temp_dir.iterdir():
            if not audio_file.is_file():
                continue
            shutil.copy2(audio_file, destination / audio_file.name)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
