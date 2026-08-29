from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.schemas.audio import AudioQualityResult
from app.utils.audio_utils import amplitude_to_dbfs, read_wav


class AudioQualityService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def analyze(self, audio_path, speech_duration_seconds: float, original_duration_seconds: float, speech_detected: bool) -> AudioQualityResult:
        samples, sample_rate = read_wav(audio_path)
        duration = len(samples) / sample_rate if sample_rate else 0.0
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        clipping_ratio = float(np.mean(np.abs(samples) >= self.settings.audio_clipping_threshold)) if len(samples) else 0.0
        peak_dbfs = amplitude_to_dbfs(peak)
        rms_dbfs = amplitude_to_dbfs(rms)
        volume_too_low = rms_dbfs is None or rms_dbfs < self.settings.audio_low_volume_rms_dbfs
        volume_too_high = clipping_ratio > self.settings.audio_max_clipping_ratio
        ratio = speech_duration_seconds / original_duration_seconds if original_duration_seconds else 0.0

        if not speech_detected or volume_too_high or (rms_dbfs is not None and rms_dbfs < self.settings.audio_extremely_low_volume_rms_dbfs):
            quality = "bad"
        elif volume_too_low or clipping_ratio > 0 or ratio < self.settings.audio_min_speech_ratio:
            quality = "acceptable"
        else:
            quality = "good"
        return AudioQualityResult(
            quality=quality, duration_seconds=round(duration, 3), speech_duration_seconds=round(speech_duration_seconds, 3),
            speech_ratio=round(ratio, 3), sample_rate=sample_rate, channels=1,
            peak_dbfs=round(peak_dbfs, 2) if peak_dbfs is not None else None,
            rms_dbfs=round(rms_dbfs, 2) if rms_dbfs is not None else None,
            clipping_detected=clipping_ratio > 0, clipping_ratio=round(clipping_ratio, 5),
            volume_too_low=volume_too_low, volume_too_high=volume_too_high, speech_detected=speech_detected,
        )
