from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.utils.audio_utils import read_wav, write_wav


@dataclass
class VADResult:
    path: Path
    speech_detected: bool
    speech_duration_seconds: float


class EnergyVADService:
    """Deterministic energy VAD for clipping silence around short microphone clips."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def trim_to_speech(self, input_path: Path, output_path: Path) -> VADResult:
        samples, sample_rate = read_wav(input_path)
        if not self.settings.vad_enabled:
            write_wav(output_path, samples, sample_rate)
            return VADResult(output_path, True, len(samples) / sample_rate)
        frame_size = max(1, int(sample_rate * self.settings.vad_frame_ms / 1000))
        frame_count = int(np.ceil(len(samples) / frame_size))
        if frame_count == 0:
            return VADResult(input_path, False, 0.0)
        padded = np.pad(samples, (0, frame_count * frame_size - len(samples)))
        frames = padded.reshape(frame_count, frame_size)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        threshold = 10 ** (self.settings.vad_speech_threshold_dbfs / 20)
        active = rms >= threshold
        min_frames = max(1, int(np.ceil(self.settings.vad_min_speech_ms / self.settings.vad_frame_ms)))
        regions: list[tuple[int, int]] = []
        start = None
        for index, is_active in enumerate(active):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                if index - start >= min_frames:
                    regions.append((start, index))
                start = None
        if start is not None and frame_count - start >= min_frames:
            regions.append((start, frame_count))
        if not regions:
            return VADResult(input_path, False, 0.0)

        pad_frames = int(np.ceil(self.settings.vad_speech_pad_ms / self.settings.vad_frame_ms))
        start_frame = max(0, regions[0][0] - pad_frames)
        end_frame = min(frame_count, regions[-1][1] + pad_frames)
        trimmed = samples[start_frame * frame_size:min(len(samples), end_frame * frame_size)]
        write_wav(output_path, trimmed, sample_rate)
        speech_frames = sum(end - start for start, end in regions)
        return VADResult(output_path, True, speech_frames * frame_size / sample_rate)
