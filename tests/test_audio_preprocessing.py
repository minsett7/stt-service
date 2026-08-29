import asyncio
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from app.core.config import Settings
from app.services.audio_preprocessing_service import AudioPreprocessingService
from app.services.audio_quality_service import AudioQualityService
from app.services.vad_service import EnergyVADService
from app.utils.audio_utils import read_wav, write_wav


def _tone(seconds: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.2, frequency: float = 440) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_dc_offset_removal_centers_waveform(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    write_wav(source, _tone() + 0.2, 16000)
    service = AudioPreprocessingService(Settings())
    service._remove_dc_offset(source, output)
    samples, _ = read_wav(output)
    assert abs(float(np.mean(samples))) < 0.001


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is available in the service container, not this local test environment")
def test_canonical_conversion_produces_16k_mono_pcm_wav(tmp_path: Path):
    source = tmp_path / "stereo_48k.wav"
    output = tmp_path / "canonical.wav"
    stereo = np.column_stack([_tone(sample_rate=48000), _tone(sample_rate=48000, frequency=660)])
    sf.write(source, stereo, 48000, subtype="PCM_16")
    service = AudioPreprocessingService(Settings())
    asyncio.run(service.convert_to_canonical_format(source, output))
    info = sf.info(output)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_high_pass_reduces_low_frequency_energy(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    low_rumble = _tone(frequency=20, amplitude=0.4)
    write_wav(source, low_rumble, 16000)
    service = AudioPreprocessingService(Settings())
    service._apply_high_pass(source, output)
    filtered, _ = read_wav(output)
    assert np.sqrt(np.mean(filtered**2)) < np.sqrt(np.mean(low_rumble**2)) * 0.5
    assert len(filtered) == len(low_rumble)


def test_vad_trims_silence_and_retains_padding(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "speech.wav"
    samples = np.concatenate([np.zeros(16000), _tone(), np.zeros(16000)])
    write_wav(source, samples, 16000)
    result = EnergyVADService(Settings()).trim_to_speech(source, output)
    trimmed, _ = read_wav(output)
    assert result.speech_detected
    assert 0.9 <= result.speech_duration_seconds <= 1.1
    assert 1.3 <= len(trimmed) / 16000 <= 1.7


def test_vad_rejects_silence(tmp_path: Path):
    source = tmp_path / "silence.wav"
    write_wav(source, np.zeros(16000, dtype=np.float32), 16000)
    result = EnergyVADService(Settings()).trim_to_speech(source, tmp_path / "output.wav")
    assert not result.speech_detected


def test_quality_detects_low_volume_and_clipping(tmp_path: Path):
    service = AudioQualityService(Settings(audio_normalization_enabled=False))
    quiet_path = tmp_path / "quiet.wav"
    write_wav(quiet_path, _tone(amplitude=0.0005), 16000)
    quiet = service.analyze(quiet_path, 1.0, 1.0, True)
    assert quiet.volume_too_low

    clipped_path = tmp_path / "clipped.wav"
    write_wav(clipped_path, np.ones(16000, dtype=np.float32), 16000)
    clipped = service.analyze(clipped_path, 1.0, 1.0, True)
    assert clipped.clipping_detected
    assert clipped.quality == "bad"

    good_path = tmp_path / "good.wav"
    write_wav(good_path, _tone(), 16000)
    good = service.analyze(good_path, 1.0, 1.0, True)
    assert good.quality == "good"


def test_noise_suppression_unavailable_falls_back(tmp_path: Path):
    source = tmp_path / "source.wav"
    write_wav(source, _tone(), 16000)
    service = AudioPreprocessingService(Settings(audio_noise_suppression_provider="unavailable"))
    result_path, warning = asyncio.run(service._suppress_noise(source, tmp_path / "output.wav"))
    assert result_path == source
    assert warning == "noise_suppression_unavailable"
