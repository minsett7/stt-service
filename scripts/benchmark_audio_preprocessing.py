"""Compare deterministic preprocessing profiles on one local recording."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services.audio_preprocessing_service import AudioPreprocessingService


PROFILES = {
    "resample_only": {"audio_dc_offset_removal_enabled": False, "audio_high_pass_enabled": False, "audio_noise_suppression_enabled": False, "audio_normalization_enabled": False, "vad_enabled": False},
    "resample_vad": {"audio_dc_offset_removal_enabled": False, "audio_high_pass_enabled": False, "audio_noise_suppression_enabled": False, "audio_normalization_enabled": False, "vad_enabled": True},
    "resample_vad_denoise": {"audio_dc_offset_removal_enabled": False, "audio_high_pass_enabled": False, "audio_noise_suppression_enabled": True, "audio_normalization_enabled": False, "vad_enabled": True},
    "full": {},
}


async def benchmark(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise SystemExit(f"Audio file does not exist: {audio_path}")
    for name, overrides in PROFILES.items():
        settings = Settings().model_copy(update=overrides)
        service = AudioPreprocessingService(settings)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark-{name}-"))
        try:
            source = temp_dir / f"input{audio_path.suffix.lower()}"
            shutil.copy2(audio_path, source)
            result = await service.preprocess(source, temp_dir)
            print(f"{name}: {result.quality.model_dump_json()} timings={result.timings.model_dump_json()} warnings={result.warnings}")
        except Exception as exc:
            print(f"{name}: failed ({getattr(exc, 'code', type(exc).__name__)})")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path)
    asyncio.run(benchmark(parser.parse_args().audio_file))
