from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32), sample_rate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(path, np.clip(samples, -1.0, 1.0), sample_rate, subtype="PCM_16")


def amplitude_to_dbfs(amplitude: float) -> float | None:
    if amplitude <= 0:
        return None
    return 20 * math.log10(amplitude)

