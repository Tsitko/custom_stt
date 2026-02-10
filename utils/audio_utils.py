"""Audio utility helpers."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_rms_dbfs(path: str | Path) -> Optional[float]:
    """Return RMS level in dBFS for a mono mixdown, or None if unreadable."""
    audio_path = Path(path)
    if not audio_path.exists():
        return None

    samples = None
    try:
        import librosa  # type: ignore

        samples, _sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception:
        try:
            import soundfile as sf  # type: ignore

            data, _sr = sf.read(str(audio_path), always_2d=False)
            if isinstance(data, np.ndarray):
                if data.ndim > 1:
                    data = data.mean(axis=1)
                samples = data.astype(np.float32, copy=False)
        except Exception as exc:
            logger.warning("Failed to load audio for RMS check: %s", exc)
            return None

    if samples is None or len(samples) == 0:
        return -float("inf")

    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 1e-12:
        return -float("inf")
    return 20.0 * math.log10(rms)
