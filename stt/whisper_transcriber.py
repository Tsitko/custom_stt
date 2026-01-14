"""Whisper-based speech-to-text using faster-whisper."""
import importlib
import logging
import sys
import types
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from utils.interfaces import ISTTTranscriber


class WhisperTranscriber(ISTTTranscriber):
    """Transcribes audio using faster-whisper."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            logging.info(
                "Загрузка STT модели faster-whisper: name=%s device=%s compute_type=%s",
                self._model_name,
                self._device,
                self._compute_type,
            )
            try:
                fw = importlib.import_module("faster_whisper")
            except ModuleNotFoundError as exc:
                if "No module named 'av'" in str(exc):
                    logging.warning("faster-whisper: av not found, using lightweight stub for numpy audio input")
                    sys.modules["av"] = types.SimpleNamespace()
                    fw = importlib.import_module("faster_whisper")
                else:
                    raise
            WhisperModel = getattr(fw, "WhisperModel")
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        model = self._get_model()
        logging.info("STT распознавание файла: %s (sr=%s -> %s)", audio_path, sr, target_sr)
        segments, info = model.transcribe(
            audio,
            beam_size=self._beam_size,
            language="ru",
        )

        full_text = " ".join(segment.text.strip() for segment in segments).strip()
        logging.info(
            "STT завершено (language=%s duration=%.2fs text_len=%s)",
            info.language if info else "unknown",
            info.duration if info else 0,
            len(full_text),
        )
        return full_text
