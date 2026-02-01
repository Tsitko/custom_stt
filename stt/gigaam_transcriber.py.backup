"""GigaAM-v3 speech-to-text transcriber."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

import torch

from utils.interfaces import ISTTTranscriber

try:  # pragma: no cover - used for test injection
    import gigaam as _GIGAAM  # type: ignore
except Exception:  # pragma: no cover - resolved lazily
    _GIGAAM = None


class GigaAMTranscriber(ISTTTranscriber):
    """Transcribes speech using Salute GigaAM (PyTorch backend)."""

    def __init__(self, model_name: str = "e2e_rnnt", device: str = "auto") -> None:
        self._model_name = model_name
        self._device_pref = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            gigaam_module = _GIGAAM or importlib.import_module("gigaam")  # type: ignore

            device = self._resolve_device()
            logging.info("Загрузка GigaAM: model=%s device=%s", self._model_name, device)
            model = gigaam_module.load_model(self._model_name)
            if device:
                if hasattr(model, "to"):
                    model = model.to(device)
                elif hasattr(model, "model") and hasattr(model.model, "to"):
                    model.model.to(device)
            self._model = model
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        model = self._load_model()
        logging.info("STT GigaAM: распознавание %s", audio_path)
        text = model.transcribe(str(audio_path))
        if isinstance(text, tuple):
            # некоторые реализации могут возвращать (text, meta)
            text = text[0]
        return str(text).strip()

    def _resolve_device(self) -> torch.device | None:
        if self._device_pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device_pref:
            return torch.device(self._device_pref)
        return None
