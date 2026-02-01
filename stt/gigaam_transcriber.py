"""GigaAM-v3 speech-to-text transcriber."""
from __future__ import annotations

import importlib
import logging
import os
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
            # Allowlist omegaconf objects for torch.load(weights_only=True) in newer torch.
            try:  # pragma: no cover - best effort compatibility shim
                from omegaconf import DictConfig, ListConfig  # type: ignore
                from omegaconf.base import ContainerMetadata  # type: ignore
            except Exception:
                model = gigaam_module.load_model(self._model_name, device="cpu")
            else:
                # Force weights_only=False for compatibility with newer torch defaults.
                original_torch_load = torch.load
                try:
                    import inspect

                    if "weights_only" in inspect.signature(torch.load).parameters:
                        def _torch_load_compat(*args, **kwargs):
                            kwargs.setdefault("weights_only", False)
                            return original_torch_load(*args, **kwargs)

                        torch.load = _torch_load_compat  # type: ignore[assignment]

                    if hasattr(torch.serialization, "safe_globals"):
                        with torch.serialization.safe_globals([DictConfig, ListConfig, ContainerMetadata]):
                            model = gigaam_module.load_model(self._model_name, device="cpu")
                    elif hasattr(torch.serialization, "add_safe_globals"):
                        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
                        model = gigaam_module.load_model(self._model_name, device="cpu")
                    else:
                        model = gigaam_module.load_model(self._model_name, device="cpu")
                finally:
                    torch.load = original_torch_load
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
        try:
            text = model.transcribe(str(audio_path))
        except ValueError as exc:
            if "Too long wav file" not in str(exc) or not hasattr(model, "transcribe_longform"):
                raise
            logging.info("STT GigaAM: longform mode for %s", audio_path)
            text = self._transcribe_longform(model, audio_path)
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

    def _transcribe_longform(self, model, audio_path: Path) -> str:
        try:
            from stt.vad_segmenter import segment_audio_vad
        except Exception as exc:
            raise RuntimeError(
                "Longform STT requires webrtcvad (offline VAD) to be installed "
                "in the STT venv. Run scripts/prepare_stt_longform_offline.sh."
            ) from exc

        from gigaam.preprocess import SAMPLE_RATE, load_audio

        wav_int = load_audio(str(audio_path), sample_rate=SAMPLE_RATE, return_format="int")
        segments, _boundaries = segment_audio_vad(wav_int, SAMPLE_RATE)
        if not segments:
            segments = [wav_int.float() / 32768.0]

        parts: list[str] = []
        device = self._resolve_device() or model._device
        for segment in segments:
            wav = segment.to(device).unsqueeze(0).to(model._dtype)
            length = torch.full([1], wav.shape[-1], device=wav.device)
            encoded, encoded_len = model.forward(wav, length)
            result = model.decoding.decode(model.head, encoded, encoded_len)[0]
            if result:
                parts.append(str(result).strip())
        return " ".join(parts).strip()
