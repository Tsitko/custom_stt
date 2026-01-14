"""Silero TTS engine (v5 family) using torch.hub."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torchaudio

from utils.interfaces import ITTSEngine, VoiceReference


class SileroTTSEngine(ITTSEngine):
    """Lightweight wrapper over snakers4/silero-models."""

    _MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}

    def __init__(
        self,
        voice: str = "eugene",
        language: str = "ru",
        variant: str = "v5_ru",
        sample_rate: int = 48000,
        normalize: bool = True,
    ) -> None:
        self._voice = voice
        self._language = language
        self._variant = variant
        self._sample_rate = sample_rate
        self._normalize = normalize

    def synthesize_to_wav(
        self,
        text: str,
        references: Sequence[VoiceReference],  # ignored by Silero
        output_path: Path,
        emotion: str | None = None,
        style: str | None = None,
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("Text for synthesis must not be empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        speaker = self._resolve_speaker(model)
        logging.info("Silero speaker=%s variant=%s", speaker, self._variant)

        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=self._sample_rate,
            put_accent=True,
            put_yo=True,
        )

        tensor = audio if isinstance(audio, torch.Tensor) else torch.tensor(audio)
        tensor = tensor.squeeze().float()
        if self._normalize:
            peak = torch.max(torch.abs(tensor))
            if peak > 0:
                tensor = tensor / peak * 0.98
            logging.info("Normalization applied (peak=%.6f)", float(peak))

        torchaudio.save(str(output_path), tensor.unsqueeze(0), self._sample_rate)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Silero synthesis failed: output WAV not created")
        return output_path

    def _load_model(self) -> torch.nn.Module:
        key = (self._language, self._variant)
        if key not in self._MODEL_CACHE:
            cache_repo = Path(torch.hub.get_dir()) / "snakers4_silero-models_master"
            repo_or_dir = str(cache_repo if cache_repo.exists() else "snakers4/silero-models")
            source = "local" if cache_repo.exists() else "github"
            logging.info(
                "Loading Silero model via torch.hub source=%s repo=%s language=%s variant=%s",
                source,
                repo_or_dir,
                *key,
            )
            model, _ = torch.hub.load(
                repo_or_dir=repo_or_dir,
                model="silero_tts",
                language=self._language,
                speaker=self._variant,
                source=source,
                trust_repo=True,
            )
            self._MODEL_CACHE[key] = model
        return self._MODEL_CACHE[key]

    def _resolve_speaker(self, model: torch.nn.Module) -> str:
        available = getattr(model, "speakers", None) or getattr(model, "speaker_ids", None)
        if available and self._voice in available:
            return self._voice
        if available:
            logging.warning("Preferred speaker %s not found, using %s", self._voice, available[0])
            return available[0]
        return self._voice
