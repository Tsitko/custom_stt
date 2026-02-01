"""Qwen3-ASR-1.7B speech-to-text transcriber."""
from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from utils.interfaces import ISTTTranscriber

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


class QwenASRTranscriber(ISTTTranscriber):
    """Transcribes speech using Qwen3-ASR-1.7B."""

    _model = None
    _model_loaded_at: float = 0.0

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_audio_seconds: int = 30,
        ttl_seconds: int = 300,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dtype = getattr(torch, dtype, torch.bfloat16)
        self._max_audio_seconds = max_audio_seconds
        self._ttl_seconds = ttl_seconds

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        chunks = self._split_audio(audio_path)
        model = self._get_model()

        all_text = []
        total_start = time.perf_counter()

        for i, (chunk_audio, sr) in enumerate(chunks, 1):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                sf.write(tmp_path, chunk_audio, sr)

            try:
                chunk_start = time.perf_counter()
                results = model.transcribe(audio=str(tmp_path), language="Russian")
                chunk_elapsed = time.perf_counter() - chunk_start

                text = results[0].text if results else ""
                all_text.append(text)
                logging.info(
                    "Qwen ASR chunk %d/%d: %.1fs - '%s...'",
                    i, len(chunks), chunk_elapsed, text[:50]
                )
            finally:
                tmp_path.unlink(missing_ok=True)

        total_elapsed = time.perf_counter() - total_start
        full_text = " ".join(all_text)
        logging.info("Qwen ASR total: %.1fs, %d chars", total_elapsed, len(full_text))

        return full_text.strip()

    def _get_model(self):
        self._check_ttl()

        if QwenASRTranscriber._model is None:
            logging.info("Loading Qwen ASR model: %s", self._model_name)
            start = time.perf_counter()
            from qwen_asr import Qwen3ASRModel

            QwenASRTranscriber._model = Qwen3ASRModel.from_pretrained(
                self._model_name,
                dtype=self._dtype,
                device_map=self._device,
            )
            elapsed = time.perf_counter() - start
            logging.info("Qwen ASR model loaded in %.1fs", elapsed)

        QwenASRTranscriber._model_loaded_at = time.time()
        return QwenASRTranscriber._model

    def _check_ttl(self) -> None:
        if QwenASRTranscriber._model is None:
            return
        now = time.time()
        if now - QwenASRTranscriber._model_loaded_at > self._ttl_seconds:
            self._unload_model()

    def _unload_model(self) -> None:
        if QwenASRTranscriber._model is not None:
            logging.info("Unloading Qwen ASR model from VRAM (TTL expired)")
            del QwenASRTranscriber._model
            QwenASRTranscriber._model = None
            QwenASRTranscriber._model_loaded_at = 0.0
            gc.collect()
            torch.cuda.empty_cache()

    def _split_audio(self, audio_path: Path) -> list[tuple[np.ndarray, int]]:
        """Split audio into chunks of max_audio_seconds."""
        import librosa

        logging.info("Qwen ASR loading audio: %s", audio_path)
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        duration = len(audio) / sr
        logging.info("Qwen ASR audio duration: %.1fs, sample_rate: %d", duration, sr)

        chunk_samples = int(self._max_audio_seconds * sr)
        chunks = []

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            chunks.append((chunk, sr))

        logging.info("Qwen ASR split into %d chunks of ~%ds", len(chunks), self._max_audio_seconds)
        return chunks
