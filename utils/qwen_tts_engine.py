"""Qwen3-TTS engine with voice cloning support."""
from __future__ import annotations

import gc
import logging
import os
import re
import sys
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import soundfile as sf

from utils.interfaces import ITTSEngine, VoiceReference

# Suppress all warnings and progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class QwenTTSEngine(ITTSEngine):
    """TTS engine using Qwen3-TTS-12Hz-1.7B-Base with voice cloning."""

    _model = None
    _model_loaded_at: float = 0.0

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_text_chars: int = 200,
        crossfade_ms: int = 50,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        ttl_seconds: int = 300,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dtype = getattr(torch, dtype, torch.bfloat16)
        self._max_text_chars = max_text_chars
        self._crossfade_ms = crossfade_ms
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._ttl_seconds = ttl_seconds

    def synthesize_to_wav(
        self,
        text: str,
        references: Sequence[VoiceReference],
        output_path: Path,
        emotion: str | None = None,
        style: str | None = None,
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("Text for synthesis must not be empty")
        if not references:
            raise ValueError("At least one voice reference is required for Qwen TTS")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ref = references[0]
        ref_audio = str(ref.audio_path)
        ref_text = ref.transcript

        model = self._get_model()
        chunks = self._split_text(text)
        logging.info("Qwen TTS: %d chunks, ref=%s", len(chunks), ref_audio)

        all_wavs = []
        sample_rate = None

        for i, chunk_text in enumerate(chunks, 1):
            start = time.perf_counter()
            wavs, sr = model.generate_voice_clone(
                text=chunk_text,
                language="Russian",
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
            )
            elapsed = time.perf_counter() - start
            all_wavs.append(wavs[0])
            sample_rate = sr
            logging.info("Qwen TTS chunk %d/%d: %.1fs", i, len(chunks), elapsed)

        combined = self._crossfade_concat(all_wavs, sample_rate)
        sf.write(str(output_path), combined, sample_rate)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Qwen TTS synthesis failed: output WAV not created")

        return output_path

    def _get_model(self):
        self._check_ttl()

        if QwenTTSEngine._model is None:
            logging.info("Loading Qwen TTS model: %s", self._model_name)
            start = time.perf_counter()

            # Suppress flash-attn warning that goes to stdout
            _stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                from qwen_tts import Qwen3TTSModel

                QwenTTSEngine._model = Qwen3TTSModel.from_pretrained(
                    self._model_name,
                    device_map=self._device,
                    dtype=self._dtype,
                )
            finally:
                sys.stdout = _stdout

            # Compile model for faster inference
            try:
                QwenTTSEngine._model = torch.compile(QwenTTSEngine._model, mode="reduce-overhead")
                logging.info("Qwen TTS model compiled with torch.compile()")
            except Exception as e:
                logging.warning("torch.compile() failed, using eager mode: %s", e)

            elapsed = time.perf_counter() - start
            logging.info("Qwen TTS model loaded in %.1fs", elapsed)

        QwenTTSEngine._model_loaded_at = time.time()
        return QwenTTSEngine._model

    def _check_ttl(self) -> None:
        if QwenTTSEngine._model is None:
            return
        now = time.time()
        if now - QwenTTSEngine._model_loaded_at > self._ttl_seconds:
            self._unload_model()

    def _unload_model(self) -> None:
        if QwenTTSEngine._model is not None:
            logging.info("Unloading Qwen TTS model from VRAM (TTL expired)")
            del QwenTTSEngine._model
            QwenTTSEngine._model = None
            QwenTTSEngine._model_loaded_at = 0.0
            gc.collect()
            torch.cuda.empty_cache()

    def _normalize_text(self, text: str) -> str:
        """Remove problematic characters before TTS."""
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().rstrip('.')
        text = text.replace('—', '-').replace('–', '-')
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks by sentences, not exceeding max_chars."""
        text = self._normalize_text(text)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self._max_text_chars:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(sentence) > self._max_text_chars:
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= self._max_text_chars:
                            current_chunk = (current_chunk + " " + word).strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _crossfade_concat(self, chunks: list[np.ndarray], sr: int) -> np.ndarray:
        """Concatenate audio chunks with crossfade to eliminate clicks."""
        if not chunks:
            return np.array([])
        if len(chunks) == 1:
            return chunks[0]

        crossfade_samples = int(sr * self._crossfade_ms / 1000)
        result = chunks[0].copy()

        for i in range(1, len(chunks)):
            chunk = chunks[i].copy()

            if crossfade_samples > 0 and len(result) >= crossfade_samples and len(chunk) >= crossfade_samples:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)

                result[-crossfade_samples:] *= fade_out
                chunk[:crossfade_samples] *= fade_in

                result[-crossfade_samples:] += chunk[:crossfade_samples]
                result = np.concatenate([result, chunk[crossfade_samples:]])
            else:
                result = np.concatenate([result, chunk])

        return result
