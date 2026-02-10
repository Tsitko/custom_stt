"""Qwen3-TTS engine with voice cloning support."""
from __future__ import annotations

import gc
import importlib.util
import logging
import os
import re
import sys
import time
import warnings
import threading
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
    """TTS engine using Qwen3-TTS with voice cloning and voice design."""

    _model = None
    _model_loaded_at: float = 0.0
    _current_model_name: str | None = None
    _prompt_cache: dict[str, object] = {}
    _prompt_cache_lock = threading.Lock()
    _runtime_tuned: bool = False
    _flash_attn_checked: bool = False
    _flash_attn_available: bool = False

    # Модели для разных режимов
    MODEL_VOICE_CLONE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    MODEL_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

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
        attn_implementation: str | None = "flash_attention_2",
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
        self._attn_implementation = attn_implementation
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

        model = self._get_model(self.MODEL_VOICE_CLONE)
        chunks = self._split_text(text)
        logging.info("Qwen TTS (voice clone): %d chunks, ref=%s", len(chunks), ref_audio)

        voice_clone_prompt = self._get_voice_clone_prompt(model, ref_audio, ref_text)
        all_wavs = []
        sample_rate = None

        for i, chunk_text in enumerate(chunks, 1):
            start = time.perf_counter()
            kwargs = dict(
                text=chunk_text,
                language="Russian",
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
            )
            with torch.inference_mode():
                if voice_clone_prompt is not None:
                    try:
                        wavs, sr = model.generate_voice_clone(
                            voice_clone_prompt=voice_clone_prompt,
                            **kwargs,
                        )
                    except TypeError:
                        # Backward-compatible path for older qwen_tts APIs.
                        voice_clone_prompt = None
                        wavs, sr = model.generate_voice_clone(
                            ref_audio=ref_audio,
                            ref_text=ref_text,
                            **kwargs,
                        )
                else:
                    wavs, sr = model.generate_voice_clone(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        **kwargs,
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

    def synthesize_from_description(
        self,
        text: str,
        voice_description: str,
        output_path: Path,
    ) -> Path:
        """Синтез речи по описанию голоса (без референса)."""
        if not text or not text.strip():
            raise ValueError("Text for synthesis must not be empty")
        if not voice_description or not voice_description.strip():
            raise ValueError("Voice description must not be empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._get_model(self.MODEL_VOICE_DESIGN)
        chunks = self._split_text(text)
        logging.info("Qwen TTS (voice design): %d chunks", len(chunks))

        all_wavs = []
        sample_rate = None

        for i, chunk_text in enumerate(chunks, 1):
            start = time.perf_counter()
            with torch.inference_mode():
                wavs, sr = model.generate_voice_design(
                    text=chunk_text,
                    language="Russian",
                    instruct=voice_description,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    repetition_penalty=self._repetition_penalty,
                )
            elapsed = time.perf_counter() - start
            all_wavs.append(wavs[0])
            sample_rate = sr
            logging.info("Qwen TTS (voice design) chunk %d/%d: %.1fs", i, len(chunks), elapsed)

        combined = self._crossfade_concat(all_wavs, sample_rate)
        sf.write(str(output_path), combined, sample_rate)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Qwen TTS synthesis failed: output WAV not created")

        return output_path

    def _get_model(self, model_name: str | None = None):
        target_model = model_name or self._model_name
        self._check_ttl()
        self._tune_runtime()

        # Если загружена другая модель - выгружаем
        if QwenTTSEngine._model is not None and QwenTTSEngine._current_model_name != target_model:
            logging.info("Switching model from %s to %s", QwenTTSEngine._current_model_name, target_model)
            self._unload_model()

        if QwenTTSEngine._model is None:
            logging.info("Loading Qwen TTS model: %s", target_model)
            start = time.perf_counter()

            # Suppress flash-attn warning that goes to stdout
            _stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                from qwen_tts import Qwen3TTSModel

                pretrained_kwargs = self._build_pretrained_kwargs()
                try:
                    QwenTTSEngine._model = Qwen3TTSModel.from_pretrained(target_model, **pretrained_kwargs)
                except Exception as exc:
                    attn_impl = pretrained_kwargs.get("attn_implementation")
                    if attn_impl is None:
                        raise
                    logging.warning("Qwen TTS load failed with attn_implementation=%s: %s", attn_impl, exc)
                    # Retry on SDPA when flash-attn path is not compatible with current stack.
                    pretrained_kwargs["attn_implementation"] = "sdpa"
                    QwenTTSEngine._model = Qwen3TTSModel.from_pretrained(target_model, **pretrained_kwargs)
                    logging.info("Qwen TTS model loaded with fallback attn_implementation=sdpa")
                QwenTTSEngine._current_model_name = target_model
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

    def _build_pretrained_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "device_map": self._device,
            "dtype": self._dtype,
        }
        attn_impl = self._resolve_attn_implementation()
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl
        logging.info(
            "Qwen TTS load options: device_map=%s, dtype=%s, attn_implementation=%s",
            self._device,
            self._dtype,
            kwargs.get("attn_implementation", "default"),
        )
        return kwargs

    def _resolve_attn_implementation(self) -> str | None:
        raw_impl = self._attn_implementation
        if raw_impl is None:
            return None
        impl = str(raw_impl).strip().lower()
        if impl in ("", "none"):
            return None
        if impl == "auto":
            if self._is_cuda_device() and self._has_flash_attn():
                return "flash_attention_2"
            return "sdpa"
        if impl == "flash_attention_2":
            if not self._is_cuda_device():
                logging.warning(
                    "flash_attention_2 requested, but device_map=%s is not CUDA. Falling back to sdpa.",
                    self._device,
                )
                return "sdpa"
            if not self._has_flash_attn():
                logging.warning("flash-attn is not installed. Falling back to sdpa.")
                return "sdpa"
        return impl

    def _is_cuda_device(self) -> bool:
        return str(self._device).lower().startswith("cuda")

    @classmethod
    def _has_flash_attn(cls) -> bool:
        if not cls._flash_attn_checked:
            cls._flash_attn_available = importlib.util.find_spec("flash_attn") is not None
            cls._flash_attn_checked = True
        return cls._flash_attn_available

    def _tune_runtime(self) -> None:
        if QwenTTSEngine._runtime_tuned:
            return
        if not torch.cuda.is_available():
            QwenTTSEngine._runtime_tuned = True
            return
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception as exc:
            logging.warning("Failed to tune torch runtime: %s", exc)
        QwenTTSEngine._runtime_tuned = True

    def _check_ttl(self) -> None:
        if QwenTTSEngine._model is None:
            return
        now = time.time()
        if now - QwenTTSEngine._model_loaded_at > self._ttl_seconds:
            self._unload_model()

    def _unload_model(self) -> None:
        if QwenTTSEngine._model is not None:
            logging.info("Unloading Qwen TTS model from VRAM: %s", QwenTTSEngine._current_model_name)
            del QwenTTSEngine._model
            QwenTTSEngine._model = None
            QwenTTSEngine._model_loaded_at = 0.0
            QwenTTSEngine._current_model_name = None
            self._clear_prompt_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _clear_prompt_cache(self) -> None:
        with QwenTTSEngine._prompt_cache_lock:
            QwenTTSEngine._prompt_cache.clear()

    def _get_voice_clone_prompt(self, model, ref_audio: str, ref_text: str):
        if not hasattr(model, "create_voice_clone_prompt"):
            return None
        cache_key = f"{self._current_model_name}|{ref_audio}|{ref_text}|icl"
        with QwenTTSEngine._prompt_cache_lock:
            cached = QwenTTSEngine._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
        except Exception as e:
            logging.warning("create_voice_clone_prompt() failed, falling back: %s", e)
            return None
        with QwenTTSEngine._prompt_cache_lock:
            # Re-check to avoid stomping under contention.
            QwenTTSEngine._prompt_cache.setdefault(cache_key, prompt_items)
            return QwenTTSEngine._prompt_cache[cache_key]

    def _normalize_text(self, text: str) -> str:
        """Remove problematic characters before TTS."""
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().rstrip('.')
        text = text.replace('—', '-').replace('–', '-')
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks near max chars while preserving sentence boundaries."""
        text = self._normalize_text(text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        if not sentences:
            return [text]

        max_chars = max(1, int(self._max_text_chars))
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush_current() -> None:
            nonlocal current, current_len
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

        for sentence in sentences:
            if len(sentence) > max_chars:
                flush_current()
                words = sentence.split()
                part: list[str] = []
                part_len = 0
                for word in words:
                    candidate_len = len(word) if not part else part_len + 1 + len(word)
                    if part and candidate_len > max_chars:
                        chunks.append(" ".join(part))
                        part = [word]
                        part_len = len(word)
                    else:
                        part.append(word)
                        part_len = candidate_len
                if part:
                    chunks.append(" ".join(part))
                continue

            candidate_len = len(sentence) if not current else current_len + 1 + len(sentence)
            if current and candidate_len > max_chars:
                flush_current()
            current.append(sentence)
            current_len = len(sentence) if current_len == 0 else current_len + 1 + len(sentence)

        flush_current()
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
