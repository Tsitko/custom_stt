"""Orpheus TTS engine using transformers + SNAC codec with reference audio support."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.interfaces import ITTSEngine, VoiceReference

# SNAC audio token layout (3 levels, 7 tokens per base frame: 1 + 2 + 4)
AUDIO_OFFSET = 128256  # <custom_token_0>
STRIDE = 4096  # codebook size per SNAC level
NUM_LEVELS = 3
DEFAULT_AUDIO_TOKEN = "<|audio|>"


@dataclass
class _LoadedBackend:
    model: any
    tokenizer: any
    codec: any
    device: torch.device


class OrpheusTTSEngine(ITTSEngine):
    """Loads Orpheus (LLaMA-style) checkpoint and SNAC codec for zero-shot voice cloning."""

    def __init__(
        self,
        model_name: str,
        codec_name: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        sample_rate: int = 24000,
        max_reference_seconds: float = 20.0,
        voice: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        self._model_name = model_name
        self._codec_name = codec_name
        self._device_pref = device
        self._dtype = dtype
        self._sample_rate = sample_rate
        self._max_ref_seconds = max_reference_seconds
        self._voice = voice or "dan"
        self._backend: _LoadedBackend | None = None

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

        backend = self._load_backend()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_text = self._apply_tags(text, emotion, style, backend.tokenizer)

        # Prepare conditioning tokens: optional references + target prompt.
        prompt_ids: list[int] = []
        if backend.codec and references:
            for ref in references:
                prompt_ids += self._encode_reference_tokens(ref, backend)
        prompt_ids += backend.tokenizer.encode(target_text, add_special_tokens=False)

        input_ids = torch.tensor([prompt_ids], device=backend.device)
        logging.info("Generating audio tokens (input len=%s)", input_ids.shape[-1])

        with torch.inference_mode():
            generated = backend.model.generate(
                input_ids,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_new_tokens=4096,
            )
        generated_ids = generated[0].tolist()[len(prompt_ids) :]

        audio_np: np.ndarray
        if backend.codec:
            snac_codes = self._tokens_to_snac_codes(generated_ids)
            if snac_codes is not None:
                try:
                    audio_hat = backend.codec.decode(snac_codes)
                    audio_np = audio_hat[0].cpu().numpy()
                except Exception as exc:
                    logging.warning("SNAC decode failed (%s), writing silence", exc)
                    audio_np = np.zeros(int(self._sample_rate * 0.5), dtype=np.float32)
            else:
                logging.warning("No audio tokens decoded; writing silence")
                audio_np = np.zeros(int(self._sample_rate * 0.5), dtype=np.float32)
        else:
            logging.warning("SNAC codec not available; writing silence placeholder")
            audio_np = np.zeros(int(self._sample_rate * 0.5), dtype=np.float32)

        sf.write(output_path, audio_np, self._sample_rate)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Synthesis failed: output WAV was not created")
        return output_path

    def _load_backend(self) -> _LoadedBackend:
        if self._backend:
            return self._backend

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        logging.info(
            "Loading Orpheus LLM via transformers: model=%s device=%s dtype=%s codec=%s",
            self._model_name,
            device,
            dtype,
            self._codec_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        codec = None
        if self._codec_name:
            try:
                from snac import SNAC  # type: ignore

                codec = SNAC.from_pretrained(self._codec_name).to(device)
                codec.eval()
            except Exception as exc:
                logging.warning("SNAC codec load failed (%s); proceeding without codec", exc)

        self._backend = _LoadedBackend(model=model, tokenizer=tokenizer, codec=codec, device=device)
        return self._backend

    def _encode_reference_tokens(self, reference: VoiceReference, backend: _LoadedBackend) -> list[int]:
        audio_path = Path(reference.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        if not reference.transcript.strip():
            raise ValueError("Reference transcript must not be empty")
        if backend.codec is None:
            return []

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self._sample_rate:
            logging.info("Resampling reference from %s Hz to %s Hz", sr, self._sample_rate)
            try:
                import torchaudio  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "torchaudio is required for resampling. Install it or provide reference audio at the target sample rate."
                ) from exc
            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            audio = torchaudio.functional.resample(tensor, sr, self._sample_rate).squeeze(0).numpy()

        duration = len(audio) / float(self._sample_rate)
        if duration > self._max_ref_seconds:
            raise ValueError(f"Reference audio is too long ({duration:.2f}s), expected <= {self._max_ref_seconds}s")

        waveform = torch.tensor(audio, dtype=torch.float32, device=backend.device).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            codes = backend.codec.encode(waveform)
        audio_tokens = self._snac_codes_to_tokens(codes)
        text_tokens = backend.tokenizer.encode(reference.transcript, add_special_tokens=False)
        return text_tokens + audio_tokens

    def _snac_codes_to_tokens(self, codes) -> list[int]:
        """Map SNAC codes to LLM token ids using the 1:2:4 layout (7 tokens per frame)."""
        if len(codes) != NUM_LEVELS:
            logging.warning("Unexpected SNAC levels: %s (expected %s)", len(codes), NUM_LEVELS)
            return []

        # Level lengths follow 1:2:4 ratio (e.g., [12, 24, 48] for 1s of audio)
        len_l0, len_l1, len_l2 = [c.shape[1] for c in codes]
        frames = min(len_l0, len_l1 // 2, len_l2 // 4)
        tokens: list[int] = []
        for t in range(frames):
            # 1 token from level 0
            tokens.append(int(codes[0][0][t].item()) + AUDIO_OFFSET + 0 * STRIDE)
            # 2 tokens from level 1
            tokens.append(int(codes[1][0][2 * t].item()) + AUDIO_OFFSET + 1 * STRIDE)
            tokens.append(int(codes[1][0][2 * t + 1].item()) + AUDIO_OFFSET + 4 * STRIDE)
            # 4 tokens from level 2
            base2 = 4 * t
            tokens.append(int(codes[2][0][base2].item()) + AUDIO_OFFSET + 2 * STRIDE)
            tokens.append(int(codes[2][0][base2 + 1].item()) + AUDIO_OFFSET + 3 * STRIDE)
            tokens.append(int(codes[2][0][base2 + 2].item()) + AUDIO_OFFSET + 5 * STRIDE)
            tokens.append(int(codes[2][0][base2 + 3].item()) + AUDIO_OFFSET + 6 * STRIDE)
        return tokens

    def _tokens_to_snac_codes(self, token_ids: list[int]):
        """Extract SNAC codebooks from generated token ids (expects 7-token frames)."""
        audio_tokens = [tid for tid in token_ids if tid >= AUDIO_OFFSET]
        if not audio_tokens:
            return None

        level0: list[int] = []
        level1: list[int] = []
        level2: list[int] = []

        # Keep original order to respect time steps.
        for tid in audio_tokens:
            level = (tid - AUDIO_OFFSET) // STRIDE
            code = tid - AUDIO_OFFSET - level * STRIDE
            if level == 0:
                level0.append(code)
            elif level in (1, 4):  # level 1 tokens occupy two sub-bands
                level1.append(code)
            elif level in (2, 3, 5, 6):  # level 2 tokens occupy four sub-bands
                level2.append(code)

        frames = min(len(level0), len(level1) // 2, len(level2) // 4)
        if frames == 0:
            return None

        def to_tensor(values, stride_count):
            return torch.tensor(values[: stride_count * frames], dtype=torch.int32).view(1, stride_count, frames)

        l0 = to_tensor(level0, 1)
        l1 = to_tensor(level1, 2)
        l2 = to_tensor(level2, 4)

        # SNAC expects shape [B, T] per level -> squeeze channel dimension.
        return [l0.view(1, frames), l1.view(1, frames * 2), l2.view(1, frames * 4)]

    def _resolve_device(self) -> torch.device:
        if self._device_pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_pref)

    def _resolve_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if self._dtype.lower() not in dtype_map:
            raise ValueError(f"Unsupported dtype for Orpheus: {self._dtype}")
        return dtype_map[self._dtype.lower()]

    def _apply_tags(self, text: str, emotion: str | None, style: str | None, tokenizer) -> str:
        # Build prompt with BOS, audio token, voice prefix, optional tags, and EOS.
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        audio_token = DEFAULT_AUDIO_TOKEN
        specials = getattr(tokenizer, "additional_special_tokens", []) or []
        for v in specials:
            if "audio" in str(v):
                audio_token = str(v)
                break

        prefix = f"{bos}{audio_token}"
        if self._voice:
            prefix += f"{self._voice}: "
        if emotion:
            prefix += f"<emotion:{emotion}>"
        if style:
            prefix += f"<style:{style}>"
        # Seed one empty audio frame to nudge the model into audio token space.
        seed_audio = " ".join(["<custom_token_0>"] * 7)
        return prefix + text + " " + seed_audio + eos

    def _bytes_to_numpy(self, audio_bytes: bytes) -> "np.ndarray":
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32")
        if sr != self._sample_rate:
            import torchaudio

            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            data = torchaudio.functional.resample(tensor, sr, self._sample_rate).squeeze(0).numpy()
        return data
