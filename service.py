"""FastAPI service exposing TTS and STT endpoints."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torchaudio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from utils.config_loader import ConfigLoader
from utils.file_handler import FileHandler
from utils.llm_preprocessor import LLMPreprocessor
from utils.orpheus_engine import OrpheusTTSEngine
from utils.silero_engine import SileroTTSEngine
from utils.qwen_tts_engine import QwenTTSEngine
from stt.qwen_asr_transcriber import QwenASRTranscriber
from utils.interfaces import Config, VoiceReference


logger = logging.getLogger("tts-stt-service")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(title="TTS/STT Service", version="1.0.0")

# Cached instances
_QWEN_TTS_ENGINE = None
_QWEN_ASR_TRANSCRIBER = None


class ReferenceInput(BaseModel):
    text: str
    audio_path: str | None = None
    audio_base64: str | None = None


class TTSRequest(BaseModel):
    text: str
    references: List[ReferenceInput] | None = None
    emotion: str | None = None
    style: str | None = None
    output: str | None = None
    sample_rate: int | None = None


class TTSResponse(BaseModel):
    processed_text: str
    ogg_path: str


class STTResponse(BaseModel):
    raw_text: str
    processed_text: str
    raw_text_path: str
    processed_text_path: str


def _load_config():
    return ConfigLoader("config.yml").load()


def _normalize_path(raw: str | Path) -> Path:
    """Convert input path to POSIX when Windows-style paths are passed (e.g., from WSL clients)."""
    if isinstance(raw, Path):
        return raw
    text = str(raw)
    # Handle Windows drive paths like C:\path\to\file
    if len(text) > 2 and text[1] == ":" and ("\\" in text or "/" in text):
        drive = text[0].lower()
        tail = text[2:].lstrip("\\/")
        return Path("/mnt") / drive / tail.replace("\\", "/")
    return Path(text)


def _wsl_to_windows_path(wsl_path: Path) -> str:
    r"""Convert WSL path (/mnt/c/...) to Windows path (C:\...)."""
    path_str = str(wsl_path)
    # Check if path starts with /mnt/X/ where X is a drive letter
    if path_str.startswith("/mnt/") and len(path_str) > 6 and path_str[6] == "/":
        drive = path_str[5].upper()
        rest = path_str[7:]
        return f"{drive}:\\{rest.replace('/', '\\')}"
    # Return as-is if not a /mnt/ path
    return path_str


def _default_tts_output_path(output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"speech_{timestamp}.ogg"


def _default_stt_output_paths(output_dir: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed = Path(output_dir) / f"stt_{timestamp}.txt"
    raw = Path(output_dir) / f"stt_{timestamp}_raw.txt"
    return processed, raw


def _split_text_into_chunks(text: str, max_len: int = 400) -> list[str]:
    """Split text into sentence-ish chunks not exceeding max_len."""
    if len(text) <= max_len:
        return [text]
    # Split by sentence delimiters, keep periods/question/exclamation.
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in parts:
        if current_len + len(part) + (1 if current else 0) <= max_len:
            current.append(part)
            current_len += len(part) + (1 if current else 0)
        else:
            if current:
                chunks.append(" ".join(current).strip())
            current = [part]
            current_len = len(part)
    if current:
        chunks.append(" ".join(current).strip())
    # Fallback in case some part was too long (no punctuation).
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_len):
                final_chunks.append(chunk[i : i + max_len])
    return final_chunks


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _materialize_references(inputs: List[ReferenceInput]) -> tuple[list[VoiceReference], list[Path]]:
    references: list[VoiceReference] = []
    temp_paths: list[Path] = []
    for idx, ref in enumerate(inputs):
        if not ref.text or not ref.text.strip():
            raise HTTPException(status_code=400, detail=f"Reference #{idx + 1} text must not be empty")

        if ref.audio_path:
            audio_path = _normalize_path(ref.audio_path)
        elif ref.audio_base64:
            try:
                raw = base64.b64decode(ref.audio_base64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Reference #{idx + 1} audio_base64 is invalid") from exc
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio.write(raw)
                audio_path = Path(tmp_audio.name)
                temp_paths.append(audio_path)
        else:
            raise HTTPException(status_code=400, detail=f"Reference #{idx + 1} must include audio_path or audio_base64")
        references.append(VoiceReference(audio_path=audio_path, transcript=ref.text))
    return references, temp_paths


def _run_tts(request: TTSRequest) -> TTSResponse:
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    config = _load_config()
    file_handler = FileHandler(target_rate=request.sample_rate or config.sample_rate)
    processed_text = request.text
    if config.use_llm:
        preprocessor = LLMPreprocessor(
            module_dir=config.llm_module_dir,
            mode="tts",
            llm_settings=config.llm_settings,
        )
        processed_text = preprocessor.process(request.text)

    output_path = _normalize_path(request.output) if request.output else _default_tts_output_path(config.output_dir)
    _ensure_parent(output_path)
    emotion = request.emotion or config.default_emotion
    style = request.style or config.default_style

    engine_choice = (config.tts_engine or "orpheus").lower()
    references: list[VoiceReference] = []
    temp_paths: list[Path] = []
    ref_inputs = request.references or []

    if engine_choice == "silero":
        base_rate = config.silero_sample_rate or config.sample_rate
        sample_rate = request.sample_rate or base_rate
        tts_engine = SileroTTSEngine(
            voice=config.silero_voice,
            language=config.silero_language,
            variant=config.silero_variant,
            sample_rate=sample_rate,
            normalize=config.silero_normalize,
        )
        chunks = _split_text_into_chunks(processed_text, max_len=400)
        if len(chunks) > 1:
            logger.info("Silero chunking text into %s parts", len(chunks))
        wav_paths: list[Path] = []
        tensors: list[torch.Tensor] = []
        combined_wav: Path | None = None
        try:
            for idx, chunk in enumerate(chunks, start=1):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = Path(tmp.name)
                wav_paths.append(wav_path)
                logger.info("Silero synth chunk %s/%s, len=%s", idx, len(chunks), len(chunk))
                tts_engine.synthesize_to_wav(chunk, references, wav_path, emotion=emotion, style=style)
                tensor, sr = torchaudio.load(str(wav_path))
                if sr != sample_rate:
                    raise HTTPException(status_code=500, detail="Silero returned unexpected sample rate")
                tensors.append(tensor)
            combined = torch.cat(tensors, dim=1)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                combined_wav = Path(tmp_out.name)
            torchaudio.save(str(combined_wav), combined, sample_rate)
            file_handler.save_ogg(combined_wav, output_path)
        finally:
            for p in wav_paths:
                if p.exists():
                    p.unlink(missing_ok=True)
            if combined_wav and combined_wav.exists():
                combined_wav.unlink(missing_ok=True)
    elif engine_choice == "qwen":
        # Qwen TTS with voice cloning
        if not config.voice_clone_ref_audio or not config.voice_clone_ref_text:
            raise HTTPException(
                status_code=400,
                detail="Qwen TTS requires voice_clone_ref_audio and voice_clone_ref_text in config"
            )
        ref_audio_path = Path(config.voice_clone_ref_audio)
        ref_text_path = Path(config.voice_clone_ref_text)
        if not ref_audio_path.exists():
            raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_audio_path}")
        if not ref_text_path.exists():
            raise HTTPException(status_code=400, detail=f"Reference text not found: {ref_text_path}")

        ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        references = [VoiceReference(audio_path=ref_audio_path, transcript=ref_text)]

        global _QWEN_TTS_ENGINE
        if _QWEN_TTS_ENGINE is None:
            _QWEN_TTS_ENGINE = QwenTTSEngine(
                model_name=config.qwen_tts_model,
                device=config.qwen_tts_device,
                max_text_chars=config.qwen_tts_max_chars,
                crossfade_ms=config.qwen_tts_crossfade_ms,
                temperature=config.qwen_tts_temperature,
                top_p=config.qwen_tts_top_p,
                repetition_penalty=config.qwen_tts_repetition_penalty,
                ttl_seconds=config.model_ttl_seconds,
            )

        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)
            _QWEN_TTS_ENGINE.synthesize_to_wav(processed_text, references, wav_path)
            file_handler.save_ogg(wav_path, output_path)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)
    else:
        if not ref_inputs:
            raise HTTPException(status_code=400, detail="At least one reference is required")
        sample_rate = request.sample_rate or config.sample_rate
        references, temp_paths = _materialize_references(ref_inputs)
        tts_engine = OrpheusTTSEngine(
            model_name=config.orpheus_model,
            codec_name=config.orpheus_codec,
            device=config.orpheus_device,
            dtype=config.orpheus_dtype,
            sample_rate=sample_rate,
            max_reference_seconds=config.max_reference_seconds,
        )

        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)

            tts_engine.synthesize_to_wav(processed_text, references, wav_path, emotion=emotion, style=style)
            file_handler.save_ogg(wav_path, output_path)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)
            for path in temp_paths:
                if path.exists():
                    path.unlink(missing_ok=True)

    logger.info("TTS done, saved: %s", output_path)
    return TTSResponse(processed_text=processed_text, ogg_path=_wsl_to_windows_path(output_path))


def _run_stt(upload_path: Path, context: str | None = None) -> STTResponse:
    config = _load_config()
    processed_txt_path, raw_txt_path = _default_stt_output_paths(config.output_dir)
    _ensure_parent(processed_txt_path)
    _ensure_parent(raw_txt_path)

    stt_engine = (config.stt_engine or "gigaam").lower()
    stt_start = time.perf_counter()

    # Get transcriber
    global _QWEN_ASR_TRANSCRIBER
    if stt_engine == "qwen":
        if _QWEN_ASR_TRANSCRIBER is None:
            _QWEN_ASR_TRANSCRIBER = QwenASRTranscriber(
                model_name=config.qwen_asr_model,
                device=config.qwen_asr_device,
                max_audio_seconds=config.qwen_asr_max_audio_seconds,
                ttl_seconds=config.model_ttl_seconds,
            )
        raw_text = _QWEN_ASR_TRANSCRIBER.transcribe(upload_path)
    else:
        from stt.gigaam_transcriber import GigaAMTranscriber
        transcriber = GigaAMTranscriber(model_name=config.stt_model, device=config.stt_device)
        raw_text = transcriber.transcribe(upload_path)

    stt_elapsed = time.perf_counter() - stt_start
    logger.info("STT transcription: %.3fs", stt_elapsed)

    # LLM postprocessing
    processed_text = raw_text
    if config.use_llm:
        llm_start = time.perf_counter()
        postprocessor = LLMPreprocessor(
            module_dir=config.llm_module_dir,
            mode="stt",
            llm_settings=config.llm_settings,
        )
        processed_text = postprocessor.process(raw_text, context=context)
        llm_elapsed = time.perf_counter() - llm_start
        logger.info("STT LLM postprocessing: %.3fs", llm_elapsed)

    raw_txt_path.write_text(raw_text, encoding="utf-8")
    processed_txt_path.write_text(processed_text, encoding="utf-8")

    logger.info("STT done, saved: %s (raw) %s (processed)", raw_txt_path, processed_txt_path)
    return STTResponse(
        raw_text=raw_text,
        processed_text=processed_text,
        raw_text_path=_wsl_to_windows_path(raw_txt_path),
        processed_text_path=_wsl_to_windows_path(processed_txt_path),
    )



@app.post("/tts", response_model=TTSResponse)
async def synthesize(request: TTSRequest) -> TTSResponse:
    return await asyncio.to_thread(_run_tts, request)


@app.post("/stt", response_model=STTResponse)
async def transcribe(
    file: UploadFile = File(...),
    context: str | None = Form(None),
) -> STTResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")
    req_start = time.perf_counter()
    context_len = len(context) if context else 0
    logger.info("STT request context length: %s", context_len)
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        read_start = time.perf_counter()
        content = await file.read()
        read_elapsed = time.perf_counter() - read_start
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        write_start = time.perf_counter()
        tmp.write(content)
        write_elapsed = time.perf_counter() - write_start
        logger.info(
            "STT upload: read=%.3fs write=%.3fs size=%d bytes",
            read_elapsed,
            write_elapsed,
            len(content),
        )

    try:
        result = await asyncio.to_thread(_run_stt, tmp_path, context)
        total_elapsed = time.perf_counter() - req_start
        logger.info("STT request total: %.3fs", total_elapsed)
        return result
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _warmup_engines() -> None:
    """Placeholder warmup hook (Orpheus loads on demand)."""
    logger.info("Warmup skipped: Orpheus loads on first request")


@app.on_event("startup")
async def on_startup() -> None:
    await asyncio.to_thread(_warmup_engines)
