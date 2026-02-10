"""FastAPI service exposing TTS and STT endpoints."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import signal
import tempfile
import time
import io
import wave
import threading
from datetime import datetime
from pathlib import Path
from typing import List
from contextlib import contextmanager

import torch
import torchaudio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from utils.config_loader import ConfigLoader
from utils.file_handler import FileHandler
from utils.audio_utils import compute_rms_dbfs
from utils.llm_preprocessor import LLMPreprocessor
from utils.orpheus_engine import OrpheusTTSEngine
from utils.silero_engine import SileroTTSEngine
from utils.qwen_tts_engine import QwenTTSEngine
from utils.stress_processor import StressProcessor
from stt.qwen_asr_transcriber import QwenASRTranscriber
from utils.interfaces import Config, VoiceReference


logger = logging.getLogger("tts-stt-service")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(title="TTS/STT Service", version="1.0.0")

# Cached instances
_QWEN_TTS_ENGINE = None
_QWEN_ASR_TRANSCRIBER = None
_STRESS_PROCESSOR = None
_FORCE_EXITING = False
_INFER_LOCK = threading.Lock()


class ReferenceInput(BaseModel):
    text: str = Field(..., description="Transcript for the reference audio.")
    audio_path: str | None = Field(default=None, description="Local filesystem path to reference audio file.")
    audio_base64: str | None = Field(default=None, description="Base64-encoded reference audio (WAV/compatible).")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize.")
    references: List[ReferenceInput] | None = Field(default=None, description="Reference list (mainly for Orpheus).")
    emotion: str | None = Field(default=None, description="Optional emotion hint.")
    style: str | None = Field(default=None, description="Optional style hint.")
    output: str | None = Field(default=None, description="Output path (used by /tts OGG endpoint).")
    sample_rate: int | None = Field(default=None, description="Target sample rate override.")
    # Кастомный референс для voice cloning (Qwen TTS)
    reference_audio_base64: str | None = Field(default=None, description="Custom base64 reference audio for Qwen voice cloning.")
    reference_text: str | None = Field(default=None, description="Transcript for custom reference audio.")
    # Описание голоса для voice design (Qwen TTS)
    voice_description: str | None = Field(default=None, description="Voice design prompt for Qwen voice design mode.")


class TTSResponse(BaseModel):
    processed_text: str
    ogg_path: str


class STTResponse(BaseModel):
    raw_text: str
    processed_text: str
    raw_text_path: str
    processed_text_path: str


class TTSStreamingProtocol(BaseModel):
    websocket_path: str
    notes: list[str]
    client_messages: list[dict]
    server_messages: list[dict]


def _load_config():
    return ConfigLoader("config.yml").load()


@contextmanager
def _exclusive_inference_slot(operation: str):
    wait_start = time.perf_counter()
    _INFER_LOCK.acquire()
    waited = time.perf_counter() - wait_start
    if waited > 0.01:
        logger.info("%s waited %.3fs for inference slot", operation, waited)
    try:
        yield
    finally:
        _INFER_LOCK.release()


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
    with _exclusive_inference_slot("TTS"):
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

        config = _load_config()
        file_handler = FileHandler(target_rate=request.sample_rate or config.sample_rate)
        processed_text = request.text

        # 1. LLM preprocessing (цифры → слова, нормализация)
        if config.use_llm_tts:
            preprocessor = LLMPreprocessor(
                module_dir=config.llm_module_dir,
                mode="tts",
                llm_settings=config.llm_settings,
            )
            processed_text = preprocessor.process(request.text)

        # 2. Stress processing (расстановка ударений)
        if config.use_stress:
            global _STRESS_PROCESSOR
            if _STRESS_PROCESSOR is None:
                _STRESS_PROCESSOR = StressProcessor()
            if _STRESS_PROCESSOR.is_available:
                stress_start = time.perf_counter()
                processed_text = _STRESS_PROCESSOR.process(processed_text, stress_format="apostrophe")
                stress_elapsed = time.perf_counter() - stress_start
                logger.info("Stress processing: %.3fs", stress_elapsed)

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
            # Qwen TTS - voice cloning или voice design
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
                    attn_implementation=config.qwen_tts_attn_implementation,
                    ttl_seconds=config.model_ttl_seconds,
                )

            wav_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = Path(tmp.name)

                # Приоритет: кастомный референс > voice_description > дефолтный референс
                if request.reference_audio_base64 and request.reference_text:
                    # Voice cloning с кастомным референсом
                    try:
                        raw_audio = base64.b64decode(request.reference_audio_base64)
                    except Exception as exc:
                        raise HTTPException(status_code=400, detail="reference_audio_base64 is invalid") from exc
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                        tmp_audio.write(raw_audio)
                        ref_audio_path = Path(tmp_audio.name)
                        temp_paths.append(ref_audio_path)
                    ref_text = request.reference_text.strip()
                    references = [VoiceReference(audio_path=ref_audio_path, transcript=ref_text)]
                    logger.info("Using custom voice reference from request")
                    _QWEN_TTS_ENGINE.synthesize_to_wav(processed_text, references, wav_path)
                elif request.voice_description:
                    # Voice design по описанию
                    logger.info("Using voice design with description")
                    _QWEN_TTS_ENGINE.synthesize_from_description(
                        processed_text, request.voice_description.strip(), wav_path
                    )
                else:
                    # Voice cloning с дефолтным референсом из конфига
                    if not config.voice_clone_ref_audio or not config.voice_clone_ref_text:
                        raise HTTPException(
                            status_code=400,
                            detail="Qwen TTS requires reference_audio_base64+reference_text, voice_description, or config defaults"
                        )
                    ref_audio_path = Path(config.voice_clone_ref_audio)
                    ref_text_path = Path(config.voice_clone_ref_text)
                    if not ref_audio_path.exists():
                        raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_audio_path}")
                    if not ref_text_path.exists():
                        raise HTTPException(status_code=400, detail=f"Reference text not found: {ref_text_path}")
                    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
                    references = [VoiceReference(audio_path=ref_audio_path, transcript=ref_text)]
                    logger.info("Using default voice reference from config")
                    _QWEN_TTS_ENGINE.synthesize_to_wav(processed_text, references, wav_path)

                file_handler.save_ogg(wav_path, output_path)
            finally:
                if wav_path and wav_path.exists():
                    wav_path.unlink(missing_ok=True)
                for path in temp_paths:
                    if path.exists():
                        path.unlink(missing_ok=True)
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


def _get_qwen_tts_engine(config: Config) -> QwenTTSEngine:
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
            attn_implementation=config.qwen_tts_attn_implementation,
            ttl_seconds=config.model_ttl_seconds,
        )
    return _QWEN_TTS_ENGINE


def _preprocess_tts_text(text: str, config: Config) -> str:
    processed_text = text

    if config.use_llm_tts:
        preprocessor = LLMPreprocessor(
            module_dir=config.llm_module_dir,
            mode="tts",
            llm_settings=config.llm_settings,
        )
        processed_text = preprocessor.process(text)

    if config.use_stress:
        global _STRESS_PROCESSOR
        if _STRESS_PROCESSOR is None:
            _STRESS_PROCESSOR = StressProcessor()
        if _STRESS_PROCESSOR.is_available:
            processed_text = _STRESS_PROCESSOR.process(processed_text, stress_format="apostrophe")

    return processed_text


def _run_qwen_tts_to_wav(
    text: str,
    *,
    reference_audio_base64: str | None = None,
    reference_text: str | None = None,
    voice_description: str | None = None,
) -> tuple[str, bytes, int]:
    with _exclusive_inference_slot("TTS-WAV"):
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

        config = _load_config()
        if (config.tts_engine or "").lower() != "qwen":
            raise HTTPException(status_code=400, detail="This endpoint currently supports only tts_engine=qwen")

        processed_text = _preprocess_tts_text(text, config)
        qwen_engine = _get_qwen_tts_engine(config)

        temp_paths: list[Path] = []
        wav_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)

            if reference_audio_base64 and reference_text:
                try:
                    raw_audio = base64.b64decode(reference_audio_base64)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="reference_audio_base64 is invalid") from exc
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    tmp_audio.write(raw_audio)
                    ref_audio_path = Path(tmp_audio.name)
                    temp_paths.append(ref_audio_path)
                references = [VoiceReference(audio_path=ref_audio_path, transcript=reference_text.strip())]
                qwen_engine.synthesize_to_wav(processed_text, references, wav_path)
            elif voice_description and voice_description.strip():
                qwen_engine.synthesize_from_description(processed_text, voice_description.strip(), wav_path)
            else:
                if not config.voice_clone_ref_audio or not config.voice_clone_ref_text:
                    raise HTTPException(
                        status_code=400,
                        detail="Qwen TTS requires reference_audio_base64+reference_text, voice_description, or config defaults",
                    )
                ref_audio_path = Path(config.voice_clone_ref_audio)
                ref_text_path = Path(config.voice_clone_ref_text)
                if not ref_audio_path.exists():
                    raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_audio_path}")
                if not ref_text_path.exists():
                    raise HTTPException(status_code=400, detail=f"Reference text not found: {ref_text_path}")
                ref_text = ref_text_path.read_text(encoding="utf-8").strip()
                references = [VoiceReference(audio_path=ref_audio_path, transcript=ref_text)]
                qwen_engine.synthesize_to_wav(processed_text, references, wav_path)

            wav_bytes = wav_path.read_bytes()
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_info:
                sample_rate = wav_info.getframerate()
            return processed_text, wav_bytes, sample_rate
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)
            for path in temp_paths:
                if path.exists():
                    path.unlink(missing_ok=True)


def _run_stt(upload_path: Path, context: str | None = None) -> STTResponse:
    with _exclusive_inference_slot("STT"):
        config = _load_config()
        processed_txt_path, raw_txt_path = _default_stt_output_paths(config.output_dir)
        _ensure_parent(processed_txt_path)
        _ensure_parent(raw_txt_path)

        rms_db = compute_rms_dbfs(upload_path)
        if rms_db is not None:
            logger.info("STT RMS level: %.2f dBFS (threshold %.2f)", rms_db, config.stt_silence_db)
            if rms_db < config.stt_silence_db:
                raw_text = ""
                processed_text = ""
                raw_txt_path.write_text(raw_text, encoding="utf-8")
                processed_txt_path.write_text(processed_text, encoding="utf-8")
                logger.info("STT skipped: below silence threshold")
                return STTResponse(
                    raw_text=raw_text,
                    processed_text=processed_text,
                    raw_text_path=_wsl_to_windows_path(raw_txt_path),
                    processed_text_path=_wsl_to_windows_path(processed_txt_path),
                )

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
        if config.use_llm_stt:
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



@app.post(
    "/tts",
    response_model=TTSResponse,
    summary="Synthesize speech to OGG",
    description=(
        "Main TTS endpoint. Returns path to generated OGG file. "
        "For Qwen: provide custom reference (`reference_audio_base64` + `reference_text`) "
        "or `voice_description`, otherwise config defaults are used."
    ),
)
async def synthesize(request: TTSRequest) -> TTSResponse:
    return await asyncio.to_thread(_run_tts, request)


@app.post(
    "/tts/wav",
    summary="Synthesize speech to WAV bytes",
    description=(
        "Returns raw WAV audio directly in HTTP response body (`audio/wav`) without OGG conversion. "
        "Currently supports `tts_engine=qwen`."
    ),
    responses={
        200: {
            "description": "WAV audio stream in response body.",
            "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
        },
        400: {"description": "Bad request (validation or missing references)."},
        500: {"description": "Internal synthesis error."},
    },
)
async def synthesize_wav(request: TTSRequest) -> Response:
    processed_text, wav_bytes, sample_rate = await asyncio.to_thread(
        _run_qwen_tts_to_wav,
        request.text,
        reference_audio_base64=request.reference_audio_base64,
        reference_text=request.reference_text,
        voice_description=request.voice_description,
    )
    headers = {
        "X-Processed-Text-Length": str(len(processed_text)),
        "X-Sample-Rate": str(sample_rate),
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.get(
    "/tts/speak-streaming/protocol",
    response_model=TTSStreamingProtocol,
    summary="WebSocket streaming protocol for TTS",
    description=(
        "OpenAPI helper endpoint describing how to integrate with `WS /tts/speak-streaming`. "
        "WebSocket routes are not fully represented in Swagger UI, so use this schema as integration contract."
    ),
)
async def tts_speak_streaming_protocol() -> TTSStreamingProtocol:
    return TTSStreamingProtocol(
        websocket_path="/tts/speak-streaming",
        notes=[
            "Client sends JSON control messages over WebSocket text frames.",
            "Server returns binary WAV chunks and JSON status events.",
            "Audio is sent only after Flush/Close when buffer is not empty.",
        ],
        client_messages=[
            {"type": "Speak", "required": ["type", "text"], "example": {"type": "Speak", "text": "Привет, ребята"}},
            {"type": "Flush", "required": ["type"], "example": {"type": "Flush"}},
            {"type": "Clear", "required": ["type"], "example": {"type": "Clear"}},
            {"type": "Close", "required": ["type"], "example": {"type": "Close"}},
        ],
        server_messages=[
            {"type": "Metadata", "payload": ["request_id", "model_name", "attn_implementation"]},
            {"type": "Flushed", "payload": ["sequence_id", "sample_rate", "chars", "elapsed_s", "bytes"]},
            {"type": "Cleared", "payload": ["sequence_id"]},
            {"type": "Warning", "payload": ["code", "description"]},
            {"type": "Binary", "payload": "WAV bytes chunks"},
        ],
    )


@app.websocket("/tts/speak-streaming")
async def speak_streaming(websocket: WebSocket) -> None:
    await websocket.accept()
    sequence_id = 0
    text_buffer: list[str] = []

    try:
        config = _load_config()
        if (config.tts_engine or "").lower() != "qwen":
            await websocket.send_json(
                {
                    "type": "Warning",
                    "code": "UNSUPPORTED_ENGINE",
                    "description": "Streaming endpoint currently supports only tts_engine=qwen",
                }
            )
            await websocket.close(code=1003)
            return

        await websocket.send_json(
            {
                "type": "Metadata",
                "request_id": f"local-{int(time.time() * 1000)}",
                "model_name": config.qwen_tts_model,
                "attn_implementation": config.qwen_tts_attn_implementation,
            }
        )

        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "Warning",
                        "code": "BAD_JSON",
                        "description": "Message must be valid JSON",
                    }
                )
                continue

            msg_type = str(message.get("type", "")).strip().lower()
            if msg_type == "speak":
                text = str(message.get("text", "")).strip()
                if not text:
                    await websocket.send_json(
                        {
                            "type": "Warning",
                            "code": "EMPTY_TEXT",
                            "description": "Speak message has empty text",
                        }
                    )
                    continue
                text_buffer.append(text)
                continue

            if msg_type == "clear":
                text_buffer.clear()
                sequence_id += 1
                await websocket.send_json({"type": "Cleared", "sequence_id": sequence_id})
                continue

            if msg_type in ("flush", "close"):
                if text_buffer:
                    merged_text = " ".join(text_buffer).strip()
                    text_buffer.clear()
                    start = time.perf_counter()
                    try:
                        processed_text, wav_bytes, sample_rate = await asyncio.to_thread(
                            _run_qwen_tts_to_wav,
                            merged_text,
                            voice_description=message.get("voice_description"),
                        )
                    except Exception as exc:
                        await websocket.send_json(
                            {
                                "type": "Warning",
                                "code": "SYNTH_ERROR",
                                "description": str(exc),
                            }
                        )
                        if msg_type == "close":
                            await websocket.close(code=1011)
                            return
                        continue

                    chunk_size = 16384
                    for offset in range(0, len(wav_bytes), chunk_size):
                        await websocket.send_bytes(wav_bytes[offset : offset + chunk_size])

                    elapsed = time.perf_counter() - start
                    sequence_id += 1
                    await websocket.send_json(
                        {
                            "type": "Flushed",
                            "sequence_id": sequence_id,
                            "sample_rate": sample_rate,
                            "chars": len(processed_text),
                            "elapsed_s": round(elapsed, 3),
                            "bytes": len(wav_bytes),
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "Warning",
                            "code": "EMPTY_BUFFER",
                            "description": "No buffered text to flush",
                        }
                    )

                if msg_type == "close":
                    await websocket.close(code=1000)
                    return
                continue

            await websocket.send_json(
                {
                    "type": "Warning",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "description": f"Unsupported message type: {message.get('type')}",
                }
            )
    except WebSocketDisconnect:
        return


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
    _register_signal_handlers()


def _unload_engines() -> None:
    """Best-effort cleanup to release RAM/VRAM on shutdown."""
    global _QWEN_TTS_ENGINE, _QWEN_ASR_TRANSCRIBER
    if _QWEN_TTS_ENGINE is not None:
        try:
            _QWEN_TTS_ENGINE._unload_model()
        except Exception as exc:
            logger.warning("Failed to unload Qwen TTS model: %s", exc)
        _QWEN_TTS_ENGINE = None
    if _QWEN_ASR_TRANSCRIBER is not None:
        try:
            _QWEN_ASR_TRANSCRIBER._unload_model()
        except Exception as exc:
            logger.warning("Failed to unload Qwen ASR model: %s", exc)
        _QWEN_ASR_TRANSCRIBER = None


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await asyncio.to_thread(_unload_engines)


def _handle_terminate(signum, _frame) -> None:
    global _FORCE_EXITING
    if _FORCE_EXITING:
        return
    _FORCE_EXITING = True
    logger.warning("Received signal %s, forcing shutdown", signum)
    try:
        _unload_engines()
    finally:
        os._exit(0)


def _register_signal_handlers() -> None:
    try:
        signal.signal(signal.SIGTERM, _handle_terminate)
        signal.signal(signal.SIGINT, _handle_terminate)
    except Exception as exc:
        logger.warning("Failed to register signal handlers: %s", exc)
