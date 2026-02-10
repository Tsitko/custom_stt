#!/usr/bin/env python3
"""
Полный пайплайн: распознавание 1945_9_maya.mp3 → синтез голосом Сталина.
- Разбивка на чанки для ASR и TTS
- LLM постобработка для STT (исправление ошибок)
- LLM предобработка для TTS (транскрипция английских слов)
- Замеры времени
- TTL 5 минут для выгрузки моделей из VRAM
- Crossfade между чанками для устранения щелчков
"""
import gc
import os
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Отключить прогресс-бары и предупреждения
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"  # Работать оффлайн
warnings.filterwarnings("ignore")

# Добавить корень проекта в path для импорта utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import soundfile as sf
import numpy as np

# Отключить tqdm глобально
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# === Конфигурация ===
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_TTL_SECONDS = 300  # 5 минут

# Ограничения
ASR_MAX_AUDIO_SECONDS = 30  # Максимум секунд на чанк для ASR
TTS_MAX_TEXT_CHARS = 200    # Максимум символов на чанк для TTS
CROSSFADE_MS = 50           # Длина crossfade между чанками (мс)

# Параметры генерации TTS
TTS_TEMPERATURE = 0.3       # Ниже = более стабильный/похожий голос
TTS_TOP_P = 0.9
TTS_REPETITION_PENALTY = 1.1

# LLM обработка
USE_LLM = True  # Включить LLM пред/постобработку

# Файлы
INPUT_AUDIO = PROJECT_ROOT / "1945_9_maya.mp3"
REF_AUDIO = PROJECT_ROOT / "one_shot/stalin_reference.wav"
REF_TEXT_FILE = PROJECT_ROOT / "one_shot/stalin_reference.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config.yml"


@dataclass
class ModelCache:
    """Кэш моделей с TTL."""
    asr_model: Optional[object] = None
    asr_loaded_at: float = 0.0
    tts_model: Optional[object] = None
    tts_loaded_at: float = 0.0


_cache = ModelCache()
_llm_config = None


def load_llm_config() -> dict:
    """Загрузить конфигурацию LLM из config.yml."""
    global _llm_config
    if _llm_config is not None:
        return _llm_config

    from utils.config_loader import ConfigLoader
    try:
        config = ConfigLoader(CONFIG_PATH).load()
        _llm_config = {
            "use_llm_tts": config.use_llm_tts,
            "use_llm_stt": config.use_llm_stt,
            "llm_module_dir": config.llm_module_dir,
            "llm_settings": config.llm_settings,
        }
    except Exception as exc:
        print(f"[LLM] Не удалось загрузить конфиг: {exc}")
        _llm_config = {
            "use_llm_tts": False,
            "use_llm_stt": False,
            "llm_module_dir": None,
            "llm_settings": {},
        }

    return _llm_config


def process_with_llm(text: str, mode: str) -> str:
    """Обработать текст через LLM (mode: 'stt' или 'tts')."""
    if not USE_LLM:
        return text

    config = load_llm_config()
    allow_key = "use_llm_tts" if mode == "tts" else "use_llm_stt"
    if not config.get(allow_key, False):
        print(f"[LLM] Отключен в конфиге, пропускаем {mode}")
        return text

    try:
        from utils.llm_preprocessor import LLMPreprocessor

        start = time.perf_counter()
        preprocessor = LLMPreprocessor(
            module_dir=config.get("llm_module_dir"),
            mode=mode,
            llm_settings=config.get("llm_settings", {}),
        )
        result = preprocessor.process(text)
        elapsed = time.perf_counter() - start
        print(f"[LLM] {mode.upper()} обработка: {elapsed:.1f}s, {len(text)} → {len(result)} символов")
        return result
    except Exception as exc:
        print(f"[LLM] Ошибка {mode}: {exc}, используем исходный текст")
        return text


def unload_model(model_type: str) -> None:
    """Выгрузить модель из VRAM."""
    global _cache
    if model_type == "asr" and _cache.asr_model is not None:
        print(f"[TTL] Выгрузка ASR модели из VRAM...")
        del _cache.asr_model
        _cache.asr_model = None
        _cache.asr_loaded_at = 0.0
        gc.collect()
        torch.cuda.empty_cache()
    elif model_type == "tts" and _cache.tts_model is not None:
        print(f"[TTL] Выгрузка TTS модели из VRAM...")
        del _cache.tts_model
        _cache.tts_model = None
        _cache.tts_loaded_at = 0.0
        gc.collect()
        torch.cuda.empty_cache()


def check_ttl() -> None:
    """Проверить TTL и выгрузить устаревшие модели."""
    now = time.time()
    if _cache.asr_model and (now - _cache.asr_loaded_at > MODEL_TTL_SECONDS):
        unload_model("asr")
    if _cache.tts_model and (now - _cache.tts_loaded_at > MODEL_TTL_SECONDS):
        unload_model("tts")


def get_asr_model():
    """Получить ASR модель (загрузить если нужно)."""
    global _cache
    check_ttl()

    if _cache.asr_model is None:
        print(f"[ASR] Загрузка модели {ASR_MODEL}...")
        start = time.perf_counter()
        from qwen_asr import Qwen3ASRModel
        _cache.asr_model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        elapsed = time.perf_counter() - start
        print(f"[ASR] Модель загружена за {elapsed:.1f}s")

    _cache.asr_loaded_at = time.time()
    return _cache.asr_model


def get_tts_model():
    """Получить TTS модель (загрузить если нужно)."""
    global _cache
    check_ttl()

    if _cache.tts_model is None:
        print(f"[TTS] Загрузка модели {TTS_MODEL}...")
        start = time.perf_counter()
        from qwen_tts import Qwen3TTSModel
        _cache.tts_model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        elapsed = time.perf_counter() - start
        print(f"[TTS] Модель загружена за {elapsed:.1f}s")

    _cache.tts_loaded_at = time.time()
    return _cache.tts_model


def split_audio_to_chunks(audio_path: Path, max_seconds: float) -> list[tuple[np.ndarray, int]]:
    """Разбить аудио на чанки по max_seconds."""
    import librosa

    print(f"[ASR] Загрузка аудио: {audio_path}")
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"[ASR] Длительность: {duration:.1f}s, sample_rate: {sr}")

    chunk_samples = int(max_seconds * sr)
    chunks = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        chunks.append((chunk, sr))

    print(f"[ASR] Разбито на {len(chunks)} чанков по ~{max_seconds}s")
    return chunks


def crossfade_concat(chunks: list[np.ndarray], sr: int, crossfade_ms: int) -> np.ndarray:
    """Склеить аудио чанки с crossfade для устранения щелчков."""
    if not chunks:
        return np.array([])
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sr * crossfade_ms / 1000)

    result = chunks[0].copy()

    for i in range(1, len(chunks)):
        chunk = chunks[i]

        if crossfade_samples > 0 and len(result) >= crossfade_samples and len(chunk) >= crossfade_samples:
            # Создать fade out/in кривые
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)

            # Применить crossfade
            result[-crossfade_samples:] *= fade_out
            chunk[:crossfade_samples] *= fade_in

            # Наложить и склеить
            result[-crossfade_samples:] += chunk[:crossfade_samples]
            result = np.concatenate([result, chunk[crossfade_samples:]])
        else:
            # Простая склейка если чанки слишком короткие
            result = np.concatenate([result, chunk])

    return result


def normalize_text_for_tts(text: str) -> str:
    """Нормализовать текст перед TTS - убрать проблемные символы."""
    import re

    # Убрать множественные точки (многоточие -> одна точка)
    text = re.sub(r'\.{2,}', '.', text)
    # Убрать множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    # Убрать точки в конце если текст заканчивается на них (модель сама добавит паузу)
    text = text.strip().rstrip('.')
    # Заменить некоторые символы
    text = text.replace('—', '-').replace('–', '-')

    return text.strip()


def split_text_to_chunks(text: str, max_chars: int) -> list[str]:
    """Разбить текст на чанки по предложениям, не превышая max_chars."""
    import re

    # Нормализовать текст
    text = normalize_text_for_tts(text)

    # Разбить по предложениям
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Если предложение длиннее max_chars, разбить по словам
            if len(sentence) > max_chars:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk = (current_chunk + " " + word).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def transcribe_audio(audio_path: Path) -> str:
    """Распознать аудио файл."""
    import tempfile

    chunks = split_audio_to_chunks(audio_path, ASR_MAX_AUDIO_SECONDS)
    model = get_asr_model()

    all_text = []
    total_start = time.perf_counter()

    for i, (chunk_audio, sr) in enumerate(chunks, 1):
        # Сохранить чанк во временный файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(tmp_path, chunk_audio, sr)

        try:
            chunk_start = time.perf_counter()
            results = model.transcribe(audio=str(tmp_path), language="Russian")
            chunk_elapsed = time.perf_counter() - chunk_start

            text = results[0].text if results else ""
            all_text.append(text)
            print(f"[ASR] Чанк {i}/{len(chunks)}: {chunk_elapsed:.1f}s - '{text[:50]}...'")
        finally:
            tmp_path.unlink(missing_ok=True)

    total_elapsed = time.perf_counter() - total_start
    full_text = " ".join(all_text)
    print(f"[ASR] Всего: {total_elapsed:.1f}s, {len(full_text)} символов")

    return full_text


def synthesize_text(text: str, ref_audio: Path, ref_text: str) -> Path:
    """Синтезировать текст голосом из референса."""
    chunks = split_text_to_chunks(text, TTS_MAX_TEXT_CHARS)
    print(f"[TTS] Текст разбит на {len(chunks)} чанков")

    model = get_tts_model()

    all_wavs = []
    sample_rate = None
    total_start = time.perf_counter()

    for i, chunk_text in enumerate(chunks, 1):
        chunk_start = time.perf_counter()

        wavs, sr = model.generate_voice_clone(
            text=chunk_text,
            language="Russian",
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            temperature=TTS_TEMPERATURE,
            top_p=TTS_TOP_P,
            repetition_penalty=TTS_REPETITION_PENALTY,
        )

        chunk_elapsed = time.perf_counter() - chunk_start
        all_wavs.append(wavs[0])
        sample_rate = sr
        print(f"[TTS] Чанк {i}/{len(chunks)}: {chunk_elapsed:.1f}s - '{chunk_text[:40]}...'")

    # Объединить все чанки с crossfade
    combined = crossfade_concat(all_wavs, sample_rate, CROSSFADE_MS)
    total_elapsed = time.perf_counter() - total_start

    # Сохранить результат
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "qwen_full_pipeline.wav"
    sf.write(str(output_path), combined, sample_rate)

    duration = len(combined) / sample_rate
    print(f"[TTS] Всего: {total_elapsed:.1f}s, выход: {duration:.1f}s аудио")

    return output_path


def main():
    print("=" * 60)
    print("Полный пайплайн Qwen ASR → LLM → TTS")
    print("=" * 60)

    pipeline_start = time.perf_counter()

    # 1. Распознавание
    print("\n[1/4] РАСПОЗНАВАНИЕ (ASR)")
    print("-" * 40)
    asr_start = time.perf_counter()
    raw_text = transcribe_audio(INPUT_AUDIO)
    asr_elapsed = time.perf_counter() - asr_start

    # Сохранить сырую транскрипцию
    raw_transcript_path = OUTPUT_DIR / "qwen_transcript_raw.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_transcript_path.write_text(raw_text, encoding="utf-8")
    print(f"\nСырая транскрипция: {raw_transcript_path}")

    # 2. LLM постобработка STT
    print("\n[2/4] LLM ПОСТОБРАБОТКА (STT)")
    print("-" * 40)
    llm_stt_start = time.perf_counter()
    processed_text = process_with_llm(raw_text, mode="stt")
    llm_stt_elapsed = time.perf_counter() - llm_stt_start

    # Сохранить обработанную транскрипцию
    processed_transcript_path = OUTPUT_DIR / "qwen_transcript_processed.txt"
    processed_transcript_path.write_text(processed_text, encoding="utf-8")
    print(f"Обработанная транскрипция: {processed_transcript_path}")

    # 3. LLM предобработка TTS
    print("\n[3/4] LLM ПРЕДОБРАБОТКА (TTS)")
    print("-" * 40)
    llm_tts_start = time.perf_counter()
    tts_text = process_with_llm(processed_text, mode="tts")
    llm_tts_elapsed = time.perf_counter() - llm_tts_start

    # Сохранить текст для TTS
    tts_text_path = OUTPUT_DIR / "qwen_tts_input.txt"
    tts_text_path.write_text(tts_text, encoding="utf-8")
    print(f"Текст для TTS: {tts_text_path}")

    # 4. Синтез
    print("\n[4/4] СИНТЕЗ (TTS)")
    print("-" * 40)
    ref_text = REF_TEXT_FILE.read_text(encoding="utf-8").strip()

    tts_start = time.perf_counter()
    output_path = synthesize_text(tts_text, REF_AUDIO, ref_text)
    tts_elapsed = time.perf_counter() - tts_start

    # Итоги
    pipeline_elapsed = time.perf_counter() - pipeline_start
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    print(f"ASR время:          {asr_elapsed:.1f}s")
    print(f"LLM STT время:      {llm_stt_elapsed:.1f}s")
    print(f"LLM TTS время:      {llm_tts_elapsed:.1f}s")
    print(f"TTS время:          {tts_elapsed:.1f}s")
    print(f"Общее время:        {pipeline_elapsed:.1f}s")
    print(f"\nФайлы:")
    print(f"  Сырая транскрипция:      {raw_transcript_path}")
    print(f"  После LLM STT:           {processed_transcript_path}")
    print(f"  Вход TTS (после LLM):    {tts_text_path}")
    print(f"  Синтезированное аудио:   {output_path}")
    print(f"\nТекст ({len(raw_text)} → {len(processed_text)} → {len(tts_text)} символов):")
    preview = tts_text[:500] + "..." if len(tts_text) > 500 else tts_text
    print(preview)


if __name__ == "__main__":
    main()
