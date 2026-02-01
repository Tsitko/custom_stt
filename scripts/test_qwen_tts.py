#!/usr/bin/env python3
"""Тест Qwen3-TTS с клонированием голоса Сталина."""
import torch
import soundfile as sf
from pathlib import Path

print("Загрузка модели Qwen3-TTS-12Hz-1.7B-Base...")
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("Модель загружена.")

# Загрузить транскрипцию референса
ref_text_path = Path("one_shot/stalin_reference.txt")
ref_text = ref_text_path.read_text(encoding="utf-8").strip()
print(f"Референс текст: {ref_text}")

ref_audio = "one_shot/stalin_reference.wav"
print(f"Референс аудио: {ref_audio}")

# Текст для синтеза
text_to_speak = "Товарищи! Сегодня мы тестируем новую систему синтеза речи."

print(f"\nГенерация: {text_to_speak}")
wavs, sr = model.generate_voice_clone(
    text=text_to_speak,
    language="Russian",
    ref_audio=ref_audio,
    ref_text=ref_text,
    temperature=0.3,        # Ниже = более похожий голос
    top_p=0.9,
    repetition_penalty=1.1,
)

output_path = Path("outputs/qwen_tts_test.wav")
output_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(output_path), wavs[0], sr)
print(f"\nСохранено: {output_path} (sample_rate={sr})")
