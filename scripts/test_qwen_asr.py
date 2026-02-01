#!/usr/bin/env python3
"""Тест Qwen3-ASR-1.7B на распознавание референса."""
import torch
from pathlib import Path

print("Загрузка модели Qwen3-ASR-1.7B...")
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("Модель загружена.")

audio_path = Path("one_shot/stalin_reference.wav")
print(f"Транскрибирование: {audio_path}")

results = model.transcribe(
    audio=str(audio_path),
    language="Russian",
)

text = results[0].text
print(f"\nТранскрипция: {text}")

# Сохранить для TTS
output_path = Path("one_shot/stalin_reference_asr.txt")
output_path.write_text(text, encoding="utf-8")
print(f"Сохранено: {output_path}")
