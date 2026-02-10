#!/usr/bin/env python3
"""Тест API с кастомным голосовым референсом."""
import base64
import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8013"

def main():
    # Загружаем тестовые данные
    ref_audio_path = Path("one_shot/dt_reference.wav")
    ref_text_path = Path("one_shot/dt_reference.txt")
    test_text_path = Path("one_shot/reference_test_text.txt")

    if not ref_audio_path.exists():
        print(f"Файл референса не найден: {ref_audio_path}")
        sys.exit(1)
    if not ref_text_path.exists():
        print(f"Файл текста референса не найден: {ref_text_path}")
        sys.exit(1)
    if not test_text_path.exists():
        print(f"Файл тестового текста не найден: {test_text_path}")
        sys.exit(1)

    # Конвертируем аудио в base64
    audio_base64 = base64.b64encode(ref_audio_path.read_bytes()).decode("utf-8")
    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    test_text = test_text_path.read_text(encoding="utf-8").strip()

    print(f"Референс аудио: {len(audio_base64)} символов base64")
    print(f"Текст референса: {ref_text}")
    print(f"Текст для озвучки: {test_text}")
    print()

    # Отправляем запрос с кастомным референсом
    payload = {
        "text": test_text,
        "reference_audio_base64": audio_base64,
        "reference_text": ref_text,
    }

    print("Отправляем запрос к /tts с кастомным референсом...")
    response = requests.post(f"{BASE_URL}/tts", json=payload, timeout=120)

    if response.status_code != 200:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    print(f"Результат: {json.dumps(result, ensure_ascii=False, indent=2)}")
    print(f"\nФайл сохранен: {result['ogg_path']}")


if __name__ == "__main__":
    main()
