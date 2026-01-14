#!/usr/bin/env python3
"""Test script for STT prompts with everyday speech examples."""

import logging
from utils.llm_preprocessor import LLMPreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def test_stt_everyday():
    """Test STT postprocessing with everyday speech recognition errors."""
    print("\n" + "="*60)
    print("STT POSTPROCESSING TESTS (Everyday Speech)")
    print("="*60)

    postprocessor = LLMPreprocessor(mode="stt")

    test_cases = [
        # Распознанные бренды
        "купил пиво гиннесс и мёрфис в икее со скидкой пятьдесят процентов",

        # Детская тематика с ошибками
        "ребёнок ходит вдетский сад на физру и в три кружок",

        # Социальные сети и технологии
        "посмотри инстаграм и ютуб на телефоне с восемь гигабайт рам",

        # Смешанная речь с числами
        "в две тысячи двадцать четвёртом году сыну пять лет пойдёт в школу номер пятнадцать",

        # Технический + бытовой контекст
        "апий сервера работает джи пи ю загружен купил молоко хлеб сыр",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\nТест {i}:")
        print(f"Вход:  {text}")
        result = postprocessor.process(text)
        print(f"Выход: {result}")


def main():
    """Run all tests."""
    print("\nТестирование STT промптов для повседневной речи")
    print("Проверяем постобработку распознанного текста...")

    test_stt_everyday()

    print("\n" + "="*60)
    print("Тестирование завершено!")
    print("="*60)
    print("\nПримечания:")
    print("- STT: восстанавливаются оригинальные названия брендов")
    print("- STT: исправляются ошибки распознавания в контексте")
    print("\nЕсли LLM недоступен, будет возвращён исходный текст.")


if __name__ == "__main__":
    main()