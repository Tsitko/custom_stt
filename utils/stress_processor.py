"""Процессор для расстановки ударений в русском тексте."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class StressProcessor:
    """Расстановка ударений через RUAccent."""

    _instance = None
    _accentizer = None

    def __new__(cls):
        """Singleton для переиспользования загруженной модели."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if StressProcessor._accentizer is not None:
            return

        try:
            from ruaccent import RUAccent

            logger.info("Loading RUAccent model (turbo)...")
            StressProcessor._accentizer = RUAccent()
            StressProcessor._accentizer.load(omograph_model_size='turbo')
            logger.info("RUAccent model loaded")
        except ImportError:
            logger.warning("ruaccent not installed, stress processing disabled")
            StressProcessor._accentizer = None
        except Exception as e:
            logger.error("Failed to load RUAccent: %s", e)
            StressProcessor._accentizer = None

        # Кастомный словарь для терминов (+ ПЕРЕД ударной гласной, как ruaccent)
        self._custom_dict = {
            "api": "эйпиа+й",
            "http": "эйчтитип+и",
            "json": "джейс+он",
            "url": "юэр+эл",
            "cpu": "сипи+ю",
            "gpu": "джипи+ю",
            "ram": "рэм",
            "ssd": "эсэсд+и",
            "llm": "эльэл+эм",
            "rps": "эрпи+эс",
        }

    @property
    def is_available(self) -> bool:
        """Доступен ли процессор."""
        return StressProcessor._accentizer is not None

    def process(self, text: str, stress_format: str = "plus") -> str:
        """
        Расстановка ударений в тексте.

        Args:
            text: входной текст
            stress_format: 'plus' (+перед гласной) или 'apostrophe' ('после гласной)

        Returns:
            Текст с ударениями
        """
        if not self.is_available:
            return text

        if not text or not text.strip():
            return text

        # Применяем кастомный словарь (case-insensitive)
        processed = text
        for word, stressed_form in self._custom_dict.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            processed = re.sub(pattern, stressed_form, processed, flags=re.IGNORECASE)

        # Разбиваем на предложения и обрабатываем каждое отдельно
        try:
            # Разбиваем по концу предложения, сохраняя разделители
            sentences = re.split(r'(?<=[.!?»"])\s+', processed)
            result_parts = []

            for sentence in sentences:
                if sentence.strip():
                    stressed = StressProcessor._accentizer.process_all(sentence)
                    result_parts.append(stressed)

            result = ' '.join(result_parts)

            # Конвертируем формат: +гласная → гласная'
            if stress_format == "apostrophe":
                result = re.sub(r'\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])', r"\1'", result)

            return result
        except Exception as e:
            logger.error("RUAccent processing error: %s", e)
            return text

    def add_custom_word(self, word: str, stressed_form: str) -> None:
        """Добавить слово в кастомный словарь."""
        self._custom_dict[word.lower()] = stressed_form
