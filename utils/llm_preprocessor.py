"""Wrapper around the local safe_llm LM Studio client to preprocess text."""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import re

try:
    from num2words import num2words
except Exception:  # pragma: no cover - optional dependency for STT mode
    num2words = None

DEFAULT_LLM_DIR = Path(__file__).resolve().parents[1] / "llm"
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "llm" / "prompts"
logger = logging.getLogger(__name__)


class LLMPreprocessor:
    """Uses local safe_llm client to enrich text for TTS (transcription, accents)."""

    def __init__(
        self,
        module_dir: str | Path | None = None,
        mode: str = "tts",
        llm_settings: Optional[Dict[str, Any]] = None,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self._module_dir = Path(module_dir) if module_dir else DEFAULT_LLM_DIR
        self._mode = mode
        self._llm_settings = llm_settings or {}
        self._prompts_dir = Path(prompts_dir) if prompts_dir else DEFAULT_PROMPTS_DIR
        self._prompt_cache: Optional[str] = None

    def process(self, text: str) -> str:
        if not text or not text.strip():
            return text

        model = self._llm_settings.get("model", "default")
        text_for_llm = self._numbers_to_words(text) if self._mode == "tts" else text
        logger.info(
            "LLM preprocessing started (module_dir=%s, model=%s, mode=%s)",
            self._module_dir,
            model,
            self._mode,
        )
        forced_text = os.environ.get("LLM_FORCE_TEXT")
        if forced_text:
            logger.warning("LLM forced response via env LLM_FORCE_TEXT (mode=%s): %s", self._mode, forced_text)
            return forced_text
        try:
            return asyncio.run(self._process_async(text_for_llm))
        except FileNotFoundError:
            logger.warning("LLM module directory not found: %s, using raw text", self._module_dir)
        except ModuleNotFoundError as exc:
            logger.warning("LLM module not importable (%s), using raw text", exc)
        except Exception as exc:  # pragma: no cover - network/runtime errors
            logger.warning("LLM preprocessing failed (%s), using raw text", exc)
        return text

    async def _process_async(self, text: str) -> str:
        client_cls, settings_cls = self._import_client()
        settings = settings_cls.from_mapping(self._llm_settings)

        prompt = self._build_prompt(text)
        client = client_cls(settings)
        try:
            result = await client.generate(prompt)
        finally:
            await client.close()

        if not result:
            logger.warning("LLM returned empty result, using raw text")
            return text
        logger.warning("LLM preprocessing complete (mode=%s, length=%s)", self._mode, len(result))
        logger.warning("LLM result (%s): %s", self._mode, result)
        return result

    def _import_client(self):
        module_root = self._module_dir
        if not module_root.exists():
            raise FileNotFoundError(module_root)

        sys.path.insert(0, str(module_root))
        safe_llm_module = importlib.import_module("safe_llm.lmstudio_client")
        SafeLMStudioClient = getattr(safe_llm_module, "SafeLMStudioClient")
        LMStudioClientSettings = getattr(safe_llm_module, "LMStudioClientSettings")
        return SafeLMStudioClient, LMStudioClientSettings

    def _numbers_to_words(self, text: str) -> str:
        """Convert bare integers to Russian words before sending to LLM."""
        if not num2words:
            logger.warning("num2words not available, skipping number normalization")
            return text

        def _convert(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                return num2words(int(raw), lang="ru")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("num2words failed for %s: %s", raw, exc)
                return raw

        return re.sub(r"\d+", _convert, text)

    def _build_prompt(self, text: str) -> str:
        """Build prompt by loading template from file and substituting text."""
        if self._prompt_cache is None:
            self._prompt_cache = self._load_prompt_template()

        return self._prompt_cache.format(text=text)

    def _load_prompt_template(self) -> str:
        """Load prompt template from file based on mode."""
        prompt_filename = f"{self._mode}_prompt.txt"
        prompt_path = self._prompts_dir / prompt_filename

        if not prompt_path.exists():
            logger.warning(
                "Prompt file not found: %s, using fallback hardcoded prompt",
                prompt_path
            )
            return self._get_fallback_prompt()

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            logger.info("Loaded prompt template from: %s", prompt_path)
            return template
        except Exception as exc:
            logger.warning(
                "Failed to load prompt from %s: %s, using fallback",
                prompt_path,
                exc
            )
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Fallback hardcoded prompts in case files are not available."""
        if self._mode == "stt":
            return (
                "Ты — постобработчик распознанного текста (ASR) на русском.\n"
                "Задача: исправь возможные ошибки распознавания, числа, термины и англицизмы.\n"
                "- Английские слова оставляй оригинально, если узнал корректно; иначе замени на ближайшее корректное.\n"
                "- Числа верни буквами или цифрами по смыслу, исправь перепутанные.\n"
                "- Исправь опечатки, расставь знаки препинания, сделай текст связным без добавления пояснений.\n"
                "- Если слова — явно английские термины или сокращения — заменяй их на оригинальные английские термины и сокращения. Например джи пи ю -> GPU.\n"
                "- Верни только итоговый текст без комментариев и сервисных слов.\n\n"
                "Распознанный текст:\n{text}"
            )

        return (
            "Ты — предобработчик текста для TTS на русском.\n"
            "Твоя задача:\n"
            "- Все английские слова транскрибируй русскими буквами (пример: 'GPU' -> 'джи пи ю').\n"
            "- Расставь ударения: ставь апостроф (') перед ударным слогом (пример: раз'работка, серв'ер).\n"
            "- Добавляй паузы через запятые и точки, чтобы речь звучала естественно.\n"
            "- Все слова на других языках переделывай в транскрипцию. Например stable -> стэйбл.\n"
            "- Сохрани исходную структуру смысловых блоков, но без пояснений.\n"
            "- Никогда не отвечай на вопрос, не добавляй свой смысл — только преобразуй входной текст.\n"
            "- Верни только итоговый текст, без комментариев и префиксов.\n\n"
            "Текст:\n{text}"
        )
