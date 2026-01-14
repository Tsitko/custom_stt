# Vendor copy adapted for LM Studio OpenAI-compatible API
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Mapping, Optional

import aiohttp


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LMStudioClientSettings:
    """Параметры подключения к LM Studio (OpenAI-compatible API)."""

    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen/qwen3-30b-a3b-2507"
    temperature: float = 0.3
    max_tokens: int = 30_000
    request_timeout: float = 520.0
    connect_timeout: float = 10.0
    max_connections: int = 10

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "LMStudioClientSettings":
        if not data:
            return cls()
        kwargs = {**data}
        defaults = cls()
        return cls(
            base_url=str(kwargs.get("base_url", defaults.base_url)),
            model=str(kwargs.get("model", defaults.model)),
            temperature=float(kwargs.get("temperature", defaults.temperature)),
            max_tokens=int(kwargs.get("max_tokens", defaults.max_tokens)),
            request_timeout=float(kwargs.get("request_timeout", defaults.request_timeout)),
            connect_timeout=float(kwargs.get("connect_timeout", defaults.connect_timeout)),
            max_connections=int(kwargs.get("max_connections", defaults.max_connections)),
        )


class SafeLMStudioClient:
    """
    Клиент LM Studio (OpenAI-compatible) для локальной модели.

    Используется только во внутреннем безопасном пайплайне.
    """

    def __init__(self, settings: Optional[LMStudioClientSettings] = None) -> None:
        self._settings = settings or LMStudioClientSettings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def generate(self, prompt: str) -> Optional[str]:
        """Синхронное получение полного ответа от LM Studio."""
        payload = {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "stream": False,
        }

        try:
            session = await self._get_session()
            logger.info("🔄 LM Studio запрос: %s символов", len(prompt))
            started = time.monotonic()
            async with session.post(self._chat_completions_url(), json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("❌ Ошибка LM Studio API: %s - %s", response.status, error_text)
                    return None

                data = await response.json()
                content = (
                    ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                ).strip()
                elapsed = time.monotonic() - started
                logger.info("✅ LM Studio ответил, длина: %s символов, время: %.2fs", len(content), elapsed)
                return content or None

        except asyncio.TimeoutError:
            logger.error("❌ LM Studio API: превышен таймаут ожидания ответа")
            return None
        except Exception as error:  # pragma: no cover - сетевые ошибки
            logger.error("❌ Ошибка LM Studio API: %s", error)
            return None

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Стриминговый ответ от LM Studio."""
        payload = {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "stream": True,
        }

        try:
            session = await self._get_session()
            logger.info("🔄 LM Studio стрим-запрос: %s символов", len(prompt))
            started = time.monotonic()
            async with session.post(self._chat_completions_url(), json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("❌ Ошибка LM Studio API: %s - %s", response.status, error_text)
                    return

                async for line in response.content:
                    if not line:
                        continue
                    chunk = line.decode("utf-8").strip()
                    if not chunk:
                        continue
                    if not chunk.startswith("data:"):
                        continue
                    data = chunk.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("LM Studio stream decode failed: %s", data)
                        continue
                    delta = (
                        (event.get("choices") or [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        yield delta
            elapsed = time.monotonic() - started
            logger.info("✅ LM Studio стрим завершен за %.2fs", elapsed)

        except Exception as error:  # pragma: no cover - сетевые ошибки
            logger.error("❌ Ошибка при стриминге LM Studio: %s", error)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._loop = None

    def _chat_completions_url(self) -> str:
        base = self._settings.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            current_loop = asyncio.get_running_loop()
            if self._loop is not current_loop:
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = None
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    total=self._settings.request_timeout,
                    connect=self._settings.connect_timeout,
                )
                connector = aiohttp.TCPConnector(limit=self._settings.max_connections)
                self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
                self._loop = current_loop
        return self._session


__all__ = ["SafeLMStudioClient", "LMStudioClientSettings"]
