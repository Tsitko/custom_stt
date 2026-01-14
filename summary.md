# Сводка по проекту custom_tts

Гибридный офлайн/онлайн TTS/STT сервис для русской речи с предобработкой через локальный LLM (LM Studio). Цель — автономность и фоллбек. По умолчанию TTS — Silero v5 (chunking длинных текстов), Orpheus оставлен опционально.

## Архитектура
- CLI: `app.py` (tts) и `stt_app.py`; FastAPI `service.py` с `/tts` и `/stt`.
- Конфиг: `config.yml`, LLM-настройки `configs/llm/settings.yml`, промпты `configs/llm/prompts/`.
- LLM-предобработка (`llm/`): транскрипция англицизмов, ударения/паузы, числа → слова; для STT — обратная нормализация.
- TTS: Silero v5 (по умолчанию, текст режется и склеивается) или Orpheus voice cloning (`utils/orpheus_engine.py`) с эталонными аудио/текстами.
- STT: GigaAM-v3 (`stt/gigaam_transcriber.py`); утилиты для конвертации и LLM-предобработки.

## Сервис и автозапуск
- `serve.sh` поднимает uvicorn `0.0.0.0:8013` в `.venv-wsl`.
- systemd unit `custom-tts.service` стартует с WSL, перезапускается при падении; логи: `journalctl -u custom-tts.service`.
- Стартап прогревает Silero. Веса положите заранее: `~/.cache/torch/hub/checkpoints/{v4_ru.pt,...}`.

## Поток `/tts`
1) Проверка текста, выбор пути (`outputs/speech_<timestamp>.ogg`).
2) Предобработка LLM (опционально).
3) Синтез в WAV: Silero режет длинные тексты на куски и собирает; Orpheus использует эталоны.
4) Конверсия в OGG и ответ с обработанным текстом и путём к файлу.

## Поток `/stt`
1) Приём файла и временное сохранение.
2) Распознавание GigaAM-v3.
3) Опциональный постпроцесс LLM.
4) Сохранение raw/processed в `outputs`, возврат путей с учётом Windows.

## Интеграции
- В `telegram_agent/yandex_speech` CLI сначала идёт в локальный сервис (`LOCAL_SPEECH_URL=http://localhost:8013`, `LOCAL_SPEECH_TIMEOUT` 30 с); при таймауте падает на SpeechKit, поэтому держите сервис прогретым.

## Тесты
- `python -m unittest` — юнит/интеграция; e2e с длинным текстом (Silero) пишет `outputs/e2e_minute.ogg`.

## Эксплуатация
- Конфиг `stt_model: e2e_rnnt`, `stt_device: auto|cuda|cpu`.
- При `use_llm: false` текст идёт напрямую в движок.
- Пути в ответах автоматически переводятся в формат Windows при работе из WSL.
