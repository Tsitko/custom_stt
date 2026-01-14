# Локальный TTS/STT (Silero + GigaAM)

Консольное и HTTP‑приложение для синтеза и распознавания речи. По умолчанию TTS — **Silero v5_ru (eugene)**, STT — **GigaAM‑v3 (v2_rnnt)**. LLM‑предобработка/постобработка через LM Studio (OpenAI-compatible API, qwen/qwen3-30b-a3b-2507). Для Orpheus‑варианта сохранены настройки, но по умолчанию не используются.
Silero принимает текст кусками: сервис автоматически режет длинный текст (~400 символов) и склеивает результат.

## Подготовка окружения
- Python 3.10+.
- Системные зависимости: `ffmpeg` и `libsndfile`.
- Два окружения:
  - `.venv-wsl` — TTS (Silero/Orpheus) + сервис:  
    `pip install -r requirements.txt`
  - `.venv-stt` — STT (GigaAM): отдельное окружение с gigaam/torch.
- В `config.yml` по умолчанию `tts_engine: silero`. Orpheus‑настройки оставлены на случай возврата.

Пример основных настроек:
```yaml
sample_rate: 48000
output_dir: outputs
use_llm: true
llm_module_dir: llm
llm_settings_path: configs/llm/settings.yml

tts_engine: silero
silero_language: ru
silero_variant: v5_ru
silero_voice: eugene
silero_sample_rate: 48000

# STT (GigaAM-v3)
stt_model: v2_rnnt
stt_device: cpu
```

## Требования к эталону
- 5–15 секунд чистой речи одного спикера (mono, без музыки/шумов).
- Точный транскрипт сказанного (в одном стиле с текстом для синтеза).
- Допустимы несколько эталонов — клонирование стабильнее.

## CLI (локальный запуск)
TTS (Silero):
```
. .venv-wsl/bin/activate
python app.py --text "Новый текст" [--output outputs/result.ogg]
```
Orpheus (только если tts_engine=orpheus): добавить `--ref-dir` или пары `--ref-audio/--ref-text`.

STT:
```
. .venv-stt/bin/activate
python stt_app.py --audio path/to/audio.wav [--json]
```

## HTTP API
Запуск (в .venv-wsl):
```
uvicorn service:app --host 0.0.0.0 --port 8013
```
Запрос `POST /tts` (минимальный пример, для Silero достаточно текста):
```json
{
  "text": "Новый текст для синтеза."
}
```
Для Orpheus передавайте `references` (audio+text). Ответ: `processed_text` + путь к OGG.

### STT
CLI:
```
python stt_app.py --audio path/to/audio.wav [--config path/to/config.yml]
```
API: `POST /stt` с файлом (`multipart/form-data`), ответ — сырой и постобработанный текст.

## Сервис/автозапуск
- Рабочее окружение: `.venv-wsl`, старт: `./serve.sh` (uvicorn `0.0.0.0:8013`).
- systemd unit `custom-tts.service` автоперезапускает сервис; для ручного рестарта — `systemctl restart custom-tts` или завершить процесс на 8013.

## Как работает TTS
- Конфиг читается из `config.yml`, движок задаётся ключом `tts_engine` (по умолчанию `silero`), окружение — `.venv-wsl`.
- Перед отправкой в LLM все цифры превращаются в слова через `num2words(..., lang="ru")`, затем при `use_llm: true` текст прогоняется через `utils/llm_preprocessor.py` (LM Studio промпт для транскрипции англицизмов, ударений и пауз).
- Silero: `utils/silero_engine.py` грузит модель через `torch.hub`, для длинных текстов сервис режет на куски ~400 символов, синтезирует каждый, нормализует пиковую громкость и склеивает тензоры в один WAV.
- Orpheus: требует эталоны (аудио+текст). `utils/orpheus_engine.py` кодирует эталон в SNAC-токены, добавляет текст (эмоция/стиль по запросу или из конфига), генерирует аудио-токены LLaMA‑моделью и декодирует в WAV.
- Готовый WAV сохраняется в OGG Vorbis через `utils/file_handler.py` (`ffmpeg`), путь по умолчанию — `outputs/speech_<timestamp>.ogg`.

## Как работает STT
- Отдельное окружение `.venv-stt` (чтобы не конфликтовало с TTS/torch). CLI `stt_app.py` и сервис `/stt` используют его через `./.venv-stt/bin/python`.
- Сервис принимает файл, сохраняет во временный WAV/OGG/FLAC и запускает CLI. CLI читает `config.yml`, инициализирует `stt/gigaam_transcriber.py` (GigaAM-v3, устройство из `stt_device`) и выдаёт сырой текст.
- При включённом `use_llm` сырой текст постобрабатывается `utils/llm_preprocessor.py` в режиме `stt` (нормализация, знаки препинания).
- Результат сохраняется в `outputs`: `stt_<timestamp>_raw.txt` и `stt_<timestamp>.txt`; сервис возвращает пути (в WSL конвертирует их в формат Windows для клиентов).

## Тесты
```
. .venv-wsl/bin/activate
python -m unittest
```
Юнит-тесты используют заглушки Orpheus, поэтому не скачивают веса. Функциональный тест проверяет полный цикл: генерация WAV → конверсия в OGG.

## Структура
- `app.py` — CLI TTS с эталонами голоса.
- `service.py` — FastAPI сервис (`/tts` и `/stt`).
- `utils/` — конфиг/LLM/Orpheus движок, конвертер WAV→OGG.
- `stt/` — GigaAM-v3 transcriber.
- `configs/llm/` — настройки LM Studio и промпты.
- `tests/` — юнит и функциональные проверки.
