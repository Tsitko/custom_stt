# Локальный TTS/STT (Qwen + Silero/Orpheus + GigaAM)

Консольное и HTTP‑приложение для синтеза и распознавания речи. По умолчанию TTS — **Qwen3-TTS**, STT — **Qwen3-ASR** (см. `config.yml`). Silero/Orpheus и GigaAM доступны как альтернативы. LLM‑предобработка/постобработка опциональна (TTS по умолчанию выключен, STT включён).
Silero принимает текст кусками: сервис автоматически режет длинный текст (~400 символов) и склеивает результат.

## Подготовка окружения
- Python 3.10+.
- Системные зависимости: `ffmpeg`, `libsndfile`, `sox`.
- Одно окружение для сервиса и CLI:
  - `.venv-qwen`  
    `pip install -r requirements.txt`
- В `config.yml` по умолчанию `tts_engine: qwen`, `stt_engine: qwen`. Silero/Orpheus и GigaAM доступны как альтернативы.
- Опциональное ускорение Qwen TTS через FlashAttention: `flash_attn_integration.md`.

Пример основных настроек:
```yaml
sample_rate: 48000
output_dir: outputs
use_llm_tts: false
use_llm_stt: true
llm_module_dir: llm
llm_settings_path: configs/llm/settings.yml
stt_silence_db: -45

tts_engine: qwen
stt_engine: qwen

qwen_tts_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
qwen_tts_device: cuda:0
qwen_tts_attn_implementation: flash_attention_2
qwen_asr_model: Qwen/Qwen3-ASR-1.7B
qwen_asr_device: cuda:0

# STT (GigaAM-v3, если stt_engine=gigaam)
stt_model: v2_rnnt
stt_device: cuda
```

## Требования к эталону
- 5–15 секунд чистой речи одного спикера (mono, без музыки/шумов).
- Точный транскрипт сказанного (в одном стиле с текстом для синтеза).
- Допустимы несколько эталонов — клонирование стабильнее.

## CLI (локальный запуск)
Важно: `app.py` **не поддерживает Qwen TTS**. В CLI доступны только Silero и Orpheus.
Если `tts_engine` в `config.yml` стоит `qwen`, `app.py` всё равно пойдёт по ветке Orpheus и потребует эталоны.

Пример запуска (скобки `[]` в README — это обозначение опциональных аргументов, их не нужно писать):
```
. .venv-qwen/bin/activate
python app.py --text "Новый текст" --output outputs/result.ogg
```
Silero: работает без эталонов (`tts_engine: silero`).
Orpheus: обязательно добавить `--ref-dir` или пары `--ref-audio/--ref-text`.
Qwen TTS вызывается через HTTP API (см. ниже).

STT (CLI использует GigaAM):
```
. .venv-qwen/bin/activate
python stt_app.py --audio path/to/audio.wav [--json]
```

## HTTP API
Запуск (в .venv-qwen):
```
uvicorn service:app --host 0.0.0.0 --port 8013
```
Запрос `POST /tts` (минимальный пример, для Silero достаточно текста):
```json
{
  "text": "Новый текст для синтеза."
}
```
Для Orpheus передавайте `references` (audio+text). Для Qwen: `reference_audio_base64` + `reference_text` или `voice_description` (VoiceDesign). Если не передали, используются `voice_clone_ref_audio` и `voice_clone_ref_text` из `config.yml`. Ответ: `processed_text` + путь к OGG.

### WAV без OGG
`POST /tts/wav` принимает тот же JSON, что и `/tts`, но возвращает `audio/wav` напрямую (без конвертации в OGG).  
Заголовки ответа:
- `X-Processed-Text-Length`
- `X-Sample-Rate`

### Streaming (Deepgram-like)
`WS /tts/speak-streaming` — текстовый control-channel + бинарные аудио-фреймы WAV.

Входящие JSON-сообщения:
- `{"type":"Speak","text":"..."}`
- `{"type":"Flush"}`
- `{"type":"Clear"}`
- `{"type":"Close"}`

Исходящие сообщения:
- JSON `Metadata` при подключении;
- бинарные WAV-чанки после `Flush`;
- JSON `Flushed` с метаданными (`sequence_id`, `sample_rate`, `elapsed_s`, `bytes`);
- JSON `Warning` при ошибках протокола.

Документация протокола в Swagger:
- `GET /tts/speak-streaming/protocol` (контракт сообщений для `WS /tts/speak-streaming`)

Тестовый клиент:
```bash
python scripts/test_tts_stream_ws.py \
  --url ws://127.0.0.1:8013/tts/speak-streaming \
  --text "Привет, ребята. Это тест потокового синтеза." \
  --speak-split 32 \
  --out outputs/ws_stream_test.wav
```

### STT
CLI:
```
python stt_app.py --audio path/to/audio.wav [--config path/to/config.yml]
```
API: `POST /stt` с файлом (`multipart/form-data`), ответ — сырой и постобработанный текст.
Дополнительное поле `context` (опционально) добавляется к промпту LLM для постобработки.
Если меняли обработку контекста или промпт, перезапустите сервис: STT воркер живёт долго и подхватывает код/шаблоны только при старте.

## Сервис/автозапуск
- Рабочее окружение: `.venv-qwen`, старт: `./serve.sh` (uvicorn `0.0.0.0:8013`).
- systemd unit `custom-tts.service` автоперезапускает сервис (включая запуск при старте WSL).
- В сервисе включён единый слот инференса: одновременно обрабатывается только 1 запрос TTS/STT/stream flush, остальные ждут очередь.
- Управление сервисом в WSL:
```bash
sudo systemctl stop custom-tts.service
sudo systemctl start custom-tts.service
sudo systemctl restart custom-tts.service
sudo systemctl status custom-tts.service
```
- При рестарте сервис теперь принудительно выгружает модели и завершает процесс даже если генерация идёт (SIGTERM/SIGINT → очистка VRAM/RAM → `os._exit(0)`), чтобы рестарт не зависал.

## Как работает TTS
- Конфиг читается из `config.yml`, окружение — `.venv-qwen`.
- При `use_llm_tts: true` текст прогоняется через `utils/llm_preprocessor.py` (LLM‑нормализация); при `use_stress: true` добавляется расстановка ударений.
- Qwen (только через HTTP API): `utils/qwen_tts_engine.py` выполняет voice cloning по эталону (из запроса или из `voice_clone_ref_*`) или VoiceDesign по `voice_description`.
- Silero (CLI и API): `utils/silero_engine.py` грузит модель через `torch.hub`, для длинных текстов сервис режет на куски ~400 символов, синтезирует каждый, нормализует пиковую громкость и склеивает тензоры в один WAV.
- Orpheus (CLI и API): требует эталоны (аудио+текст). `utils/orpheus_engine.py` кодирует эталон в SNAC‑токены, добавляет текст (эмоция/стиль по запросу или из конфига), генерирует аудио‑токены LLaMA‑моделью и декодирует в WAV.
- Готовый WAV сохраняется в OGG Vorbis через `utils/file_handler.py` (`ffmpeg`), путь по умолчанию — `outputs/speech_<timestamp>.ogg`.

## Как работает STT
- Сервис `/stt` использует `stt_engine` из `config.yml`: `qwen` или `gigaam`.
- Qwen ASR (`stt/qwen_asr_transcriber.py`) держит модель в памяти с TTL (`model_ttl_seconds`), что убирает холодный старт для частых запросов.
- CLI `stt_app.py` использует только GigaAM, читает `config.yml`, инициализирует `stt/gigaam_transcriber.py` (устройство из `stt_device`) и выдаёт сырой текст.
- При включённом `use_llm_stt` сырой текст постобрабатывается `utils/llm_preprocessor.py` в режиме `stt` (нормализация, знаки препинания).
- Если `stt_silence_db` задан, перед распознаванием считается RMS в dBFS и тишина ниже порога отсекается (возвращается пустой текст).
- Длинные записи режутся оффлайн через VAD (webrtcvad) на сегменты до ~22 секунд и распознаются по частям.
- Подготовка оффлайн‑VAD: `scripts/prepare_stt_longform_offline.sh` (ставит webrtcvad и запускает оффлайн‑проверку).
- Результат сохраняется в `outputs`: `stt_<timestamp>_raw.txt` и `stt_<timestamp>.txt`; сервис возвращает пути (в WSL конвертирует их в формат Windows для клиентов).

## Тесты
```
. .venv-qwen/bin/activate
python -m unittest
```
Юнит-тесты используют заглушки Orpheus, поэтому не скачивают веса. Функциональный тест проверяет полный цикл: генерация WAV → конверсия в OGG.

## Структура
- `app.py` — CLI TTS с эталонами голоса.
- `service.py` — FastAPI сервис (`/tts` и `/stt`).
- `utils/` — конфиг/LLM/Orpheus движок, конвертер WAV→OGG.
- `stt/` — STT transcribers (Qwen ASR, GigaAM).
- `configs/llm/` — настройки LM Studio и промпты.
- `tests/` — юнит и функциональные проверки.
