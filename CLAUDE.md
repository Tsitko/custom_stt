# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Offline Russian Text-to-Speech (TTS) and Speech-to-Text (STT) system. STT includes LLM-based postprocessing for error correction. The system works completely offline (after initial model downloads) and includes a FastAPI web service for remote access.

## Development Commands

### Environment Setup
```bash
source .venv-wsl/bin/activate
```

### Running the Applications
```bash
# TTS - Text to Speech
python app.py --text "Привет, это тестовый пример" [--config config.yml] [--output outputs/result.ogg]

# STT - Speech to Text
python stt_app.py --audio path/to/audio.wav [--config config.yml]

# FastAPI Web Service
./serve.sh
# Or: uvicorn service:app --host 0.0.0.0 --port 8013
```

### Testing
```bash
# Run all tests
python -m unittest

# Run specific test file
python -m unittest tests.test_config_loader
python -m unittest tests.test_silero_engine
python -m unittest tests.test_e2e_minute

# Run specific test class or method
python -m unittest tests.test_config_loader.TestConfigLoader
python -m unittest tests.test_config_loader.TestConfigLoader.test_default_values
```

### Testing with LLM Override
```bash
# Force specific LLM response for deterministic testing
LLM_FORCE_TEXT="тестовый текст" python -m unittest tests.test_llm_preprocessor
```

## Architecture

### Processing Pipelines

**TTS Pipeline:**
```
Text Input
  → TTS Engine (pyttsx3 or Silero)
      - Generates WAV audio
  → FileHandler
      - Converts WAV to OGG Vorbis via ffmpeg
      - Resamples to target sample rate
  → OGG Output File
```

**STT Pipeline:**
```
Audio Input
  → GigaAMTranscriber (GigaAM-v3)
      - Preprocesses audio with librosa (resample to 16kHz)
      - Transcribes to raw text
  → LLMPreprocessor (stt mode)
      - Corrects recognition errors
      - Fixes punctuation
      - Expands abbreviations (джи пи ю → GPU)
  → Clean Text Output
```

### Interface-Based Design

All major components implement abstract interfaces defined in `utils/interfaces.py`:

- **`ITTSEngine`** - TTS synthesis interface
  - `TTSEngine` (pyttsx3/espeak-ng wrapper)
  - `SileroTTSEngine` (neural TTS)

- **`ISTTTranscriber`** - STT transcription interface
- `GigaAMTranscriber` (GigaAM-v3 wrapper)

- **`IConfigLoader`** - Configuration loading interface
  - `ConfigLoader` (manual YAML parser, no pyyaml dependency)

- **`IFileHandler`** - Audio conversion interface
  - `FileHandler` (WAV → OGG via ffmpeg)

### LLM Integration Architecture

**LLM Preprocessor** (`utils/llm_preprocessor.py`):
- Dynamically imports `llm/safe_llm/lmstudio_client.py` at runtime
- Uses async LM Studio client (`SafeLMStudioClient`)
- **STT mode only**: Loads prompt from `configs/llm/prompts/stt_prompt.txt`
  - Corrects recognition errors, fixes punctuation, expands abbreviations
- Prompts use `{text}` placeholder for text substitution
- Fallback: Hardcoded prompts if files unavailable
- Graceful degradation: Falls back to raw text if LLM unavailable
- Testing support: `LLM_FORCE_TEXT` environment variable for deterministic tests

**LM Studio Client** (`llm/safe_llm/lmstudio_client.py`):
- Async HTTP client using aiohttp (OpenAI-compatible `/v1/chat/completions`)
- Settings loaded from `configs/llm/settings.yml`
- Configurable: base_url, model, temperature, max_tokens, timeouts
- Session management with proper event loop handling
- Supports both streaming and non-streaming generation

**LLM Configuration** (`configs/llm/settings.yml`):
- Centralized LLM settings (model, server URL, parameters)
- Referenced from main `config.yml` via `llm_settings_path`
- All LLM-related settings in one place for easy modification

### Configuration Management

**Config File**: `config.yml` (manual YAML parsing, no pyyaml)

Key settings:
- `engine`: `pyttsx3` or `silero`
- `voice`: Speaker name (eugene, baya, kseniya, etc. for Silero)
- `sample_rate`: Audio quality (48000 for Silero, 44100 for pyttsx3)
- `use_llm`: Enable/disable LLM postprocessing for STT (default: true)
- `llm_settings_path`: Path to LLM configuration file (default: configs/llm/settings.yml)
- `stt_model`: Whisper model size (tiny, base, small, medium, large)
- `stt_device`: cpu or cuda
- `stt_device`: auto|cuda|cpu for GigaAM

### Silero Engine Specifics

**Long Text Handling** (`utils/silero_engine.py`):
- Silero has 800-character limit per synthesis call
- Engine automatically chunks text by sentences/punctuation
- Concatenates audio chunks before returning
- Model caching: Downloads once via torch.hub, reuses cached model

**Voice Compatibility**:
- Legacy mapping: `ivan` → `eugene`, `masha` → `baya`
- Available voices: baya, kseniya, eugene, aidar, xenia, irina, natasha, ruslan

### Web Service Architecture

**FastAPI Service** (`service.py`):
- `/tts` endpoint (POST): Accepts text, returns OGG audio file
- `/stt` endpoint (POST): Accepts audio file, returns transcribed text
- WSL/Windows path conversion for cross-platform compatibility
- Async processing using `asyncio.to_thread` for blocking TTS/STT operations

### Testing Strategy

**Test Types**:
- **Unit tests**: Individual component functionality (config, engines, LLM)
- **Integration tests**: Module imports and interface compliance
- **Functional tests**: Complete pipelines (TTS text → OGG file)
- **E2E tests**:
  - `test_e2e_minute.py`: ~1 minute Silero speech synthesis
  - `test_stt_e2e.py`: Full STT pipeline with LLM postprocessing

**Test Outputs**:
- E2E tests generate `outputs/e2e_minute.ogg` for manual listening verification
- Tests use `LLM_FORCE_TEXT` env variable to ensure deterministic LLM responses

## Key Implementation Details

### Prompt Modification

To modify LLM prompts, edit the prompt file directly:
- **STT prompt**: `configs/llm/prompts/stt_prompt.txt`

Prompts use `{text}` placeholder for text substitution.

**STT Prompt Features:**
- Cyrillic transcription → English terms ("джи пи ю" → GPU, "апий" → API, "кубернетис" → Kubernetes)
- Words → digits in data context ("восемьдесят процентов" → 80%, "три точка восемь" → 3.8)
- Spelling, punctuation, and grammar corrections
- First mention abbreviation expansion: "база данных" → "база данных (БД)"
- 15+ common term mappings
- 4 comprehensive before/after examples

Extended prompt ideas are documented in:
- `stt_prompt_ideas.md` - Detailed STT postprocessing instructions

Fallback prompts are hardcoded in `utils/llm_preprocessor.py` in the `_get_fallback_prompt()` method.

### LLM Model Configuration

To change the LLM model or connection settings, edit `configs/llm/settings.yml`:
- `model`: LM Studio model name (e.g., qwen/qwen3-30b-a3b-2507)
- `base_url`: LM Studio server URL (default: http://localhost:1234/v1)
- `temperature`, `max_tokens`: Generation parameters
- `request_timeout`, `connect_timeout`: Connection timeouts

Model selection is **only** configured via config files, not hardcoded in application code.

### Audio Format Conversion

**FileHandler** (`utils/file_handler.py`):
- Uses ffmpeg subprocess for WAV → OGG conversion
- Handles PCM audio with numpy/soundfile
- Supports sample rate conversion (default 44.1 kHz)
- WSL path translation for Windows/WSL interop

### Model Downloads

Models download automatically on first use:
- **Silero**: Downloads via `torch.hub` to `~/.cache/torch/hub`
- **faster-whisper**: Downloads to `~/.cache/huggingface/hub`
- **LM Studio models**: Managed by LM Studio
  - Ensure the model specified in `configs/llm/settings.yml` is loaded in LM Studio

## Python Version Constraints

- **Python 3.12**: Supported for Silero and faster-whisper
- **Python 3.10/3.11**: Required for Coqui TTS (optional dependency)
- Virtual environment: `.venv-wsl`

## External Dependencies

System packages required:
- `espeak-ng` (for pyttsx3 engine)
- `libsndfile` (for OGG encoding)
- `ffmpeg` (for audio conversion)
- LM Studio server running on the configured host/port (for LLM features)
