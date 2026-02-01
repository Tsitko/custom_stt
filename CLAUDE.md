# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Offline Russian Text-to-Speech (TTS) and Speech-to-Text (STT) system with voice cloning support. STT includes LLM-based postprocessing for error correction. The system works completely offline (after initial model downloads) and includes a FastAPI web service for remote access.

## Development Commands

### Environment Setup
```bash
source .venv-qwen/bin/activate
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
python -m unittest tests.test_functional_pipeline

# Run specific test class or method
python -m unittest tests.test_config_loader.TestConfigLoader
```

### Testing with LLM Override
```bash
# Force specific LLM response for deterministic testing
LLM_FORCE_TEXT="тестовый текст" python -m unittest tests.test_llm_preprocessor
```

## Architecture

### Processing Pipelines

**TTS Pipeline (Qwen):**
```
Text Input
  → QwenTTSEngine (Qwen3-TTS-12Hz-1.7B-Base)
      - Voice cloning from reference audio
      - Text chunking (max 200 chars)
      - Crossfade between chunks (50ms)
      - torch.compile() for faster inference
  → FileHandler
      - Converts WAV to OGG Vorbis via ffmpeg
      - Resamples to target sample rate (48kHz)
  → OGG Output File
```

**STT Pipeline (Qwen):**
```
Audio Input
  → QwenASRTranscriber (Qwen3-ASR-1.7B)
      - Audio chunking (max 30 seconds)
      - Transcribes to raw text
  → LLMPreprocessor (stt mode, optional)
      - Corrects recognition errors
      - Fixes punctuation
      - Expands abbreviations (джи пи ю → GPU)
  → Clean Text Output
```

### Interface-Based Design

All major components implement abstract interfaces defined in `utils/interfaces.py`:

- **`ITTSEngine`** - TTS synthesis interface
  - `QwenTTSEngine` (Qwen3-TTS with voice cloning) - **default**
  - `SileroTTSEngine` (neural TTS, no cloning)
  - `OrpheusTTSEngine` (experimental)

- **`ISTTTranscriber`** - STT transcription interface
  - `QwenASRTranscriber` (Qwen3-ASR-1.7B) - **default**
  - `GigaAMTranscriber` (GigaAM-v3)

- **`IConfigLoader`** - Configuration loading interface
  - `ConfigLoader` (manual YAML parser, no pyyaml dependency)

- **`IFileHandler`** - Audio conversion interface
  - `FileHandler` (WAV → OGG via ffmpeg)

### Qwen TTS Engine

**Voice Cloning** (`utils/qwen_tts_engine.py`):
- Uses reference audio + transcript for voice cloning
- Reference files configured in `config.yml`:
  - `voice_clone_ref_audio`: Path to reference WAV (4 sec, 16kHz mono recommended)
  - `voice_clone_ref_text`: Path to transcript of reference audio
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~3.9 GB VRAM)

**Text Processing:**
- Automatic chunking by sentences (max 200 chars)
- Normalization: removes multiple dots, replaces dashes
- Crossfade concatenation (50ms) to eliminate clicks

**Generation Parameters:**
- `temperature`: 0.3 (lower = more similar voice)
- `top_p`: 0.9
- `repetition_penalty`: 1.1

**VRAM Management:**
- Model TTL: 5 minutes (configurable via `model_ttl_seconds`)
- Auto-unload with `gc.collect()` + `torch.cuda.empty_cache()`

### Qwen ASR Transcriber

**Speech Recognition** (`stt/qwen_asr_transcriber.py`):
- Model: `Qwen/Qwen3-ASR-1.7B` (~3.8 GB VRAM)
- Audio chunking: max 30 seconds per chunk
- Automatic resampling to 16kHz mono
- Same VRAM TTL management as TTS

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

### Configuration Management

**Config File**: `config.yml` (manual YAML parsing, no pyyaml)

Key settings:
```yaml
# Engine selection
tts_engine: qwen          # qwen | silero | orpheus
stt_engine: qwen          # qwen | gigaam

# Qwen TTS
qwen_tts_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
qwen_tts_device: cuda:0
qwen_tts_max_chars: 200
qwen_tts_crossfade_ms: 50
qwen_tts_temperature: 0.3
qwen_tts_top_p: 0.9
qwen_tts_repetition_penalty: 1.1

# Qwen ASR
qwen_asr_model: Qwen/Qwen3-ASR-1.7B
qwen_asr_device: cuda:0
qwen_asr_max_audio_seconds: 30

# Voice cloning reference
voice_clone_ref_audio: one_shot/stalin_reference.wav
voice_clone_ref_text: one_shot/stalin_reference.txt

# VRAM management
model_ttl_seconds: 300    # 5 minutes

# LLM postprocessing
use_llm: false            # Enable for STT error correction
```

### Web Service Architecture

**FastAPI Service** (`service.py`):
- `/tts` endpoint (POST): Accepts text, returns OGG audio path
- `/stt` endpoint (POST): Accepts audio file, returns transcribed text
- WSL/Windows path conversion for cross-platform compatibility
- Async processing using `asyncio.to_thread` for blocking TTS/STT operations
- Cached model instances for performance

**Systemd Service** (`/etc/systemd/system/custom-tts.service`):
- Auto-restart on failure
- Runs as user `denis`
- Logs via journalctl: `journalctl -u custom-tts.service -f`

### Testing Strategy

**Test Types**:
- **Unit tests**: Individual component functionality (config, engines, LLM)
- **Integration tests**: Module imports and interface compliance
- **Functional tests**: Complete pipelines (TTS text → OGG file)

**Test Outputs**:
- Tests generate files in `outputs/` for manual verification
- Tests use `LLM_FORCE_TEXT` env variable to ensure deterministic LLM responses

## Key Implementation Details

### Voice Reference Preparation

To create a voice reference for cloning:
```bash
# Extract 4 seconds from source audio (no loudnorm - hurts quality)
ffmpeg -y -i source.mp3 -ss 150 -t 4 -ar 16000 -ac 1 one_shot/reference.wav
```

Create transcript file with exact text spoken in the reference audio.

### Prompt Modification

To modify LLM prompts, edit the prompt file directly:
- **STT prompt**: `configs/llm/prompts/stt_prompt.txt`

Prompts use `{text}` placeholder for text substitution.

### Model Downloads

Models download automatically on first use to `~/.cache/huggingface/hub`:
- **Qwen TTS**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~3.9 GB)
- **Qwen ASR**: `Qwen/Qwen3-ASR-1.7B` (~3.8 GB)
- **Silero**: Downloads via `torch.hub` to `~/.cache/torch/hub`

Service runs in offline mode (`HF_HUB_OFFLINE=1`) after initial download.

## Python Version & Environment

- **Python 3.12**: Primary supported version
- Virtual environment: `.venv-qwen`

## External Dependencies

System packages required:
- `ffmpeg` (for audio conversion)
- `libsndfile` (for audio I/O)
- CUDA-capable GPU with ~8 GB VRAM (for both models simultaneously)

Optional:
- LM Studio server (for LLM postprocessing features)
