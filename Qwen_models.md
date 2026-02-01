Теперь у меня достаточно информации для составления конкретного запроса для кодового агента Claude. Составлю четкий, структурированный запрос с прямыми ссылками на документацию и репозитории.

***

# Запрос для кодового агента Claude

Мне нужна помощь в установке и использовании двух моделей от Qwen для работы с речью:

1. **Qwen3-ASR-1.7B** — распознавание речи (52 языка включая русский)
2. **Qwen3-TTS-12Hz-1.7B-Base** — синтез речи с клонированием голоса (10 языков включая русский)

## Конкретные задачи

### Задача 1: Qwen3-ASR-1.7B (распознавание речи)

**Что нужно сделать:**
1. Установить модель с поддержкой vLLM backend (для максимальной скорости)
2. Написать рабочий скрипт для транскрибирования аудиофайла на русском языке
3. Показать как использовать батч-инференс для нескольких файлов

**Официальная документация:**
- **Hugging Face Model Card**: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- **GitHub репозиторий**: https://github.com/QwenLM/Qwen3-ASR
- **vLLM интеграция**: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html

**Технические требования:**
- Python 3.12 (предпочтительно)
- CUDA 11.4+ (у меня есть GPU)
- Flash Attention 2 (для снижения использования VRAM)

**Пример установки из документации:**
```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr[vllm]  # С поддержкой vLLM для скорости
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

**Что мне важно узнать:**
- Сколько VRAM потребуется для модели 1.7B в bfloat16?
- Как передавать локальные аудиофайлы (не только URL)?
- Можно ли настроить автоопределение языка или лучше явно указывать русский?

***

### Задача 2: Qwen3-TTS-12Hz-1.7B-Base (синтез речи с клонированием)

**Что нужно сделать:**
1. Установить модель для клонирования голоса
2. Написать скрипт, который:
   - Берёт референсное аудио (3+ секунды моего голоса)
   - Генерирует новую речь на русском языке в этом голосе
3. Сохранить результат в WAV файл

**Официальная документация:**
- **Hugging Face Model Card**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- **GitHub репозиторий**: https://github.com/QwenLM/Qwen3-TTS
- **README с примерами**: https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md

**Технические требования:**
- Python 3.12
- PyTorch с CUDA
- Flash Attention 2

**Пример установки из документации:**
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

**Пример кода клонирования из документации:**
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Создаём промпт из референсного аудио
prompt_items = model.create_voice_clone_prompt(
    ref_audio="/path/to/reference.wav",  # 3+ секунды
    ref_text="Точная транскрипция референсного аудио",
    x_vector_only_mode=False,
)

# Генерируем речь
wavs, sr = model.generate_voice_clone(
    text="Это новый текст в клонированном голосе.",
    language="Russian",
    voice_clone_prompt=prompt_items,
)

sf.write("output.wav", wavs[0], sr)
```

**Что мне важно узнать:**
- Как получить транскрипцию референсного аудио для `ref_text`? Можно ли использовать Qwen3-ASR для этого?
- Сколько VRAM потребуется для 1.7B TTS модели?
- Параметр `x_vector_only_mode=True` — когда его использовать? В чём разница?

***

## Дополнительные вопросы

1. **Можно ли использовать обе модели в одном окружении** или лучше создать отдельные conda environments?
2. **Какие значения `temperature`, `top_p` рекомендуются** для TTS при генерации естественной речи?
3. **Как мониторить использование VRAM** во время инференса, чтобы избежать OOM?

***

## Ссылки на ключевую документацию (для справки)

**Qwen3-ASR:**
- Model Card: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- GitHub: https://github.com/QwenLM/Qwen3-ASR
- Technical Report (arXiv): https://arxiv.org/abs/2601.21337

**Qwen3-TTS:**
- Model Card: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- GitHub: https://github.com/QwenLM/Qwen3-TTS
- Technical Report (arXiv): https://arxiv.org/abs/2601.15621
- Hugging Face Demo: https://huggingface.co/spaces/Qwen/Qwen3-TTS

***

**Формат ответа:**
Пожалуйста, предоставь:
1. Пошаговые команды установки (с пояснениями)
2. Полные рабочие Python-скрипты с комментариями
3. Примеры использования на реальных данных
4. Информацию о требованиях к ресурсам (VRAM, время инференса)
5. Best practices и потенциальные проблемы