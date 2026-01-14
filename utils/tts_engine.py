"""TTS engine wrapper built on top of pyttsx3/espeak-ng."""
import logging
from pathlib import Path

import pyttsx3

from utils.interfaces import ITTSEngine


class TTSEngine(ITTSEngine):
    """Wraps pyttsx3 to synthesize text into a temporary WAV file."""

    _VOICE_MAP = {
        "ru": {
            "ivan": ["ru+m3", "zle/ru", "ru"],
            "иван": ["ru+m3", "zle/ru", "ru"],
            "masha": ["ru+f3", "zle/ru+f3", "ru+f3"],
            "маша": ["ru+f3", "zle/ru+f3", "ru+f3"],
        }
    }

    def __init__(self, voice: str, language: str = "ru") -> None:
        if not voice:
            raise ValueError("Voice value cannot be empty")
        if not language:
            raise ValueError("Language value cannot be empty")
        self._voice = voice
        self._language = language

    def synthesize_to_wav(self, text: str, output_path: Path) -> Path:
        if not text or not text.strip():
            raise ValueError("Text for synthesis must not be empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        engine = pyttsx3.init()
        voice_id = self._resolve_voice_id(engine)
        logging.info("Using voice id: %s", voice_id)
        engine.setProperty("voice", voice_id)
        logging.info("Synthesizing to WAV: %s", output_path)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Synthesis failed: output WAV was not created")

        return output_path

    def _resolve_voice_id(self, engine: pyttsx3.Engine) -> str:
        language_key = self._language.lower()
        candidates = self._VOICE_MAP.get(language_key, {}).get(self._voice.lower())

        if not candidates:
            candidates = [self._voice]

        available = engine.getProperty("voices")
        logging.info("Voice candidates: %s", candidates)
        for candidate in candidates:
            for voice in available:
                if voice.id.lower() == candidate.lower() or voice.name.lower() == candidate.lower():
                    return voice.id

        # If none of the candidates are announced by pyttsx3, pick the first one
        # and let espeak-ng attempt to resolve it directly (e.g., ru+f3).
        return candidates[0]
