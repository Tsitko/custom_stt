"""Interface definitions for the TTS application components."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Sequence


@dataclass
class Config:
    """Configuration values loaded from config.yml."""

    sample_rate: int
    output_dir: str
    use_llm: bool
    llm_module_dir: str | None
    llm_settings: Dict[str, Any] = field(default_factory=dict)
    stt_model: str = "e2e_rnnt"
    stt_device: str = "auto"
    tts_engine: str = "orpheus"  # orpheus | silero
    # Silero TTS settings
    silero_language: str = "ru"
    silero_variant: str = "v5_ru"
    silero_voice: str = "eugene"
    silero_sample_rate: int = 48000
    silero_normalize: bool = True
    # Orpheus TTS settings
    orpheus_model: str = "papacliff/orpheus-3b-0.1-ft-ru"
    orpheus_codec: str | None = None
    orpheus_device: str = "auto"
    orpheus_dtype: str = "float16"
    orpheus_voice: str | None = None
    default_emotion: str | None = None
    default_style: str | None = None
    max_reference_seconds: float = 20.0


@dataclass
class VoiceReference:
    """Pair of reference audio path and its transcript for voice cloning."""

    audio_path: Path
    transcript: str


class ITTSEngine(ABC):
    """Interface for text-to-speech engine implementations."""

    @abstractmethod
    def synthesize_to_wav(
        self,
        text: str,
        references: Sequence[VoiceReference],
        output_path: Path,
        emotion: str | None = None,
        style: str | None = None,
    ) -> Path:
        """Generate speech audio from text using reference voice samples and store it as WAV."""


class IConfigLoader(ABC):
    """Interface for the configuration loader."""

    @abstractmethod
    def load(self) -> Config:
        """Load configuration values from disk."""


class IFileHandler(ABC):
    """Interface for saving synthesized audio."""

    @abstractmethod
    def save_ogg(self, wav_path: Path, ogg_path: Path) -> Path:
        """Convert a WAV file to OGG Vorbis and persist it."""


class ISTTTranscriber(ABC):
    """Interface for speech-to-text transcribers."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return transcription of the given audio file."""
