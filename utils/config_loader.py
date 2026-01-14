"""Configuration loader that reads settings from config.yml."""
import logging
from pathlib import Path
from typing import Dict, Any

from utils.interfaces import Config, IConfigLoader


class ConfigLoader(IConfigLoader):
    """Lightweight YAML-ish parser tailored to the Orpheus setup."""

    def __init__(self, config_path: Path | str = "config.yml") -> None:
        self._config_path = Path(config_path)

    def load(self) -> Config:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        logging.info("Loading config from %s", self._config_path)

        sample_rate: int = 24000
        output_dir: str = "outputs"
        use_llm: bool = True
        llm_module_dir: str | None = "llm"
        llm_settings_path: str = "configs/llm/settings.yml"
        stt_model: str = "e2e_rnnt"
        stt_device: str = "auto"
        # Default to silero to avoid Orpheus when config is missing tts_engine.
        tts_engine: str = "silero"
        silero_language: str = "ru"
        silero_variant: str = "v5_ru"
        silero_voice: str = "eugene"
        silero_sample_rate: int = 48000
        silero_normalize: bool = True
        orpheus_model: str = "papacliff/orpheus-3b-0.1-ft-ru"
        orpheus_codec: str | None = None
        orpheus_device: str = "auto"
        orpheus_dtype: str = "float16"
        orpheus_voice: str | None = None
        default_emotion: str | None = None
        default_style: str | None = None
        max_reference_seconds: float = 20.0

        for raw_line in self._config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key_lower = key.lower()
            if key_lower == "sample_rate":
                sample_rate = self._parse_int(value, "Sample rate")
            elif key_lower == "output_dir":
                output_dir = self._normalize_string(value, "Output directory")
            elif key_lower == "use_llm":
                use_llm = self._parse_bool(value, "Use LLM")
            elif key_lower == "llm_module_dir":
                llm_module_dir = self._normalize_string(value, "LLM module dir")
            elif key_lower == "llm_settings_path":
                llm_settings_path = self._normalize_string(value, "LLM settings path")
            elif key_lower == "stt_model":
                stt_model = self._normalize_string(value, "STT model")
            elif key_lower == "stt_device":
                stt_device = self._normalize_string(value, "STT device")
            elif key_lower == "tts_engine":
                tts_engine = self._normalize_string(value, "TTS engine")
            elif key_lower == "silero_language":
                silero_language = self._normalize_string(value, "Silero language")
            elif key_lower == "silero_variant":
                silero_variant = self._normalize_string(value, "Silero variant")
            elif key_lower == "silero_voice":
                silero_voice = self._normalize_string(value, "Silero voice")
            elif key_lower == "silero_sample_rate":
                silero_sample_rate = self._parse_int(value, "Silero sample rate")
            elif key_lower == "silero_normalize":
                silero_normalize = self._parse_bool(value, "Silero normalize")
            elif key_lower == "orpheus_model":
                orpheus_model = self._normalize_string(value, "Orpheus model")
            elif key_lower == "orpheus_codec":
                orpheus_codec = self._normalize_string(value, "Orpheus codec")
            elif key_lower == "orpheus_device":
                orpheus_device = self._normalize_string(value, "Orpheus device")
            elif key_lower == "orpheus_dtype":
                orpheus_dtype = self._normalize_string(value, "Orpheus dtype")
            elif key_lower == "orpheus_voice":
                orpheus_voice = self._normalize_string(value, "Orpheus voice")
            elif key_lower == "default_emotion":
                default_emotion = self._normalize_string(value, "Default emotion")
            elif key_lower == "default_style":
                default_style = self._normalize_string(value, "Default style")
            elif key_lower == "max_reference_seconds":
                max_reference_seconds = float(value.strip())

        llm_settings = self._load_llm_settings(llm_settings_path)
        logging.info(
            (
                "Config loaded: sample_rate=%s output_dir=%s use_llm=%s "
                "orpheus_model=%s device=%s dtype=%s"
            ),
            sample_rate,
            output_dir,
            use_llm,
            orpheus_model,
            orpheus_device,
            orpheus_dtype,
        )

        return Config(
            sample_rate=sample_rate,
            output_dir=output_dir,
            use_llm=use_llm,
            llm_module_dir=llm_module_dir,
            llm_settings=llm_settings,
            stt_model=stt_model,
            stt_device=stt_device,
            tts_engine=tts_engine,
            silero_language=silero_language,
            silero_variant=silero_variant,
            silero_voice=silero_voice,
            silero_sample_rate=silero_sample_rate,
            silero_normalize=silero_normalize,
            orpheus_model=orpheus_model,
            orpheus_codec=orpheus_codec,
            orpheus_device=orpheus_device,
            orpheus_dtype=orpheus_dtype,
            orpheus_voice=orpheus_voice,
            default_emotion=default_emotion,
            default_style=default_style,
            max_reference_seconds=max_reference_seconds,
        )

    @staticmethod
    def _normalize_string(value: str, label: str) -> str:
        stripped = value.strip().strip('"').strip("'")
        if not stripped:
            raise ValueError(f"{label} value cannot be empty")
        return stripped

    @staticmethod
    def _parse_int(value: str, label: str) -> int:
        cleaned = value.strip()
        try:
            number = int(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer") from exc
        if number <= 0:
            raise ValueError(f"{label} must be greater than zero")
        return number

    @staticmethod
    def _parse_bool(value: str, label: str) -> bool:
        cleaned = value.strip().lower()
        if cleaned in ("true", "yes", "1", "on"):
            return True
        if cleaned in ("false", "no", "0", "off"):
            return False
        raise ValueError(f"{label} must be a boolean (true/false)")

    def _load_llm_settings(self, settings_path: str) -> Dict[str, Any]:
        """Load LLM settings from a YAML file."""
        path = Path(settings_path)

        # If path is relative, resolve it relative to the config file's directory
        if not path.is_absolute():
            path = self._config_path.parent / settings_path

        if not path.exists():
            logging.warning("LLM settings file not found: %s, using defaults", path)
            return {}

        logging.info("Loading LLM settings from %s", path)
        settings: Dict[str, Any] = {}

        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = (part.strip() for part in line.split(":", 1))
                key_lower = key.lower()

                # Parse different types of values
                if key_lower in ("base_url", "model", "tts_prompt_path", "stt_prompt_path"):
                    settings[key] = self._normalize_string(value, f"LLM {key}")
                elif key_lower in ("temperature", "request_timeout", "connect_timeout"):
                    settings[key] = float(value.strip())
                elif key_lower in ("max_tokens", "max_connections"):
                    settings[key] = int(value.strip())

            logging.info("LLM settings loaded: %s", settings)
            return settings
        except Exception as exc:
            logging.warning("Failed to load LLM settings from %s: %s", path, exc)
            return {}
