import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

import torch

from utils.config_loader import ConfigLoader
from utils.file_handler import FileHandler
from utils.interfaces import VoiceReference
from utils.orpheus_engine import OrpheusTTSEngine


class _DummyCodec:
    def __init__(self) -> None:
        self.encoded = []

    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def encode(self, audio_tensor):
        self.encoded.append(audio_tensor)
        return torch.tensor([0, 1, 2])

    def decode(self, _codes):
        # Return 0.25s of silence at 24 kHz
        return torch.zeros((1, 6000), dtype=torch.float32)


class _DummyModel:
    def __init__(self) -> None:
        self.conditioning = None

    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, conditioning):
        self.conditioning = conditioning
        return torch.tensor([[1, 2, 3]])


class FunctionalPipelineTests(unittest.TestCase):
    def test_synthesize_and_save_ogg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "utils.orpheus_engine._require_orpheus", return_value=(_DummyModel, _DummyCodec, torch)
        ):
            config_path = Path(tmp) / "config.yml"
            config_path.write_text("sample_rate: 24000\n", encoding="utf-8")

            config = ConfigLoader(config_path).load()
            tts_engine = OrpheusTTSEngine(
                model_name=config.orpheus_model,
                codec_name=config.orpheus_codec,
                device="cpu",
                dtype="float32",
                sample_rate=config.sample_rate,
            )
            file_handler = FileHandler(target_rate=config.sample_rate)

            wav_path = Path(tmp) / "speech.wav"
            ogg_path = Path(tmp) / "speech.ogg"
            ref_path = Path(tmp) / "ref.wav"
            self._create_reference_audio(ref_path, sample_rate=config.sample_rate)
            refs = [VoiceReference(audio_path=ref_path, transcript="эталон")]

            tts_engine.synthesize_to_wav("Привет, мир!", refs, wav_path)
            file_handler.save_ogg(wav_path, ogg_path)

            self.assertTrue(ogg_path.exists())
            self.assertGreater(ogg_path.stat().st_size, 0)

    def _create_reference_audio(self, wav_path: Path, sample_rate: int = 24000) -> None:
        channels = 1
        duration_seconds = 0.2
        samples_count = int(sample_rate * duration_seconds)

        samples = array(
            "h",
            [
                int(0.4 * 32767 * math.sin(2 * math.pi * 220 * (i / sample_rate)))
                for i in range(samples_count)
            ],
        )

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())


if __name__ == "__main__":
    unittest.main()
