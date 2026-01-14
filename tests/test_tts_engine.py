import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import torch

from utils.interfaces import VoiceReference
from utils.orpheus_engine import OrpheusTTSEngine


class _DummyCodec:
    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def encode(self, _audio):
        return torch.tensor([1, 2])

    def decode(self, _codes):
        return torch.ones((1, 1000), dtype=torch.float32)


class _DummyModel:
    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs):
        return cls()

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, conditioning):
        return torch.tensor([[len(conditioning)]])


class OrpheusEngineTests(unittest.TestCase):
    def test_empty_text_rejected(self) -> None:
        engine = OrpheusTTSEngine("/home/denis/.cache/orpheus/papacliff/orpheus-3b-0.1-ft-ru", sample_rate=24000)
        ref = VoiceReference(audio_path=Path("ref.wav"), transcript="текст")
        with self.assertRaises(ValueError):
            engine.synthesize_to_wav("", [ref], Path("out.wav"))

    def test_missing_references_rejected(self) -> None:
            engine = OrpheusTTSEngine("/home/denis/.cache/orpheus/papacliff/orpheus-3b-0.1-ft-ru", sample_rate=24000, codec_name=None)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)
            try:
                with self.assertRaises(RuntimeError):
                    engine.synthesize_to_wav("text", [], wav_path)
            finally:
                wav_path.unlink(missing_ok=True)

    def test_generates_wav_with_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.wav"
            self._write_silence(ref_path, sample_rate=24000)
            out_path = Path(tmp) / "out.wav"

            engine = OrpheusTTSEngine("/home/denis/.cache/orpheus/papacliff/orpheus-3b-0.1-ft-ru", sample_rate=24000, codec_name=None)
            ref = VoiceReference(audio_path=ref_path, transcript="пример")
            with self.assertRaises(RuntimeError):
                engine.synthesize_to_wav("тест", [ref], out_path, emotion="happy", style="news")

    def test_allows_no_references_when_no_codec(self) -> None:
        # When codec is absent, references can be empty; generation will still try to run and likely fail,
        # but should not raise reference validation.
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.wav"
            engine = OrpheusTTSEngine("/home/denis/.cache/orpheus/papacliff/orpheus-3b-0.1-ft-ru", codec_name=None, sample_rate=24000)
            with self.assertRaises(RuntimeError):
                engine.synthesize_to_wav("тест", [], out_path)

    def _write_silence(self, path: Path, sample_rate: int) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * 100)


if __name__ == "__main__":
    unittest.main()
