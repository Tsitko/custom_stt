import math
import os
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from utils.file_handler import FileHandler


class FileHandlerTests(unittest.TestCase):
    def test_missing_wav_raises(self) -> None:
        handler = FileHandler()
        with self.assertRaises(FileNotFoundError):
            handler.save_ogg(Path("nope.wav"), Path("out.ogg"))

    def test_converts_generated_wav_to_ogg(self) -> None:
        handler = FileHandler()
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "tone.wav"
            ogg_path = Path(tmp) / "tone.ogg"
            self._create_sine_wave(wav_path)

            handler.save_ogg(wav_path, ogg_path)
            self.assertTrue(ogg_path.exists())
            self.assertGreater(ogg_path.stat().st_size, 0)

    def _create_sine_wave(self, wav_path: Path) -> None:
        channels = 1
        sample_rate = 22050
        duration_seconds = 0.25
        samples_count = int(sample_rate * duration_seconds)

        samples = array(
            "h",
            [
                int(0.4 * 32767 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
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
