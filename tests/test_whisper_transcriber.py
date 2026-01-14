import importlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from stt.whisper_transcriber import WhisperTranscriber

REAL_IMPORT = importlib.import_module


class WhisperTranscriberTests(unittest.TestCase):
    @patch("stt.whisper_transcriber.importlib.import_module")
    @patch("stt.whisper_transcriber.librosa.load")
    def test_transcribe_uses_model(self, mock_load, mock_import_module) -> None:
        def side_effect(name, package=None):
            if name == "faster_whisper":
                fake_fw = types.SimpleNamespace()
                fake_fw.WhisperModel = MagicMock(return_value=mock_model_instance)
                return fake_fw
            return REAL_IMPORT(name, package)

        fake_segment = MagicMock()
        fake_segment.text = " тест "
        fake_info = MagicMock(language="ru", duration=1.0)
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = ([fake_segment], fake_info)

        mock_import_module.side_effect = side_effect
        mock_load.return_value = (MagicMock(), 16000)

        transcriber = WhisperTranscriber(model_name="base", device="cpu", compute_type="int8", beam_size=1)
        text = transcriber.transcribe(Path(__file__))

        mock_model_instance.transcribe.assert_called_once()
        self.assertEqual(text, "тест")


if __name__ == "__main__":
    unittest.main()
