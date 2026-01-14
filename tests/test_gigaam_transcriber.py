import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import stt.gigaam_transcriber as module_under_test
from stt.gigaam_transcriber import GigaAMTranscriber


class GigaAMTranscriberTests(unittest.TestCase):
    @patch.object(module_under_test, "_GIGAAM")
    def test_transcribe_calls_model(self, mock_gigaam) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "привет мир"
        mock_model.to.return_value = mock_model
        mock_gigaam.load_model.return_value = mock_model

        transcriber = GigaAMTranscriber(model_name="e2e_rnnt", device="cpu")
        with patch.object(Path, "exists", return_value=True):
            text = transcriber.transcribe(Path("dummy.wav"))

        mock_gigaam.load_model.assert_called_once_with("e2e_rnnt")
        mock_model.transcribe.assert_called_once()
        self.assertEqual(text, "привет мир")


if __name__ == "__main__":
    unittest.main()
