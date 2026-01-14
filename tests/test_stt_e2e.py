import logging
import os
from pathlib import Path
import unittest

import soundfile as sf

from stt.whisper_transcriber import WhisperTranscriber
from utils.llm_preprocessor import LLMPreprocessor
from tests.test_e2e_minute import E2EMinuteSpeechTest


class STTE2ETest(unittest.TestCase):
    def test_transcribe_e2e_minute(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        ogg_path = Path("outputs") / "e2e_minute.ogg"
        txt_path = Path("outputs") / "e2e_minute.txt"
        raw_txt_path = Path("outputs") / "e2e_minute_raw_stt.txt"
        wav_path = Path("outputs") / "e2e_minute.wav"
        ogg_path.parent.mkdir(parents=True, exist_ok=True)

        if not ogg_path.exists():
            # Generate audio if previous e2e test has not run
            generator = E2EMinuteSpeechTest()
            generator.test_minute_speech_saved()

        # Convert OGG -> WAV using soundfile to avoid external ffmpeg dependency
        audio, sr = sf.read(str(ogg_path))
        sf.write(str(wav_path), audio, sr)

        transcriber = WhisperTranscriber(model_name="base", device="cpu", compute_type="int8", beam_size=1)
        raw_text = transcriber.transcribe(wav_path)
        print(f"[STT raw output] {raw_text}")
        raw_txt_path.write_text(raw_text, encoding="utf-8")
        postprocessor = LLMPreprocessor(
            mode="stt",
            llm_settings={"model": "qwen2.5:7b"},
        )
        processed_text = postprocessor.process(raw_text)
        print(f"[LLM STT output] {processed_text}")

        txt_path.write_text(processed_text, encoding="utf-8")

        self.assertTrue(txt_path.exists())
        self.assertGreater(os.path.getsize(txt_path), 0)
        self.assertTrue(raw_txt_path.exists())
        self.assertGreater(raw_txt_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
