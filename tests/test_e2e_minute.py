import tempfile
import unittest
from pathlib import Path
import logging

from utils.config_loader import Config
from utils.file_handler import FileHandler
from utils.llm_preprocessor import LLMPreprocessor
from utils.silero_engine import SileroTTSEngine


class E2EMinuteSpeechTest(unittest.TestCase):
    def test_minute_speech_saved(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        text = (
            "Привет, это длинный тестовый монолог почти на минуту, "
            "содержит технические термины вроде GPU, CUDA, Kubernetes, "
            "и сложные слова: электрофизиология, поликарпаты, идемпотентность. "
            "Говорим о микросервисах, балансировке нагрузки, latency и throughput, "
            "вспоминаем Витгенштейна, лингвистику и аккуратное произнесение сложных имен, "
            "например электрокардиограмма и нейроморфные процессоры. "
            "Упоминаем NASA, CPU, GPU, RAM, SSD, NVMe, "
            "числа две тысячи двадцать четыре и девяносто девять процентов, "
            "а затем обсуждаем распределённые транзакции и двухфазную фиксацию. "
            "Добавляем биологию: митохондрии, рибосомы — всё это нужно произнести с паузами. "
            "Завершаем упоминанием трансформеров, GPT и фразы attention is all you need, "
            "чтобы проверить, как синтезатор справляется с англицизмами и акцентами."
        )

        config = Config(
            voice="eugene",
            language="ru",
            sample_rate=48000,
            output_dir="outputs",
            engine="silero",
            normalize=True,
            use_llm=True,
            llm_module_dir=None,
            llm_settings={"model": "qwen2.5:7b"},
        )

        wav_path = None
        output_path = Path("outputs") / "e2e_minute.ogg"
        processed_txt_path = Path("outputs") / "e2e_minute_preprocessed.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)

            preprocessor = LLMPreprocessor(
                mode="tts",
                llm_settings=config.llm_settings,
            )
            processed_text = preprocessor.process(text)
            print(f"[LLM TTS output] {processed_text}")
            processed_txt_path.write_text(processed_text, encoding="utf-8")

            tts = SileroTTSEngine(
                config.voice,
                language=config.language,
                sample_rate=config.sample_rate,
                normalize=config.normalize,
            )
            handler = FileHandler(target_rate=config.sample_rate)

            tts.synthesize_to_wav(processed_text, wav_path)
            handler.save_ogg(wav_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertTrue(processed_txt_path.exists())
            self.assertGreater(processed_txt_path.stat().st_size, 0)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
