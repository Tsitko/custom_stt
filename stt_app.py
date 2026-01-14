"""Console entry point for speech-to-text using faster-whisper."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from utils.config_loader import ConfigLoader
from stt.gigaam_transcriber import GigaAMTranscriber
from utils.llm_preprocessor import LLMPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio to text using faster-whisper.")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/ogg/flac/mp3).")
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to config.yml (default: config.yml).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON with raw_text and processed_text.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = ConfigLoader(args.config).load()

    transcriber = GigaAMTranscriber(
        model_name=config.stt_model,
        device=config.stt_device,
    )

    raw_text = transcriber.transcribe(Path(args.audio))

    postprocessor = LLMPreprocessor(
        module_dir=config.llm_module_dir,
        mode="stt",
        llm_settings=config.llm_settings,
    )
    logging.info("Запуск постобработки распознанного текста через LLM")
    processed_text = postprocessor.process(raw_text)
    logging.info("Текст после LLM (STT): %s", processed_text)

    if args.json:
        print(json.dumps({"raw_text": raw_text, "processed_text": processed_text}, ensure_ascii=False))
    else:
        print(processed_text)


if __name__ == "__main__":
    main()
