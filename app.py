"""Console entry point for local Russian TTS synthesis."""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import logging
from typing import List

from utils.config_loader import ConfigLoader
from utils.file_handler import FileHandler
from utils.llm_preprocessor import LLMPreprocessor
from utils.orpheus_engine import OrpheusTTSEngine
from utils.silero_engine import SileroTTSEngine
from utils.interfaces import VoiceReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OGG speech from text using Orpheus voice cloning.")
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize (UTF-8).",
    )
    parser.add_argument(
        "--ref-audio",
        action="append",
        help="Path to reference WAV/FLAC/OGG with target speaker (repeatable).",
    )
    parser.add_argument(
        "--ref-text",
        action="append",
        help="Transcript for the corresponding reference audio (repeatable, same order as --ref-audio).",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="Directory with reference pairs: *.wav and matching *.txt (same stem).",
    )
    parser.add_argument(
        "--emotion",
        default=None,
        help="Optional emotion tag (<emotion:...>), e.g. happy, sad, neutral. Overrides config default.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Optional style tag (<style:...>), e.g. news, audiobook. Overrides config default.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override target sample rate for synthesis and output (default from config).",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to config.yml (default: config.yml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the resulting OGG file. Defaults to outputs/<timestamp>.ogg",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = ConfigLoader(args.config).load()

    text = args.text
    if config.use_llm:
        preprocessor = LLMPreprocessor(
            module_dir=config.llm_module_dir,
            mode="tts",
            llm_settings=config.llm_settings,
        )
        logging.info("Запуск LLM предобработки текста (TTS)")
        text = preprocessor.process(text)

    sample_rate = args.sample_rate or config.sample_rate
    emotion = args.emotion or config.default_emotion
    style = args.style or config.default_style
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path = Path(output_path)
    if not args.output:
        output_path = Path(config.output_dir) / output_path.name

    engine_choice = (config.tts_engine or "orpheus").lower()
    references: list[VoiceReference] = []
    if engine_choice == "silero":
        tts_engine = SileroTTSEngine(
            voice=config.silero_voice,
            language=config.silero_language,
            variant=config.silero_variant,
            sample_rate=sample_rate,
            normalize=config.silero_normalize,
        )
    else:
        references = _build_references(args.ref_audio, args.ref_text, args.ref_dir)
        if not references:
            raise ValueError("Нужен хотя бы один эталон (--ref-dir или пары --ref-audio/--ref-text) для Orpheus")
        tts_engine = OrpheusTTSEngine(
            model_name=config.orpheus_model,
            codec_name=config.orpheus_codec,
            device=config.orpheus_device,
            dtype=config.orpheus_dtype,
            sample_rate=sample_rate,
            max_reference_seconds=config.max_reference_seconds,
        )
    file_handler = FileHandler(target_rate=sample_rate)

    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        tts_engine.synthesize_to_wav(text, references, wav_path, emotion=emotion, style=style)
        file_handler.save_ogg(wav_path, output_path)

        print(f"Синтез завершён. OGG-файл сохранён: {output_path}")
    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def _build_references(audio_paths: List[str], transcripts: List[str]) -> list[VoiceReference]:
    raise ValueError("Количество --ref-audio и --ref-text должно совпадать")


def _build_references(
    audio_paths: List[str] | None,
    transcripts: List[str] | None,
    ref_dir: str | None,
) -> list[VoiceReference]:
    if ref_dir:
        return _load_references_from_dir(Path(ref_dir))
    if not audio_paths or not transcripts:
        # Allow empty references when codec is disabled (text-only synthesis).
        return []
    if len(audio_paths) != len(transcripts):
        raise ValueError("Количество --ref-audio и --ref-text должно совпадать")
    refs: list[VoiceReference] = []
    for audio, text in zip(audio_paths, transcripts):
        refs.append(VoiceReference(audio_path=Path(audio), transcript=text))
    return refs


def _load_references_from_dir(directory: Path) -> list[VoiceReference]:
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"--ref-dir не существует или не является папкой: {directory}")
    audio_files = sorted(directory.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"В {directory} не найдено файлов .wav")
    references: list[VoiceReference] = []
    for audio_path in audio_files:
        transcript_path = audio_path.with_suffix(".txt")
        if not transcript_path.exists():
            raise ValueError(f"Не найден transcript для {audio_path.name} (ожидался {transcript_path.name})")
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError(f"Пустой transcript: {transcript_path}")
        references.append(VoiceReference(audio_path=audio_path, transcript=transcript))
    return references


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"speech_{timestamp}.ogg"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - guard rails for CLI use
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
