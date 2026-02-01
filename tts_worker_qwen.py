#!/usr/bin/env python3
"""Persistent Qwen TTS worker process for GPU inference."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Suppress progress bars and warnings before imports
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Redirect any stray prints to stderr
import builtins
_original_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    _original_print(*args, **kwargs)
builtins.print = _stderr_print

from utils.qwen_tts_engine import QwenTTSEngine
from utils.interfaces import VoiceReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Qwen TTS worker.")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = parse_args()

    from utils.config_loader import ConfigLoader
    config = ConfigLoader(args.config).load()

    engine = QwenTTSEngine(
        model_name=config.qwen_tts_model,
        device=config.qwen_tts_device,
        max_text_chars=config.qwen_tts_max_chars,
        crossfade_ms=config.qwen_tts_crossfade_ms,
        temperature=config.qwen_tts_temperature,
        top_p=config.qwen_tts_top_p,
        repetition_penalty=config.qwen_tts_repetition_penalty,
        ttl_seconds=config.model_ttl_seconds,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            text = payload.get("text", "")
            ref_audio = payload.get("ref_audio", "")
            ref_text = payload.get("ref_text", "")
            output_path = payload.get("output_path", "")

            if not text:
                raise ValueError("Text is required")
            if not ref_audio or not ref_text:
                raise ValueError("Reference audio and text are required")

            logging.info("Qwen TTS worker: text=%d chars, ref=%s", len(text), ref_audio)

            references = [VoiceReference(audio_path=Path(ref_audio), transcript=ref_text)]

            synth_start = time.perf_counter()
            result_path = engine.synthesize_to_wav(
                text=text,
                references=references,
                output_path=Path(output_path),
            )
            synth_elapsed = time.perf_counter() - synth_start

            logging.info("Qwen TTS worker: synthesized in %.3fs", synth_elapsed)
            response = {"wav_path": str(result_path)}
        except Exception as exc:
            logging.error("Qwen TTS worker error: %s", exc)
            response = {"error": str(exc)}

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
