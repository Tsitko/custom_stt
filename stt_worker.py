#!/usr/bin/env python3
"""Persistent STT worker process for GPU inference."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from stt.gigaam_transcriber import GigaAMTranscriber
from utils.config_loader import ConfigLoader
from utils.llm_preprocessor import LLMPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent GigaAM STT worker.")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu/auto).")
    parser.add_argument("--model", default=None, help="Model override.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    args = parse_args()

    config = ConfigLoader(args.config).load()
    model_name = args.model or config.stt_model
    device = args.device or config.stt_device

    transcriber = GigaAMTranscriber(model_name=model_name, device=device)

    postprocessor = None
    if config.use_llm:
        postprocessor = LLMPreprocessor(
            module_dir=config.llm_module_dir,
            mode="stt",
            llm_settings=config.llm_settings,
        )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            audio_path = Path(payload.get("audio_path", ""))
            context = payload.get("context")
            if context is not None and not isinstance(context, str):
                context = str(context)
            context_len = len(context) if context else 0
            logging.info("STT worker received context length: %s", context_len)
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            infer_start = time.perf_counter()
            raw_text = transcriber.transcribe(audio_path)
            infer_elapsed = time.perf_counter() - infer_start
            processed_text = raw_text
            llm_elapsed = 0.0
            if postprocessor:
                llm_start = time.perf_counter()
                processed_text = postprocessor.process(raw_text, context=context)
                llm_elapsed = time.perf_counter() - llm_start
            logging.info("STT worker timings: infer=%.3fs llm=%.3fs", infer_elapsed, llm_elapsed)
            response = {"raw_text": raw_text, "processed_text": processed_text}
        except Exception as exc:
            response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
