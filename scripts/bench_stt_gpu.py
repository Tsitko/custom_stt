#!/usr/bin/env python3
"""Warm benchmark for GigaAM STT (keeps model in memory)."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from stt.gigaam_transcriber import GigaAMTranscriber
from utils.config_loader import ConfigLoader
from utils.llm_preprocessor import LLMPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm benchmark for STT (GigaAM).")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--device", default="cuda", help="Device override (cuda/cpu/auto).")
    parser.add_argument("--model", default=None, help="Model override (e.g. v2_rnnt).")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs.")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM postprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    config = ConfigLoader(args.config).load()
    model_name = args.model or config.stt_model
    device = args.device or config.stt_device

    transcriber = GigaAMTranscriber(model_name=model_name, device=device)

    postprocessor = None
    if args.with_llm:
        postprocessor = LLMPreprocessor(
            module_dir=config.llm_module_dir,
            mode="stt",
            llm_settings=config.llm_settings,
        )

    for _ in range(max(0, args.warmup)):
        _ = transcriber.transcribe(audio_path)

    timings: list[float] = []
    last_text = ""
    for _ in range(max(1, args.repeats)):
        start = time.perf_counter()
        text = transcriber.transcribe(audio_path)
        if postprocessor:
            text = postprocessor.process(text)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        last_text = text

    avg = statistics.mean(timings)
    med = statistics.median(timings)
    print(f"runs={len(timings)} warmup={args.warmup} device={device} model={model_name}")
    print(f"avg={avg:.3f}s median={med:.3f}s min={min(timings):.3f}s max={max(timings):.3f}s")
    print(f"text={last_text}")


if __name__ == "__main__":
    main()
