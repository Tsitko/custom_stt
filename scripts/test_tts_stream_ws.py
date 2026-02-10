#!/usr/bin/env python3
"""Test client for WS /tts/speak-streaming endpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test WS TTS streaming endpoint.")
    parser.add_argument("--url", default="ws://127.0.0.1:8013/tts/speak-streaming", help="WebSocket URL.")
    parser.add_argument("--text", default="привет, ребята", help="Text to synthesize.")
    parser.add_argument("--speak-split", type=int, default=0, help="Split text into N-char Speak messages (0=single).")
    parser.add_argument("--out", default="outputs/ws_stream_test.wav", help="Output WAV path.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Receive timeout in seconds.")
    return parser.parse_args()


def split_text(text: str, n: int) -> list[str]:
    if n <= 0 or len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(0, len(text), n)]


async def run() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav_parts: list[bytes] = []
    metadata = None
    flushed = None
    warnings: list[dict] = []
    first_audio_latency = None

    t0 = time.perf_counter()
    async with websockets.connect(args.url, max_size=50 * 1024 * 1024) as ws:
        t_connected = time.perf_counter()

        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            if isinstance(msg, str):
                payload = json.loads(msg)
                if payload.get("type") == "Metadata":
                    metadata = payload
        except TimeoutError:
            print("No Metadata received (timeout)")
            return 2

        for piece in split_text(args.text, args.speak_split):
            await ws.send(json.dumps({"type": "Speak", "text": piece}, ensure_ascii=False))

        t_flush = time.perf_counter()
        await ws.send(json.dumps({"type": "Flush"}))

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
            except TimeoutError:
                print("Timeout while waiting for stream data")
                return 3

            if isinstance(msg, bytes):
                wav_parts.append(msg)
                if first_audio_latency is None:
                    first_audio_latency = time.perf_counter() - t_flush
                continue

            payload = json.loads(msg)
            ptype = payload.get("type")
            if ptype == "Warning":
                warnings.append(payload)
            elif ptype == "Flushed":
                flushed = payload
                break

        await ws.send(json.dumps({"type": "Close"}))

    total_elapsed = time.perf_counter() - t0
    audio_bytes = b"".join(wav_parts)
    out_path.write_bytes(audio_bytes)

    print(f"url={args.url}")
    print(f"connected_s={t_connected - t0:.3f}")
    print(f"first_audio_after_flush_s={(first_audio_latency if first_audio_latency is not None else -1):.3f}")
    print(f"total_s={total_elapsed:.3f}")
    print(f"audio_bytes={len(audio_bytes)} chunks={len(wav_parts)}")
    print(f"output={out_path}")
    print(f"metadata={json.dumps(metadata, ensure_ascii=False)}")
    print(f"flushed={json.dumps(flushed, ensure_ascii=False)}")
    if warnings:
        print(f"warnings={json.dumps(warnings, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
