#!/usr/bin/env python3
"""Direct test of Qwen ASR transcriber."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt.qwen_asr_transcriber import QwenASRTranscriber

def main():
    audio_path = Path("one_shot/stalin_reference.wav")
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    print(f"Testing Qwen ASR with: {audio_path}")

    transcriber = QwenASRTranscriber(
        model_name="Qwen/Qwen3-ASR-1.7B",
        device="cuda:0",
        max_audio_seconds=30,
        ttl_seconds=300,
    )

    result = transcriber.transcribe(audio_path)
    print(f"Transcription result: {result}")

if __name__ == "__main__":
    main()
