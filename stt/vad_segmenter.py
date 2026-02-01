from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import webrtcvad


@dataclass(frozen=True)
class _Frame:
    bytes: bytes
    timestamp: float
    duration: float


def _frame_generator(
    audio_int16: torch.Tensor, sample_rate: int, frame_ms: int
) -> List[_Frame]:
    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size <= 0:
        raise ValueError("frame_ms too small")
    audio = audio_int16.cpu().contiguous()
    if audio.dtype != torch.int16:
        audio = audio.to(torch.int16)
    total_len = int(audio.numel())
    frames: List[_Frame] = []
    offset = 0
    timestamp = 0.0
    duration = frame_size / sample_rate
    while offset + frame_size <= total_len:
        chunk = audio[offset : offset + frame_size]
        try:
            frame_bytes = chunk.numpy().tobytes()
        except Exception as exc:
            raise RuntimeError("numpy is required for VAD segmentation") from exc
        frames.append(_Frame(frame_bytes, timestamp, duration))
        offset += frame_size
        timestamp += duration
    return frames


def _vad_collect(
    vad: webrtcvad.Vad,
    frames: Iterable[_Frame],
    frame_ms: int,
    padding_ms: int,
    sample_rate: int,
) -> List[Tuple[float, float]]:
    num_padding_frames = max(1, int(padding_ms / frame_ms))
    ring: deque[Tuple[_Frame, bool]] = deque(maxlen=num_padding_frames)
    triggered = False
    segments: List[Tuple[float, float]] = []
    start_time = 0.0
    frames_list = list(frames)
    for frame in frames_list:
        is_speech = vad.is_speech(frame.bytes, sample_rate=sample_rate)
        if not triggered:
            ring.append((frame, is_speech))
            num_voiced = sum(1 for _f, speech in ring if speech)
            if num_voiced > 0.9 * ring.maxlen:
                triggered = True
                start_time = ring[0][0].timestamp
                ring.clear()
        else:
            ring.append((frame, is_speech))
            num_unvoiced = sum(1 for _f, speech in ring if not speech)
            if num_unvoiced > 0.9 * ring.maxlen:
                end_time = frame.timestamp + frame.duration
                segments.append((start_time, end_time))
                triggered = False
                ring.clear()
    if triggered and frames_list:
        end_time = frames_list[-1].timestamp + frames_list[-1].duration
        segments.append((start_time, end_time))
    return segments


def segment_audio_vad(
    audio_int16: torch.Tensor,
    sample_rate: int,
    *,
    frame_ms: int = 30,
    padding_ms: int = 300,
    vad_mode: int = 2,
    max_duration: float = 22.0,
    min_duration: float = 0.5,
    max_gap: float = 0.2,
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    if sample_rate != 16000:
        raise ValueError("segment_audio_vad expects 16kHz audio")
    if audio_int16.dim() != 1:
        audio_int16 = audio_int16.view(-1)
    duration = float(audio_int16.numel()) / sample_rate
    if duration <= max_duration:
        audio_float = audio_int16.float() / 32768.0
        return [audio_float], [(0.0, duration)]

    vad = webrtcvad.Vad(vad_mode)
    frames = _frame_generator(audio_int16, sample_rate, frame_ms)
    speech_segments = _vad_collect(vad, frames, frame_ms, padding_ms, sample_rate)
    if not speech_segments:
        audio_float = audio_int16.float() / 32768.0
        return [audio_float], [(0.0, duration)]

    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = speech_segments[0]
    for start, end in speech_segments[1:]:
        if start - cur_end <= max_gap and (end - cur_start) <= max_duration:
            cur_end = end
        else:
            if cur_end - cur_start >= min_duration:
                merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_end - cur_start >= min_duration:
        merged.append((cur_start, cur_end))

    chunks: List[Tuple[float, float]] = []
    for start, end in merged:
        while end - start > max_duration:
            chunks.append((start, start + max_duration))
            start += max_duration
        chunks.append((start, end))

    audio_float = audio_int16.float() / 32768.0
    tensors: List[torch.Tensor] = []
    for start, end in chunks:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        tensors.append(audio_float[s:e])
    return tensors, chunks
