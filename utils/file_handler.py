"""File handling helpers to transform WAV output into OGG Vorbis."""
import audioop
import logging
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.interfaces import IFileHandler


class FileHandler(IFileHandler):
    """Converts synthesized WAV audio into 44.1 kHz OGG Vorbis files."""

    def __init__(self, target_rate: int = 44100) -> None:
        self._target_rate = target_rate

    def save_ogg(self, wav_path: Path, ogg_path: Path) -> Path:
        wav_path = Path(wav_path)
        ogg_path = Path(ogg_path)

        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        ogg_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Reading WAV for conversion: %s", wav_path)
        audio_bytes, channels = self._read_wav_as_pcm(wav_path)
        logging.info("Writing OGG (rate=%s, channels=%s): %s", self._target_rate, channels, ogg_path)
        # Write on a local tmp path first to avoid WSL/remote FS issues, then copy.
        tmp_target = ogg_path
        if ogg_path.drive or str(ogg_path).startswith("/mnt/"):
            import tempfile
            import shutil

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_target = Path(tmp.name)
        self._write_ogg_vorbis(audio_bytes, channels, tmp_target)
        if tmp_target != ogg_path:
            shutil.copyfile(tmp_target, ogg_path)
            tmp_target.unlink(missing_ok=True)

        if not ogg_path.exists() or ogg_path.stat().st_size == 0:
            raise RuntimeError("Failed to create OGG file")

        return ogg_path

    def _read_wav_as_pcm(self, wav_path: Path) -> tuple[bytes, int]:
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            source_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2

        converted, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            source_rate,
            self._target_rate,
            None,
        )
        return converted, channels

    def _write_ogg_vorbis(self, pcm_bytes: bytes, channels: int, ogg_path: Path) -> None:
        # Use ffmpeg via subprocess to convert PCM to OGG Vorbis
        # This avoids torchaudio/soundfile format issues on WSL
        import subprocess
        import tempfile

        # Write PCM data to temporary WAV file first
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        else:
            samples = samples.reshape(-1, 1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)
            # Use wave module to write PCM as WAV (safe, no segfault)
            import wave
            with wave.open(str(tmp_wav_path), 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(self._target_rate)
                wav_file.writeframes(pcm_bytes)

        try:
            # Convert WAV to OGG Vorbis using ffmpeg
            result = subprocess.run(
                [
                    '/usr/bin/ffmpeg', '-y', '-i', str(tmp_wav_path),
                    '-c:a', 'libvorbis', '-q:a', '4',
                    str(ogg_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        finally:
            tmp_wav_path.unlink(missing_ok=True)
