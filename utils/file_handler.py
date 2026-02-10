"""File handling helpers to transform WAV output into OGG Vorbis."""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

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
        logging.info("Converting WAV -> OGG (rate=%s): %s -> %s", self._target_rate, wav_path, ogg_path)
        # Write on a local tmp path first to avoid WSL/remote FS issues, then copy.
        tmp_target = ogg_path
        if ogg_path.drive or str(ogg_path).startswith("/mnt/"):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_target = Path(tmp.name)
        self._write_ogg_vorbis(wav_path, tmp_target)
        if tmp_target != ogg_path:
            shutil.copyfile(tmp_target, ogg_path)
            tmp_target.unlink(missing_ok=True)

        if not ogg_path.exists() or ogg_path.stat().st_size == 0:
            raise RuntimeError("Failed to create OGG file")

        return ogg_path

    def _write_ogg_vorbis(self, wav_path: Path, ogg_path: Path) -> None:
        # ffmpeg handles decode + resample + OGG encode in one pass.
        result = subprocess.run(
            [
                "/usr/bin/ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-ar",
                str(self._target_rate),
                "-c:a",
                "libvorbis",
                "-q:a",
                "4",
                str(ogg_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
