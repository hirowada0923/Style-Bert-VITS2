"""Compatibility fixes for Google Colab runtime 2026.04 / latest.

This script keeps Colab-specific workarounds out of the notebook and the
upstream Style-Bert-VITS2 source tree.
"""

from __future__ import annotations

import importlib.util
import site
import sys
from pathlib import Path
from typing import NamedTuple


SITECUSTOMIZE_MARKER = "Style-Bert-VITS2 Colab 2026.04 compatibility shims"

SITECUSTOMIZE_BODY = '''# Style-Bert-VITS2 Colab 2026.04 compatibility shims

from typing import NamedTuple

try:
    import torchaudio
except Exception:
    torchaudio = None

if torchaudio is not None:
    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            return ["soundfile", "ffmpeg"]

        torchaudio.list_audio_backends = list_audio_backends

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData(NamedTuple):
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData
'''


def patch_current_torchaudio() -> tuple[bool, bool]:
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            return ["soundfile", "ffmpeg"]

        torchaudio.list_audio_backends = list_audio_backends

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData(NamedTuple):
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData

    return (
        hasattr(torchaudio, "list_audio_backends"),
        hasattr(torchaudio, "AudioMetaData"),
    )


def sitecustomize_path() -> Path:
    candidates = site.getsitepackages()
    if not candidates:
        candidates = [sysconfig_path()]
    return Path(candidates[0]) / "sitecustomize.py"


def sysconfig_path() -> str:
    import sysconfig

    return sysconfig.get_paths()["purelib"]


def append_sitecustomize(path: Path) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SITECUSTOMIZE_MARKER not in existing:
        separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
        path.write_text(existing + separator + SITECUSTOMIZE_BODY, encoding="utf-8")


def write_sitecustomize() -> list[Path]:
    paths = [sitecustomize_path(), Path.cwd() / "sitecustomize.py"]
    written_paths = []
    for path in paths:
        append_sitecustomize(path)
        written_paths.append(path)
    return written_paths


def verify_subprocess_torchaudio() -> tuple[bool, bool]:
    import subprocess

    code = (
        "import torchaudio; "
        "print(hasattr(torchaudio, 'list_audio_backends')); "
        "print(hasattr(torchaudio, 'AudioMetaData'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    return tuple(line == "True" for line in lines[-2:])


def patch_lightning_cloud_io() -> tuple[bool, Path | None]:
    spec = importlib.util.find_spec("lightning_fabric.utilities.cloud_io")
    if spec is None or spec.origin is None:
        return False, None

    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")
    if "weights_only=False" in text:
        return True, path

    patched = text.replace("weights_only=weights_only", "weights_only=False")
    if patched == text:
        return False, path

    path.write_text(patched, encoding="utf-8")
    return True, path


def main() -> None:
    import torch
    import torchaudio

    has_backends, has_metadata = patch_current_torchaudio()
    sitecustomize_paths = write_sitecustomize()
    subprocess_has_backends, subprocess_has_metadata = verify_subprocess_torchaudio()
    cloud_io_patched, cloud_io_path = patch_lightning_cloud_io()

    print("Colab 2026.04 compatibility check")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"torchaudio.list_audio_backends: {has_backends}")
    print(f"torchaudio.AudioMetaData: {has_metadata}")
    for path in sitecustomize_paths:
        print(f"sitecustomize: {path}")
    print(f"subprocess torchaudio.list_audio_backends: {subprocess_has_backends}")
    print(f"subprocess torchaudio.AudioMetaData: {subprocess_has_metadata}")
    print(f"lightning cloud_io patched: {cloud_io_patched}")
    if cloud_io_path is not None:
        print(f"lightning cloud_io: {cloud_io_path}")
    if not cloud_io_patched:
        raise RuntimeError("Failed to patch lightning_fabric.utilities.cloud_io")


if __name__ == "__main__":
    main()
