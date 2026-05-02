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
RUNTIME_PATCH_MODULE = "colab_202604_runtime_patch"
PTH_FILENAME = "style_bert_vits2_colab_202604.pth"

TORCHAUDIO_PATCH_BODY = '''# Style-Bert-VITS2 Colab 2026.04 compatibility shims

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

try:
    import inspect
    import torch
except Exception:
    torch = None

if torch is not None and not getattr(torch.load, "_style_bert_vits2_colab_202604", False):
    _original_torch_load = torch.load

    def _torch_load_for_lightning_cloud_io(*args, **kwargs):
        for frame in inspect.stack()[1:8]:
            filename = frame.filename.replace("\\\\", "/")
            if filename.endswith("lightning_fabric/utilities/cloud_io.py"):
                kwargs["weights_only"] = False
                break
        return _original_torch_load(*args, **kwargs)

    _torch_load_for_lightning_cloud_io._style_bert_vits2_colab_202604 = True
    torch.load = _torch_load_for_lightning_cloud_io
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
        path.write_text(existing + separator + TORCHAUDIO_PATCH_BODY, encoding="utf-8")


def site_packages_path() -> Path:
    candidates = site.getsitepackages()
    if candidates:
        return Path(candidates[0])
    return Path(sysconfig_path())


def write_python_startup_hooks() -> list[Path]:
    repo_root = Path.cwd()
    site_packages = site_packages_path()
    paths = [sitecustomize_path(), repo_root / "sitecustomize.py"]
    written_paths = []
    for path in paths:
        append_sitecustomize(path)
        written_paths.append(path)

    runtime_patch = repo_root / f"{RUNTIME_PATCH_MODULE}.py"
    runtime_patch.write_text(TORCHAUDIO_PATCH_BODY, encoding="utf-8")
    written_paths.append(runtime_patch)

    pth_path = site_packages / PTH_FILENAME
    pth_line = (
        f"import sys; sys.path.insert(0, {str(repo_root)!r}); "
        f"import {RUNTIME_PATCH_MODULE}\n"
    )
    pth_path.write_text(pth_line, encoding="utf-8")
    written_paths.append(pth_path)
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


def verify_subprocess_torch_load_patch() -> bool:
    import subprocess

    code = (
        "import torch; "
        "print(getattr(torch.load, '_style_bert_vits2_colab_202604', False))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1] == "True"


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
    startup_hook_paths = write_python_startup_hooks()
    subprocess_has_backends, subprocess_has_metadata = verify_subprocess_torchaudio()
    subprocess_has_torch_load_patch = verify_subprocess_torch_load_patch()
    cloud_io_patched, cloud_io_path = patch_lightning_cloud_io()

    print("Colab 2026.04 compatibility check")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"torchaudio.list_audio_backends: {has_backends}")
    print(f"torchaudio.AudioMetaData: {has_metadata}")
    for path in startup_hook_paths:
        print(f"python startup hook: {path}")
    print(f"subprocess torchaudio.list_audio_backends: {subprocess_has_backends}")
    print(f"subprocess torchaudio.AudioMetaData: {subprocess_has_metadata}")
    print(f"subprocess torch.load cloud_io patch: {subprocess_has_torch_load_patch}")
    print(f"lightning cloud_io patched: {cloud_io_patched}")
    if cloud_io_path is not None:
        print(f"lightning cloud_io: {cloud_io_path}")
    if not cloud_io_patched:
        raise RuntimeError("Failed to patch lightning_fabric.utilities.cloud_io")


if __name__ == "__main__":
    main()
