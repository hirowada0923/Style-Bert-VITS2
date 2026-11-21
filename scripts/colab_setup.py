"""Create and verify an isolated Python 3.12 virtualenv for Google Colab.

This script runs on Colab's host Python (3.13+) using only the standard library,
provisions `.venv` via `uv`, installs pinned dependencies, and runs in-venv
verification without touching upstream Style-Bert-VITS2 source files or the
Colab global Python environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path


COMPAT_MODULE_NAME = "_style_bert_vits2_colab_compat"
COMPAT_PTH_NAME = "style_bert_vits2_colab_compat.pth"

COMPAT_MODULE_BODY = '''"""Runtime compatibility shims for Style-Bert-VITS2 inside Colab .venv."""

from __future__ import annotations

import os
from typing import NamedTuple


if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"


def _parse_major_minor(version_str: str) -> tuple[int, int]:
    base = version_str.split("+", 1)[0]
    parts = base.split(".")
    major_digits = "".join(ch for ch in parts[0] if ch.isdigit())
    major = int(major_digits) if major_digits else 0
    if len(parts) > 1:
        minor_digits = "".join(ch for ch in parts[1] if ch.isdigit())
        minor = int(minor_digits) if minor_digits else 0
    else:
        minor = 0
    return (major, minor)


try:
    import torchaudio
except Exception:
    torchaudio = None

if torchaudio is not None:
    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends() -> list[str]:
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

if (
    torch is not None
    and _parse_major_minor(torch.__version__) >= (2, 6)
    and not getattr(torch.load, "_style_bert_vits2_colab_compat", False)
):
    _original_torch_load = torch.load

    def _torch_load_for_lightning_cloud_io(*args, **kwargs):
        for frame in inspect.stack()[1:8]:
            filename = frame.filename.replace("\\\\", "/")
            if filename.endswith("lightning_fabric/utilities/cloud_io.py"):
                kwargs["weights_only"] = False
                break
        return _original_torch_load(*args, **kwargs)

    _torch_load_for_lightning_cloud_io._style_bert_vits2_colab_compat = True
    torch.load = _torch_load_for_lightning_cloud_io
'''


def _parse_major_minor(version_str: str) -> tuple[int, int]:
    base = version_str.split("+", 1)[0]
    parts = base.split(".")
    major_digits = "".join(ch for ch in parts[0] if ch.isdigit())
    major = int(major_digits) if major_digits else 0
    if len(parts) > 1:
        minor_digits = "".join(ch for ch in parts[1] if ch.isdigit())
        minor = int(minor_digits) if minor_digits else 0
    else:
        minor = 0
    return (major, minor)


def _find_uv_command() -> list[str]:
    uv_bin = shutil.which("uv")
    if uv_bin:
        return [uv_bin]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    raise RuntimeError("uv not found. Run: python3 -m pip install -q uv")


def _venv_python_path(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _site_packages_dir() -> Path:
    candidates = site.getsitepackages()
    if candidates:
        return Path(candidates[0])
    return Path(sysconfig.get_paths()["purelib"])


def _install_venv_startup_hook() -> Path:
    site_packages = _site_packages_dir()
    module_path = site_packages / f"{COMPAT_MODULE_NAME}.py"
    pth_path = site_packages / COMPAT_PTH_NAME

    module_path.write_text(COMPAT_MODULE_BODY, encoding="utf-8")
    pth_path.write_text(f"import {COMPAT_MODULE_NAME}\n", encoding="utf-8")

    # Apply immediately to the current verification process as well.
    if COMPAT_MODULE_NAME in sys.modules:
        del sys.modules[COMPAT_MODULE_NAME]
    importlib.import_module(COMPAT_MODULE_NAME)
    return pth_path


def _verify_subprocess() -> None:
    code = (
        "import matplotlib, soxr, torchaudio, pyannote.audio, transformers, librosa, "
        "pyopenjtalk, tensorboard, pkg_resources; "
        "from transformers import WhisperProcessor, pipeline; "
        "assert hasattr(torchaudio, 'list_audio_backends'), 'missing list_audio_backends'; "
        "assert hasattr(torchaudio, 'AudioMetaData'), 'missing AudioMetaData'; "
        "assert matplotlib.get_backend().lower() == 'agg', "
        "f'unexpected matplotlib backend: {matplotlib.get_backend()}'; "
        "print('subprocess_imports_ok=True')"
    )
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or lines[-1] != "subprocess_imports_ok=True":
        raise RuntimeError(f"Unexpected subprocess verification output: {result.stdout!r}")


def verify_in_venv() -> None:
    py_ver = sys.version.split()[0]
    print(f"python: {py_ver} ({sys.executable})")
    if sys.version_info >= (3, 13):
        print(
            f"[WARNING] Expected Python 3.12 inside .venv, got {py_ver}. "
            "Legacy audio/build dependencies may fail on Python 3.13+."
        )

    import numpy
    import torch
    import torchaudio

    np_major = _parse_major_minor(numpy.__version__)[0]
    assert np_major < 2, f"Expected numpy<2, got {numpy.__version__}"

    torch_mm = _parse_major_minor(torch.__version__)
    cuda_available = torch.cuda.is_available()
    print(f"numpy: {numpy.__version__}")
    print(f"torch: {torch.__version__} (cuda_available={cuda_available})")
    if cuda_available:
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "[WARNING] CUDA is not available in this Colab session. "
            "Switch Colab runtime to a GPU backend (e.g., T4, L4, or A100) before training."
        )

    has_backends = hasattr(torchaudio, "list_audio_backends")
    has_metadata = hasattr(torchaudio, "AudioMetaData")
    print(
        f"torchaudio: {torchaudio.__version__} "
        f"(list_audio_backends={has_backends}, AudioMetaData={has_metadata})"
    )

    needs_torchaudio_shim = not (has_backends and has_metadata)
    needs_torch_load_shim = torch_mm >= (2, 6)
    pth_path = _install_venv_startup_hook()
    print(
        f"compat hook installed: {pth_path} "
        f"(torchaudio_shim={needs_torchaudio_shim}, torch_load_shim={needs_torch_load_shim})"
    )

    import gradio
    import huggingface_hub
    import librosa
    import matplotlib
    import pkg_resources  # noqa: F401
    import pydantic
    import pyopenjtalk
    import pyworld
    import sentencepiece  # noqa: F401
    import soundfile
    import soxr
    import tensorboard
    import transformers
    from pyannote.audio import Inference, Model  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModel,
        AutoModelForMaskedLM,
        AutoTokenizer,
        DebertaV2Model,
        DebertaV2TokenizerFast,
        WhisperProcessor,
        pipeline,
    )

    tf_major = _parse_major_minor(transformers.__version__)[0]
    assert tf_major < 5, f"Expected transformers<5, got {transformers.__version__}"

    print(f"transformers: {transformers.__version__}, huggingface_hub: {huggingface_hub.__version__}")
    print(f"gradio: {gradio.__version__}, pydantic: {pydantic.__version__}")
    print(
        f"audio/nlp: librosa={librosa.__version__}, soundfile={soundfile.__version__}, "
        f"soxr={soxr.__version__}, "
        f"pyopenjtalk={getattr(pyopenjtalk, '__version__', 'installed')}, "
        f"pyworld={getattr(pyworld, '__version__', 'installed')}"
    )
    print(
        f"extra: tensorboard={tensorboard.__version__}, "
        f"matplotlib={matplotlib.__version__}"
    )

    _verify_subprocess()
    print("colab_setup verification passed!")


def run_host_setup(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    venv_path = Path(args.venv)
    venv_dir = venv_path if venv_path.is_absolute() else repo_root / venv_path
    venv_python = _venv_python_path(venv_dir)

    if not args.skip_install:
        uv_cmd = _find_uv_command()
        uv_env = {k: v for k, v in os.environ.items() if k != "UV_SYSTEM_PYTHON"}
        if args.recreate and venv_dir.exists():
            print(f"Removing existing virtualenv: {venv_dir}")
            shutil.rmtree(venv_dir)

        if not venv_python.exists():
            print(f"Creating Python {args.python} virtualenv at {venv_dir}...")
            subprocess.run(
                [*uv_cmd, "venv", str(venv_dir), "--python", args.python],
                cwd=repo_root,
                env=uv_env,
                check=True,
            )

        print("Installing Colab dependencies into .venv...")
        subprocess.run(
            [
                *uv_cmd,
                "pip",
                "install",
                "--python",
                str(venv_python),
                "--torch-backend",
                args.torch_backend,
                "-r",
                str(repo_root / "requirements-colab.txt"),
                "-r",
                str(repo_root / "requirements-colab-202611.txt"),
                "-c",
                str(repo_root / "constraints-colab-202611.txt"),
                "--no-progress",
            ],
            cwd=repo_root,
            env=uv_env,
            check=True,
        )
    elif not venv_python.exists():
        raise FileNotFoundError(
            f"Virtualenv Python not found at {venv_python}. Run without --skip-install first."
        )

    subprocess.run(
        [str(venv_python), str(Path(__file__).resolve()), "--verify-in-venv"],
        cwd=repo_root,
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision and verify an isolated Python 3.12 virtualenv for Google Colab."
    )
    parser.add_argument(
        "--venv",
        type=str,
        default=".venv",
        help="Path to the virtual environment directory (default: .venv).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="3.12",
        help="Python version to provision with uv venv (default: 3.12).",
    )
    parser.add_argument(
        "--torch-backend",
        type=str,
        default="cu124",
        help="uv --torch-backend value (default: cu124).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the virtual environment before installing.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip uv venv/install and only run in-venv verification.",
    )
    parser.add_argument(
        "--verify-in-venv",
        action="store_true",
        help="Internal flag to run compatibility checks inside the target virtualenv.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.verify_in_venv:
        verify_in_venv()
    else:
        run_host_setup(args)


if __name__ == "__main__":
    main()
