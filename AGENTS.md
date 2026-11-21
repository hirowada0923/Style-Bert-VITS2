# AGENTS.md

## Project purpose

This repository is a personal fork of Style-Bert-VITS2.

The main maintenance goal is to keep the Google Colab notebook working on newer Google Colab runtimes (currently Colab 2026.11 / latest with Python 3.13 host, and historically 2026.04).

## Development environment

Primary maintenance environment:

- VS Code
- OpenAI Codex extension / Codex chat
- GitHub repository: hirowada0923/Style-Bert-VITS2
- Manual smoke testing in Google Colab

## Maintenance policy

- Keep upstream Style-Bert-VITS2 source changes minimal.
- Prefer Colab-specific compatibility scripts over scattered notebook patches.
- Do not leave temporary debug cells in `colab.ipynb`.
- Avoid heredoc syntax in Colab cells. It caused `SyntaxError: invalid syntax` in this environment.
- Keep Colab runtime-specific dependency pins in separate constraint files.
- Document runtime-specific workarounds under `docs/`.
- Before editing, explain the intended change and affected files.
- Prefer small, reviewable commits.

## Current Colab 2026.11 / latest setup (isolated `.venv`)

Colab 2026.11 / latest runs Python 3.13 (`3.13.16` GPU / `3.13.18` CPU), `torch 2.14.1`, `torchaudio 2.11.0`, and `numpy 2.4.6` on the host runtime (tested on `Tesla T4` / driver `580.82.07`). Because `numpy==1.26.4` and legacy audio/build packages (`librosa==0.9.2`, `pyopenjtalk-dict`, etc.) do not support Python 3.13 or NumPy 2.x, `colab.ipynb` isolates Style-Bert-VITS2 inside `.venv` (`Python 3.12`) managed by `uv`:

- `requirements-colab-202611.txt`: extra packages needed in `.venv` (`matplotlib`, `sentencepiece`, `setuptools<81`, `soxr`, `tensorboard`, `wheel`).
- `constraints-colab-202611.txt`: coherent Python 3.12 / CUDA 12.4 constraints (`numpy==1.26.4`, `torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`, `transformers>=4.34.0,<5.0.0`, `huggingface-hub<1.0.0`, `pyannote.audio>=3.1.0,<4.0.0`, `lightning<2.6.0`, `pytorch-lightning<2.6.0`, `gradio>=4.32,<5.0.0`, `pydantic>=2.0,<2.10`, `fastapi<0.116`, `setuptools<81`).
- `scripts/colab_setup.py`: stdlib-only host script that strips `UV_SYSTEM_PYTHON`, provisions `.venv` via `uv` (`--torch-backend cu124`), installs a `.venv` `.pth` startup hook sanitizing Colab's `MPLBACKEND=module://matplotlib_inline.backend_inline` to `Agg` (and wrapping `torch.load` for `lightning_fabric/utilities/cloud_io.py` on PyTorch 2.6+), and verifies all imports (`check_torch_load_is_safe()` included) and subprocess behavior inside `.venv/bin/python --verify-in-venv`.
- All `colab.ipynb` script invocations (`initialize.py`, `slice.py`, `transcribe.py`, `preprocess_all.py`, `train_ms_jp_extra.py`, `train_ms.py`, `app.py`, `convert_onnx.py`) must use `.venv/bin/python`.
- Host Python notebook cells must NOT import `gradio_tabs` or `style_bert_vits2`; use pure stdlib + `pyyaml` for `configs/paths.yml` and `config.yml`.
- No `exit()` forced session restart is needed.

## Historical Colab 2026.04 fixes (preserved for reference)

### Dependency constraints

Use a separate constraints file for Colab 2026.04, for example `constraints-colab-202604.txt`.

Required constraints:

```txt
numpy==1.26.4
transformers>=4.34.0,<5.0.0
```

Reason:

- Style-Bert-VITS2 currently does not work reliably with `transformers 5.x`.
- The original notebook expects `numpy<2`.

### torchaudio compatibility

Colab 2026.04 uses `torchaudio 2.10.0+cu128`.

Known issues:

- `torchaudio.list_audio_backends` is missing.
- `torchaudio.AudioMetaData` is missing.

These must be patched for old dependencies such as Silero VAD and pyannote.audio.

### PyTorch checkpoint loading compatibility

Colab 2026.04 uses `torch 2.10.0+cu128`.

PyTorch 2.6+ changed checkpoint loading behavior around `torch.load(..., weights_only=...)`.

Known failing point:

- `pyannote/wespeaker-voxceleb-resnet34-LM`
- loaded through `pyannote.audio`
- via `lightning_fabric.utilities.cloud_io`

The working workaround is to patch `lightning_fabric.utilities.cloud_io` so that `weights_only=False` is passed for the trusted pyannote checkpoint path.

Security note:

- `weights_only=False` can execute arbitrary code when loading untrusted checkpoints.
- Use it only for trusted checkpoints. In this project it is used for the known pyannote checkpoint required by Style-Bert-VITS2 style feature generation.

### Whisper transcription

When using HF Whisper, pass the repo id explicitly:

```bash
--use_hf_whisper --hf_repo_id "openai/whisper-large-v3"
```

Otherwise, an empty repo id can cause:

```text
HFValidationError: Repo id must use alphanumeric chars...
OSError: Can't load feature extractor for ''
```

## Recommended implementation structure

Prefer this structure for the current Colab compatibility work (while preserving historical `202604` files):

```text
Style-Bert-VITS2/
├── AGENTS.md
├── constraints-colab-202604.txt        # historical 2026.04
├── constraints-colab-202611.txt        # current 2026.11+ (.venv Python 3.12)
├── requirements-colab-202611.txt       # extra packages for isolated .venv
├── scripts/
│   ├── colab_202604_compat.py          # historical 2026.04
│   └── colab_setup.py                  # current .venv setup & verifier
└── docs/
    ├── colab_202604_handoff.md         # historical 2026.04
    └── colab_202611_handoff.md         # current 2026.11+
```

## Manual smoke test

Run in Google Colab (with a GPU runtime such as T4, L4, or A100 selected):

1. Environment setup (`colab_setup.py` + `initialize.py --skip_default_models`)
2. Initial setup (`drive.mount` + `configs/paths.yml`)
3. Audio slicing and transcription (`.venv/bin/python slice.py` + `transcribe.py`)
4. Training preprocessing (`.venv/bin/python preprocess_all.py`)
5. Start training (`.venv/bin/python train_ms_jp_extra.py` or `train_ms.py`)

Expected result:

- `colab_setup verification passed!` with `cuda_available=True`
- preprocessing ends with `Success: All preprocess finished!`
- training starts without immediate import/runtime errors

## Codex working style

When working in this repository:

1. Read `AGENTS.md` and `docs/colab_202611_handoff.md` (plus `docs/colab_202604_handoff.md` for historical context) first.
2. Inspect `colab.ipynb`, `requirements-colab.txt`, and any Colab-specific files before editing.
3. First propose a small implementation plan and list affected files.
4. Keep changes narrow and reversible.
5. Do not rewrite large parts of the notebook unless explicitly requested.
6. Prefer adding or updating reusable scripts and docs rather than adding temporary notebook cells.
