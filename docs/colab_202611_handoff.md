# Colab 2026.11 / latest handoff

## Background

In April 2026, [docs/colab_202604_handoff.md](colab_202604_handoff.md) kept `colab.ipynb` running on Colab 2026.04 (Python 3.12) by installing into Colab's system Python (`uv pip install --system`) and restarting the runtime with `exit()`.

By November 2026, the default Google Colab runtime moved to **Python 3.13**, **PyTorch 2.14.1 / torchaudio 2.11.0**, and **NumPy 2.4.6**. Installing directly into Colab's system Python breaks immediately because:

- `numpy==1.26.4` (and `numpy<2` in `requirements-colab.txt`) has no Python 3.13 support.
- `librosa==0.9.2` and legacy audio/build packages depend on stdlib modules removed in Python 3.13 (PEP 594: `aifc`, `sunau`, `audioop`) and older `setuptools` (`pkg_resources`, removed in `setuptools 82.0.0`).
- Unbounded dependencies (`pyannote.audio>=3.1.0`, `gradio>=4.32`, `transformers`) pull incompatible major versions (`pyannote.audio 4.x` with `torchcodec`, `gradio 5.x/6.x` breaking `gr.Blocks(theme=...)`, `transformers 5.x`, `huggingface-hub 1.x`, and `pydantic>=2.10` triggering schema errors in Gradio 4.44).

## Confirmed host environment (2026-11-21)

Observed on Google Colab GPU runtime:

- Python `3.13.16` (`GCC 13.3.0`) on GPU runtime (and `3.13.18` on CPU runtime)
- `torch`: `2.14.1+cu130` (GPU) / `2.14.1+cpu` (CPU)
- `torchaudio`: `2.11.0+cu130` (GPU) / `2.11.0+cpu` (CPU)
- `numpy`: `2.4.6`
- GPU / driver tested: `Tesla T4`, NVIDIA Driver `580.82.07`

> **Important:** Make sure a **GPU runtime** (`T4`, `L4`, `A100`, etc.) is selected in Colab (**Runtime > Change runtime type**) before running training or GPU-accelerated preprocessing.

## Architecture: Isolated `uv` Python 3.12 Virtualenv (Option A)

Instead of fighting Colab's global Python 3.13 upgrades, `colab.ipynb` now provisions an isolated `.venv` inside `/content/Style-Bert-VITS2/.venv` using `uv`:

1. `python3 -m pip install -q uv`
2. `python3 scripts/colab_setup.py`
   - Creates `.venv` with Python `3.12` via `uv venv .venv --python 3.12`
   - Installs `requirements-colab.txt`, `requirements-colab-202611.txt`, and `constraints-colab-202611.txt` with `--torch-backend cu124` (`torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`)
   - Runs `.venv/bin/python scripts/colab_setup.py --verify-in-venv` to verify versions, imports, CUDA availability, and subprocess startup
3. All repository scripts (`initialize.py`, `slice.py`, `transcribe.py`, `preprocess_all.py`, `train_ms_jp_extra.py`, `train_ms.py`, `app.py`, `convert_onnx.py`) run explicitly via `.venv/bin/python`.
4. Notebook Python cells that run in Colab's host kernel (`configs/paths.yml` and `config.yml` generation) use only Python stdlib + `pyyaml` and never import `gradio_tabs.train` or `style_bert_vits2`.
5. No `exit()` crash-restart is needed anymore.

## Pinned constraints and extra packages

### `requirements-colab-202611.txt`

Packages previously supplied implicitly by Colab's global Python or omitted by `uv venv`:

- `matplotlib`: plotting spectrograms in training utils and clustering UI
- `sentencepiece`: tokenizer dependency for English DeBERTa v3 (`spm.model`) when `use_jp_extra=False`
- `setuptools<81`: retains `pkg_resources` without `setuptools 81+` deprecation warnings or `82.0.0` removal
- `soxr`: required by `transformers/audio_utils.py` (`WhisperProcessor` during HF Whisper transcription) because `librosa==0.9.2` does not pull `soxr`
- `tensorboard`: training logs and `SummaryWriter`
- `wheel`: build helper

### `constraints-colab-202611.txt`

- `numpy==1.26.4`: required by `librosa==0.9.2` and Style-Bert-VITS2
- `torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`: coherent CUDA 12.4 wheel stack (`--torch-backend cu124`); required because `transformers==4.57.6` enforces `torch >= 2.6` (`check_torch_load_is_safe()` / CVE-2025-32434) when loading `.bin` checkpoints such as `bert/deberta-v2-large-japanese-char-wwm/pytorch_model.bin`, while `torchaudio==2.6.0` still retains native `list_audio_backends` / `AudioMetaData`
- `transformers>=4.34.0,<5.0.0` & `huggingface-hub<1.0.0`: avoids `transformers 5.x` and `huggingface-hub 1.x` breaking changes
- `pyannote.audio>=3.1.0,<4.0.0`, `lightning<2.6.0`, `pytorch-lightning<2.6.0`: avoids `pyannote.audio 4.x` (`torchcodec` / `token`) breaks and `lightning 2.6+` forcing `weights_only=True` when loading `pyannote/wespeaker-voxceleb-resnet34-LM`
- `gradio>=4.32,<5.0.0`, `pydantic>=2.0,<2.10`, `fastapi<0.116`: avoids Gradio 5/6 `Blocks(theme=...)` breaks and the Gradio 4.44 + Pydantic 2.10+ `bool is not iterable` schema bug
- `setuptools<81`: preserves `pkg_resources` without `setuptools 81+` deprecation warnings

### `.venv` startup hook (`MPLBACKEND` & `torch.load` shim)

Colab exports `MPLBACKEND=module://matplotlib_inline.backend_inline` and `UV_SYSTEM_PYTHON=1` in the notebook kernel environment. Because `matplotlib_inline` is not installed inside `.venv`, any `.venv/bin/python` process importing `matplotlib` crashes with `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend`. Also, PyTorch 2.6 changed `torch.load(..., weights_only=...)` to default to `True`, which breaks loading `pyannote/wespeaker-voxceleb-resnet34-LM` through `lightning_fabric/utilities/cloud_io.py`.

To handle both cleanly without touching upstream code:

- `scripts/colab_setup.py` strips `UV_SYSTEM_PYTHON` when invoking `uv` so `uv venv` does not emit the `--system` warning.
- `scripts/colab_setup.py --verify-in-venv` installs `_style_bert_vits2_colab_compat.py` + `style_bert_vits2_colab_compat.pth` inside `.venv`'s `site-packages`, resetting `MPLBACKEND` from `module://matplotlib_inline*` to `Agg` at startup for every `.venv/bin/python` process and subprocess.
- On `torch >= 2.6`, the `.pth` startup hook also wraps `torch.load` so calls originating from `lightning_fabric/utilities/cloud_io.py` pass `weights_only=False` for the trusted `pyannote/wespeaker-voxceleb-resnet34-LM` checkpoint (and provides conditional `torchaudio` shims if `torchaudio` is ever overridden).

## Colab rerun / update notes

To pull updates in an existing Colab session without recreating `.venv` from scratch:

```bash
cd /content/Style-Bert-VITS2
git pull
python3 scripts/colab_setup.py
```

To wipe and rebuild `.venv` cleanly:

```bash
python3 scripts/colab_setup.py --recreate
```

## Manual smoke test checklist (Google Colab GPU runtime)

Before running, verify **Runtime > Change runtime type** has a GPU (`T4`, `L4`, or `A100`) selected (do not run training on a CPU runtime):

1. **Cell 0 (Environment setup)**: completes without `exit()` crash dialog; `colab_setup verification passed!` and `cuda_available=True` appear.
2. **Drive mount & Cell 1 (Initial setup)**: `configs/paths.yml` points to Drive paths.
3. **Cell 2.1 (Slicing & transcription)**: `.venv/bin/python slice.py` and `.venv/bin/python transcribe.py` finish with `esd.list` generated.
4. **Cell 3 (Training preprocessing)**: `.venv/bin/python preprocess_all.py ...` finishes with `Success: All preprocess finished!`.
5. **Cell 4 (Training)**: `config.yml` is written and `.venv/bin/python train_ms_jp_extra.py ...` starts training without import/runtime errors.
6. **Optional**: `.venv/bin/python app.py --share` launches Gradio WebUI without Pydantic/Blocks errors.

## Verified repository implementation

Verified end-to-end in Google Colab on `2026-11-21` at commit `8b7a7e7` (`Tesla T4`, NVIDIA Driver `580.82.07`):

- **Colab host runtime**: `Python 3.13.16`, `torch 2.14.1+cu130`, `torchaudio 2.11.0+cu130`, `numpy 2.4.6`
- **Isolated `.venv` stack**: `Python 3.12.3`, `numpy 1.26.4`, `torch==2.6.0` / `torchaudio==2.6.0` / `torchvision==0.21.0` (`--torch-backend cu124`), `transformers 4.57.6`, `pyannote-audio 3.4.0`, `lightning 2.5.6`, `gradio 4.44.1`, `pydantic 2.9.2`, `librosa 0.9.2`, `soxr`
- **Steps verified**:
  - **Cell 0**: `scripts/colab_setup.py` + `initialize.py --skip_default_models` completed cleanly without `exit()` runtime restart
  - **Cell 1**: Google Drive mounted and `configs/paths.yml` configured
  - **Cell 2.1**: `.venv/bin/python slice.py` (`Slice done! Total time: 0.39 min, 8 files.`) and `.venv/bin/python transcribe.py` (`openai/whisper-large-v3` on `cuda`) completed
  - **Cell 3**: `.venv/bin/python preprocess_all.py` completed `initialize`, `resample.py`, `preprocess_text.py`, `bert_gen.py` (`8 bert.pt files`), and `style_gen.py` (`8 npy files`), ending with `Success: All preprocess finished!`
  - **Cell 4**: `config.yml` written and `.venv/bin/python train_ms_jp_extra.py` loaded pretrained checkpoints and reached `Start training.`
- **Non-fatal warnings observed**:
  - `librosa/util/files.py` `UserWarning: pkg_resources is deprecated` (`setuptools<81`)
  - `transformers` Whisper pipeline `torch_dtype` / `chunk_length_s` notices during `transcribe.py`
  - `pyannote` TF32 `ReproducibilityWarning` during `style_gen.py`
  - Expected initial `Missing key` warnings (`enc_p.style_proj.*`, `emb_g.weight`) when loading `G_0.safetensors`
- **Not run in smoke test**: `app.py --share` and `convert_onnx.py` (covered by notebook/import validation only).
