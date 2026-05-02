# Colab 2026.04 / latest handoff

## Background

The original Style-Bert-VITS2 Colab notebook worked on Google Colab runtime 2025.07, but failed on runtime 2026.04 / latest due to dependency and API changes.

A manual debugging session successfully reached:

- environment setup completed
- initial setup completed
- slicing and transcription completed
- training preprocessing completed
- training started

The goal now is to move the successful manual fixes into the repository in a maintainable form, suitable for VS Code + Codex maintenance.

## Confirmed environment

Observed in Google Colab latest / runtime 2026.04:

- Python 3.12
- torch 2.10.0+cu128
- torchaudio 2.10.0+cu128

## Issues found and working fixes

### 1. transformers 5.x incompatibility

#### Symptom

Style-Bert-VITS2 can fail in Japanese text processing / tokenizer-related paths when `transformers 5.x` is installed.

#### Working fix

Use a Colab runtime-specific constraints file:

```txt
numpy==1.26.4
transformers>=4.34.0,<5.0.0
```

Recommended install command:

```bash
uv pip install --system -r requirements-colab.txt -c constraints-colab-202604.txt --no-progress
```

#### Rationale

- The original notebook expects `numpy<2`.
- `transformers 5.x` introduced breaking behavior for this workflow.

## 2. Colab heredoc problem

In this Colab environment, notebook cells using heredoc-style shell syntax caused `SyntaxError: invalid syntax`.

Problematic examples:

```bash
python - <<'PY'
...
PY
```

```bash
cat <<'EOF'
...
EOF
```

Avoid heredoc in notebook cells. Prefer normal Python cells, `Path.write_text()`, or reusable `.py` files.

## 3. torchaudio.list_audio_backends removed

### Symptom

During audio slicing / Silero VAD path:

```text
module 'torchaudio' has no attribute 'list_audio_backends'
```

### Working fix

Provide a compatibility shim:

```python
if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return ["soundfile", "ffmpeg"]
    torchaudio.list_audio_backends = list_audio_backends
```

## 4. Whisper HF repo id was empty

### Symptom

During transcription:

```text
Loading HF Whisper model ()
HFValidationError: Repo id must use alphanumeric chars...
OSError: Can't load feature extractor for ''
```

### Working fix

Pass the Hugging Face repo id explicitly:

```bash
--use_hf_whisper --hf_repo_id "openai/whisper-large-v3"
```

A more robust notebook implementation should avoid shell quoting issues, especially when `initial_prompt` is empty or contains spaces / Japanese text.

## 5. torchaudio.AudioMetaData removed

### Symptom

During style feature generation:

```text
AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'
```

Observed through:

```text
pyannote.audio.core.io
```

### Working fix

Provide a compatibility shim:

```python
from typing import NamedTuple

if not hasattr(torchaudio, "AudioMetaData"):
    class AudioMetaData(NamedTuple):
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData
```

### Important note

Patching only the current notebook process is not enough.

`style_gen.py` is launched in a subprocess, so the patch must be available to subprocesses. Prefer one of these approaches:

- create/update `sitecustomize.py`, or
- run a reusable compatibility script before subprocesses start, or
- patch the relevant script in a controlled, documented way.

The recommended maintainable approach is to put this in `scripts/colab_202604_compat.py`.

## 6. PyTorch 2.6+ torch.load weights_only issue

### Symptom

During style feature generation:

```text
_pickle.UnpicklingError: Weights only load failed.
In PyTorch 2.6, we changed the default value of the weights_only argument in torch.load from False to True.
Unsupported global: GLOBAL torch.torch_version.TorchVersion
```

Observed during:

```python
Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
```

The actual failing path went through:

```text
lightning_fabric.utilities.cloud_io
```

### Working fix

Patch `lightning_fabric.utilities.cloud_io` so that its `torch.load` path uses `weights_only=False`.

The working manual patch was effectively:

```python
text = text.replace("weights_only=weights_only", "weights_only=False")
```

### Security note

`weights_only=False` should only be used for trusted checkpoints because loading arbitrary pickled objects can execute code.

In this workflow, it is used for the known pyannote model checkpoint required by Style-Bert-VITS2 style feature generation:

```text
pyannote/wespeaker-voxceleb-resnet34-LM
```

## Recommended target structure

```text
Style-Bert-VITS2/
├── AGENTS.md
├── constraints-colab-202604.txt
├── scripts/
│   └── colab_202604_compat.py
└── docs/
    └── colab_202604_handoff.md
```

## Recommended `constraints-colab-202604.txt`

```txt
numpy==1.26.4
transformers>=4.34.0,<5.0.0
```

## Recommended `scripts/colab_202604_compat.py` responsibilities

The compatibility script should:

1. Patch current-process torchaudio compatibility:
   - `list_audio_backends`
   - `AudioMetaData`
2. Create or update `sitecustomize.py` so subprocesses receive the torchaudio compatibility shims.
3. Patch `lightning_fabric.utilities.cloud_io` to use `weights_only=False` for the trusted checkpoint loading path.
4. Print concise verification output:
   - torch version
   - torchaudio version
   - whether `list_audio_backends` exists
   - whether `AudioMetaData` exists
   - whether cloud_io has been patched

## Acceptance criteria

A cleaned Colab notebook should:

1. Install dependencies using `requirements-colab.txt` plus `constraints-colab-202604.txt`.
2. Run `scripts/colab_202604_compat.py` once after install.
3. Avoid temporary debug cells.
4. Avoid heredoc syntax.
5. Use explicit HF Whisper repo id.
6. Complete preprocessing with:

```text
Success: All preprocess finished!
```

7. Start training successfully.

## Manual success state already reached

The manual debugging session reached this state:

```text
Success: All preprocess finished!
```

and training started successfully afterward.

## Suggested first Codex task

Ask Codex to start with a plan, not edits:

```text
This repository is a personal fork of Style-Bert-VITS2.
Please read AGENTS.md, docs/colab_202604_handoff.md, colab.ipynb, and requirements-colab.txt.
Do not edit yet. First propose a minimal implementation plan to move the successful Colab 2026.04 manual fixes into maintainable repository files.
List the files you would change and explain why.
```

After reviewing the plan, continue with:

```text
Proceed with the plan using small, reviewable changes. Keep upstream Style-Bert-VITS2 source changes minimal. Prefer scripts/colab_202604_compat.py and constraints-colab-202604.txt over scattered notebook patches.
```
