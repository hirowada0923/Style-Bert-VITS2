# AGENTS.md

## Project purpose

This repository is a personal fork of Style-Bert-VITS2.

The main maintenance goal is to keep the Google Colab notebook working on newer Google Colab runtimes, especially Colab runtime 2026.04 / latest.

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

## Known Colab 2026.04 / latest fixes

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

Prefer this structure for the Colab 2026.04 compatibility work:

```text
Style-Bert-VITS2/
├── AGENTS.md
├── constraints-colab-202604.txt
├── scripts/
│   └── colab_202604_compat.py
└── docs/
    └── colab_202604_handoff.md
```

## Manual smoke test

Run in Google Colab:

1. Environment setup
2. Initial setup
3. Audio slicing and transcription
4. Training preprocessing
5. Start training

Expected result:

- preprocessing ends with `Success: All preprocess finished!`
- training starts without immediate import/runtime errors

## Codex working style

When working in this repository:

1. Read `AGENTS.md` and `docs/colab_202604_handoff.md` first.
2. Inspect `colab.ipynb`, `requirements-colab.txt`, and any Colab-specific files before editing.
3. First propose a small implementation plan and list affected files.
4. Keep changes narrow and reversible.
5. Do not rewrite large parts of the notebook unless explicitly requested.
6. Prefer adding or updating reusable scripts and docs rather than adding temporary notebook cells.
