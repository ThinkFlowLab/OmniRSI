# Local B300 × 4 / LTX-2.5 campaign status (2026-10-10)

The user changed the campaign to **720p class / 20% lower E2E latency** and allowed shorter diagnostic steps. This is preparation evidence, not a measured performance or accuracy result. The current verdict is **INCONCLUSIVE** because the exact model revision cannot be downloaded with the available HF credential.

## Prepared protocol

- Four local B300 devices: physical IDs 0–3, 275040 MiB each, driver 610.43.02, NV18 links, NUMA 0 / CPUs 0–239. No compute processes were active during inspection.
- Baseline and candidate source: `2f289bb179d85671c116c10208b5c84b5c53f18f`. Latest upstream main was fetched; development uses `codex-` branches and separate pinned worktrees.
- Model: `Lightricks/LTX-2.5-Diffusers`, revision `a6de4b5354f078db24d9cf4778c14846788aea3d`.
- Default output: 1280×704, the closest height to 720 divisible by 64, required by the two-stage pipeline. `--width` / `--height` are frozen into each protocol; changing them requires a new reference and campaign.
- Formal benchmark: full 8+3 schedule, 121 frames, 24 FPS, BF16, Native DiffVAE, 48kHz stereo, three fixed prompt/seed pairs, six independent full-shape warmups per fresh server.
- Diagnostic: default 2+1 schedule, independently adjustable per stage. Its actual sigma schedules and protocol hash are recorded, it has no performance metric, and reference/search/quality gates reject diagnostic manifests.
- Accuracy policy remains per-frame SSIM ≥0.99 and PSNR ≥35 dB, stereo audio NRMSE ≤0.01, complete A/V decoding, dimensions, frames, FPS and timing checks. None of these thresholds was relaxed.
- Acceptance remains five alternating AB/BA groups against the freshly remeasured quality-passing four-worker winner, requiring `L1/L0 ≤ 0.80`.

## Completed checks

- Prepared an independent uv-managed environment at `/home/zjy/code/hsliu/venvs/codex-ltx25-b300`; verified imports resolve to the pinned baseline checkout.
- Environment: Torch 2.13.0+cu130, vLLM 0.31.0, Diffusers 0.40.0, Transformers 5.14.1, kernels 0.16.1, NATTEN 0.21.7, flashinfer-python 0.7.0.post1, cuDNN distribution 9.20.0.48, NCCL distribution 2.29.7. Distribution versions do not certify loaded-runtime or model compatibility.
- `python -m unittest discover -s tests -v`: **78 passed, no skips**, including real FFmpeg A/V validation and Linux owned-process cleanup.
- Pre-commit debug-statement, EOF, line-ending and whitespace checks passed; `git diff --check` passed.
- Diagnostic dry-run confirms 1280×704 / 2+1 schedule and fixed model/decoder/attention settings without allocating GPUs.

## Actual blocker and remaining work

The exact revision's `model_index.json` request returned **HTTP 401 / GatedRepoError**. The inherited `HF_TOKEN` also failed account validation; no saved local token or downloaded snapshot was found. An authorized account must authenticate locally, or the operator must provide the exact revision's complete model cache. Credentials must not be posted in PR comments or committed.

Canonical generation, repeatability, the nine-cell parallel search, GPU profiling, performance code iterations and final A/V accuracy validation have **not run**. No baseline/candidate latency or improvement is available, and no performance change has been made before profiling.

After model access is restored, freeze guards and asset hashes, generate/repeat the canonical reference, qualify all nine parallel cells, freeze the winner, profile with short steps, then iterate by pipeline stage → distributed communication/layout → operators → full E2E validation. Keep every accepted and rejected hypothesis with its raw evidence. [The executable test plan](ltx25-b300-test-plan.md) contains the commands and unchanged quality gates.

Local evidence lives outside the source tree at `/home/zjy/code/david/tmp/ltx25-b300-rsi-20261010/artifacts`: `environment.json`, `model-access.json`, `campaign-status.json`, `rsi-events.jsonl`, `cpu-tests.log`, `pip-freeze.txt`, device/topology/storage snapshots and `profile-dry-run.json`. These local paths are not downloadable GitHub artifacts.
