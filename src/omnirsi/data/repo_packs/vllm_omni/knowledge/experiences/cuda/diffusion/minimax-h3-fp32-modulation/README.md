# MiniMax-H3 modulation fusion changes rounding and requires platform guards

Merged FP32-accumulating modulation fusion reduces launch count but removes intermediate BF16 rounding; its numerical drift and historical non-CUDA dispatch finding must remain visible.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `upstream_reported`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `cuda`
- Scenario: `diffusion.video_generation`
- Hardware: CUDA GPU SKU not stated in PR benchmark; do not substitute H200/B200

### Software and workload

- PR #6281 head 7d74126f0723d355d82b47e98f820394320e1a4a; merged 14f2c16fb28e1e0c8615372675f2757466260ad8
- Triton modulation path, eager, USP4
- MiniMax-H3 Ref2VA
- 5-second output, 49 denoise iterations; both configurations warmed up
- Report hardware/driver/runtime identity before reproduction

## Findings and evidence

- **upstream_reported**: Engine E2E fell from 136961 ms to 134707 ms (1.65%); a single DiT profile reports 1969→1234 CUDA kernels (37.33% fewer). Profiled wall time is diagnostic and not interchangeable with unprofiled engine E2E.
- **upstream_reported**: FP32 accumulation removes intermediate BF16 rounding. Reported output PSNR is 39.55 dB and SSIM 0.9750; the author explicitly says outputs are not bit-exact.
- **review_finding**: Review identified x.is_cpu-only gating, which could launch Triton on NPU/XPU/MUSA, and noted row-major contiguity assumptions. The fetched PR patch still contains that historical guard; verify subsequent target-source fixes rather than inferring safe non-CUDA support.

## How to use this experience

- Classify this as a numerically changed implementation and require an explicit quality tolerance.
- Preserve original BF16 materialization boundaries when designing an exact fusion; otherwise report the changed rounding.
- Gate fast paths by platform, device, dtype, shape and layout; retain backend-native norm/eager fallbacks.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Verify current source platform guards and contiguity contract before remote use.
- Compare operator outputs, trajectory drift and decoded video/audio against a same-seed reference and baseline self-run variance.
- Repeat unprofiled E2E measurements with exact device/runtime identity; retain profile evidence separately.
- Run fallback checks on each advertised backend; CPU tests alone do not validate CUDA/NPU numerical paths.

## Limits

- Hardware SKU, repeat count and broader prompt coverage are not stated in the PR body.
- A merged PR is not proof of quality equivalence or safe support on every backend.
- Historical review findings may have changed later; source revision must be rechecked.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/6281) — `7d74126f0723d355d82b47e98f820394320e1a4a` (merged); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/6281#discussion_r3809815474) — `7d74126f0723d355d82b47e98f820394320e1a4a` (merged); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/6281#discussion_r3809815564) — `7d74126f0723d355d82b47e98f820394320e1a4a` (merged); accessed 2026-10-09.
