# MiniMax-H3 Ref2VA TeaCache needs task-specific calibration and step defaults

An open Ref2VA TeaCache PR calibrates a conservative late-step budget; quality and speed depend on task/step policy, and some cases produce zero hits and no gain.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `open_pr`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `cuda`
- Scenario: `diffusion.video_generation`
- Hardware: 1× NVIDIA B300

### Software and workload

- Open PR head f86bd6ad7bfc7e6eb04df0aaacd35ec68643d6a2
- BF16, CUDNN_ATTN, task-specific TeaCache coefficients
- MiniMax-H3 Ref2VA
- 448×256, 107 output frames, 50 configured steps; fixed-seed comparison
- Calibration uses 15 full-compute Ref2VA cases: 6 image, 6 image+audio, 3 video+audio
- 35 initial full-compute denoise calls, then at most five cache hits; other step counts remain uncached

## Findings and evidence

- **upstream_reported**: Video+audio example: 5/49 cache hits; latency 244190→221240 ms, video SSIM 0.9795/PSNR 41.60 dB and audio PSNR 168.8 dB. This demonstrates a speed/quality tradeoff, not exact equivalence.
- **counterexample**: Image-only example: 0/49 hits; 53340→53260 ms, SSIM 1.000 and sample-identical audio. No meaningful speed gain was observed.
- **upstream_reported**: Both video+audio runs peak at 140972 MiB; caching reduces compute and is not a weight-memory solution.
- **review_finding**: An omitted num_inference_steps initially resolved to 0 in cache refresh while the pipeline used 50, bypassing calibrated warmup. Author reports a fix at 0d9594f21 preserved in f86bd6ad7, with shared effective defaults and regressions.

## How to use this experience

- Treat residual reuse as approximate; keep task-specific calibrated policies and explicit step fallback.
- Maintain separate hooks/request state for FL2VA and Ref2VA; do not copy residuals across tasks or concurrent requests.
- Record actual cache hits and effective step schedule, not just an enabled flag; validate custom thresholds across video and audio.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Use the exact PR head or verify target-source task support; the PR was open/draft at import.
- Test explicit 50 steps, omitted steps, non-50 steps, task alternation and error/abort reset behavior.
- Same-seed A/B across multiple prompt/media types; measure video, temporal and audio metrics with an explicit acceptance threshold.
- Run at least three unprofiled warm repetitions and measure device/host memory separately from compute reduction.

## Limits

- Open/draft PR; no claim of main support or OmniRSI reproduction.
- Reported quality results are a small same-seed sample set; full perception/temporal quality acceptance remains user-specific.
- TeaCache and Cache-DiT are separate mutually exclusive backend choices in the reviewed branch.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5952) — `f86bd6ad7bfc7e6eb04df0aaacd35ec68643d6a2` (open_draft); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5952#discussion_r3756706344) — `f86bd6ad7bfc7e6eb04df0aaacd35ec68643d6a2` (open_draft); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5952#issuecomment-5288600990) — `f86bd6ad7bfc7e6eb04df0aaacd35ec68643d6a2` (open_draft); accessed 2026-10-09.
