# MiniMax-H3 online CUDA MXFP8 is an opt-in approximate deployment lane

An open Blackwell MXFP8 PR quantizes selected DiT projections after loading/fixed-adapter fusion and preserves sensitive components; small/shared-host observations do not establish stable speed or quality equivalence.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `open_pr`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `cuda`
- Scenario: `diffusion.video_generation`
- Hardware: NVIDIA device reported as NVIDIA Graphics Device, compute capability 12.0, approximately 71.12 GiB/device; exact SKU unspecified

### Software and workload

- vLLM 0.29.0 wheel overlay, PyTorch 2.13.0+cu132, Python 3.12.13, driver 580.95.05
- PR head 0c248c848b5dfadef7066d54cfeae1d78c898757 based on c7cd37b61e44fb51c61212ac9a317f2be26fe638
- Public PyTorch block-scaled scaled_mm; CUDA/partition capability guards
- Native FastH3 checkpoint with fixed vsa-datafree adapter
- TP2/USP2/text-encoder TP4, eager, VAE tiling, 448×256, 4.4 seconds, seed 1101, four denoise steps
- Main DiT QKV/output/MLP quantized; conditioning/refiner/AdaLN/gates/output projections retain original precision
- Runtime adapter switching after conversion is unqualified

## Findings and evidence

- **upstream_reported**: 12 new CUDA tests and 30 regression tests passed (3 skipped). Three full-model requests produced decodable 107-frame/24-FPS MP4s with stereo 32-kHz audio; TP2 uses reduction-compatible fallback rather than TP1 fused activation path.
- **upstream_reported**: Two warm shared-host client observations: branch 3018/2968 ms versus BF16/FastVideo 3419/3419 ms. Source explicitly disclaims a stable performance claim.
- **quality_limit**: Single-prompt decoded-media drift: RGB PSNR 22.92 dB and audio MSE 0.00080053. These are not perceptual acceptance, byte equality or quality-equivalence evidence.
- **counterexample**: Checkpoint-scale TP1/USP4 resident attempt exceeded ~71.12 GiB/device during fixed-adapter fusion. Lower post-quantization residency does not qualify the loading/fusion peak.
- **upstream_reported**: Separate shared-host microbenchmark M=2048/K=14336/N=5376: median 0.8818→0.8491 ms after five warmups/30 CUDA-event samples, final GEMM output byte-identical. This is a kernel observation only.

## How to use this experience

- Keep opt-in quantization and explicit component routing; validate actual runtime prefixes and loading/fusion order.
- Record raw hardware identity rather than mapping SM12.0 to an assumed product name.
- Maintain separate TP1 fusion and TP>1 reduction paths, cold loading peaks and fixed-adapter/runtime-adapter policies.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Run pinned-source tests/diffusion/quantization/test_cuda_mxfp8.py with registered GPU markers and the linear microbenchmark.
- On the intended H200/B200 check hardware/API capability; reject or use a known supported alternative instead of inferring support from Blackwell branding.
- Measure exact-source cold peak and at least three isolated warm E2E samples.
- Run broad task/prompt/resolution/media quality evaluation and trajectory metrics against a same-base control before acceptance.

## Limits

- Open/draft PR; broad quality/performance coverage and human review remain pending.
- Reported SM12.0 hardware is not evidence for H200 or B200.
- Shared-host observations, one prompt and a microbenchmark do not support production speed/quality claims.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/7988) — `0c248c848b5dfadef7066d54cfeae1d78c898757` (open_draft); accessed 2026-10-09.
