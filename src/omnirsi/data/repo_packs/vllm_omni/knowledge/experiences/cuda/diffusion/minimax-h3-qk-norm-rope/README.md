# MiniMax-H3 guarded Q/K RMSNorm and packed RoPE fusion

A merged CUDA fusion avoids normalized-Q/K and rotary intermediates for a specific packed BF16 geometry; reported denoise benefit is small and limited to a B300 workload.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `upstream_reported`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `cuda`
- Scenario: `diffusion.video_generation`
- Hardware: 4× NVIDIA B300 SXM6

### Software and workload

- Source PR #5990 head 0d0d42fe7b4a07581cd45ac67bbe1a2f6194fcdd; merged as 596c16a550aa134faf7f3dcfa0f8adf513ccd9ce
- CUDA/Triton BF16 fast path; head_dim=128, rotary_dim=96; TP1, Ulysses4, Ring1, TRTLLM_ATTN
- MiniMax-H3
- Packed [tokens, heads, head_dim] with compatible non-interleaved [cos|sin] frequencies
- 1344×768, 243 frames, 50 configured steps, seed 0
- One warmup followed by one measured request per revision

## Findings and evidence

- **upstream_reported**: Steady denoise latency fell from 109687 ms to 106205 ms, a reported 3.18% reduction. This is denoise timing, not client E2E or a statistical performance guarantee.
- **upstream_reported**: Three CUDA unit cases tested sequence lengths 1, 257 and 1024. Author comment reports max absolute operator error 0.0625, mean absolute error 0.00072–0.00077; qualitative near-identical E2E video is reported without objective media metrics.
- **source_inspected**: The patch dispatches fusion only with HAS_TRITON, CUDA platform/tensors, BF16 and supported geometry; other valid inputs use the eager reference. It preserves a BF16 normalization materialization before RoPE but still tests tolerance rather than byte identity.

## How to use this experience

- Treat the optimization as intended math-preserving with measured numerical tolerance, not universal bit-exactness.
- Verify selected geometry, RoPE convention and actual fast-path execution before attributing a gain.
- Reuse the existing guarded shared operator rather than copying a CUDA kernel into a platform-neutral model.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Run the target revision's tests/diffusion/layers/test_fused_qk_norm_rope.py with its registered GPU test markers.
- Compare native and fused operator/block/trajectory outputs including empty, ragged and strided Q/K cases.
- Run at least three unprofiled warm A/B requests on the target H200/B200 or other GPU; record client E2E, denoise, memory and same-seed video/audio metrics.
- Exercise unsupported dtype/geometry/platform fallbacks separately.

## Limits

- The reported benchmark is B300, not H200/B200, ROCm or Ascend evidence.
- Only one measured request per arm was reported.
- No objective full-model quality metric appears in the reviewed report; imported status does not mean locally verified.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5990) — `0d0d42fe7b4a07581cd45ac67bbe1a2f6194fcdd` (merged); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5990#issuecomment-5251819347) — `0d0d42fe7b4a07581cd45ac67bbe1a2f6194fcdd` (merged); accessed 2026-10-09.
