# MiniMax-H3 SP head buckets reduce exposed All-to-All with quality still pending

An open Ascend head-bucket pipeline overlaps projection/attention with communication; reported latency improves against the same-round chunk path, but full-model output equivalence remains unproven.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `open_pr`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `ascend`
- Scenario: `diffusion.video_generation`
- Hardware: 4× Ascend 910B

### Software and workload

- vLLM 0.29.0, CANN 9.0.0
- Reported E2E source c36a9b6979a890c4ec9c3e8bf54000174134bea9; imported current PR head f9e5d544d33428e17b818be3a01cb7d0956e4ead
- MiniMax-H3 Ref2VA
- SP4, 1344×768, 124 frames, 24 FPS, 8 steps, seed 3100, guidance 1, flow shift 12
- Chunk size 64 MiB unchanged; head buckets 0→4
- Each arm: 2 warmups + formal + profile + formal; only unprofiled formal requests underpin latency comparison

## Findings and evidence

- **upstream_reported**: Same-round chunk mean 88906±4 ms to head-split 85195±574 ms (sample SD, n=2 per arm), reported 4.174% client E2E reduction.
- **upstream_reported**: Diagnostic per-block exposed All-to-All falls 16.359→2.455 ms while compute kernels rise 145.323→147.346 ms; reported benefit is overlap/critical-path shortening, not faster compute kernels.
- **quality_limit**: Author self-review says cross-arm video hashes differ; operator/refactor equivalence checks do not establish full-model quality equivalence. E2E data precedes the current refactor.
- **source_version_limit**: A previous-round baseline appears only as reference and must not be used as a matched control for the current 4.17% claim.

## How to use this experience

- Compare against a matched same-revision control and report exposed communication separately from aggregate collective/kernel duration.
- Tune bucket count as a workload/topology experiment; stream ownership, workspace lifetime and collective ordering are part of correctness.
- Do not adopt a generic no-regression claim from an illustrative scheduling diagram.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Run current-head multi-rank parity with uneven buckets, padding, bias, RoPE and workspace reuse.
- Repeat current-head same-source HTTP A/B with at least three unprofiled warm samples and identical placement.
- Evaluate intermediate latents plus video/audio/temporal metrics across prompt/reference sets, using an explicit quality gate.
- Test collective failure/cancellation, memory growth and CLI-to-runtime bucket/chunk propagation.

## Limits

- Open PR, not a main-supported feature declaration.
- Only two unprofiled formal samples per arm; current-head E2E and full-model quality need revalidation.
- Ascend910B/CANN evidence does not validate NVIDIA or ROCm transport.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/7514) — `f9e5d544d33428e17b818be3a01cb7d0956e4ead` (open); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/7514#issuecomment-5664200594) — `f9e5d544d33428e17b818be3a01cb7d0956e4ead` (open); accessed 2026-10-09.
