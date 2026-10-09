# vLLM-Omni diffusion knowledge

These are source-pinned, attributed summaries imported on 2026-10-09. No upstream skill was installed or executed and no remote GPU/NPU experiment was run for the import. All records remain `status=imported` and `verification.locally_verified=false`.

The authoritative guidance snapshot is [vllm-omni ee81f3300192](https://github.com/vllm-project/vllm-omni/commit/ee81f33001922f8d91dd09da0a57ee1531a8df82). PRs are pinned independently to their reviewed head, with merge status recorded as observed on the import date. Head/base/hardware-test revisions can differ; each record explains such differences.

## Entries

| Backend | Entry | Evidence |
| --- | --- | --- |
| common | [Diffusion measurement and profiling discipline](experiences/common/diffusion/measurement-and-profiling/README.md) | guidance |
| common | [Production diffusion integration and compatibility gates](experiences/common/diffusion/production-integration-gates/README.md) | guidance |
| cuda | [MiniMax-H3 guarded Q/K RMSNorm and packed RoPE fusion](experiences/cuda/diffusion/minimax-h3-qk-norm-rope/README.md) | upstream_reported |
| cuda | [MiniMax-H3 modulation fusion changes rounding and requires platform guards](experiences/cuda/diffusion/minimax-h3-fp32-modulation/README.md) | upstream_reported |
| cuda | [MiniMax-H3 exact AdaLN schedule cache: capacity benefit can exceed latency benefit](experiences/cuda/diffusion/minimax-h3-adaln-schedule-cache/README.md) | open_pr |
| cuda | [MiniMax-H3 Ref2VA TeaCache needs task-specific calibration and step defaults](experiences/cuda/diffusion/minimax-h3-ref2va-teacache/README.md) | open_pr |
| cuda | [MiniMax-H3 stacked VAE tiling trades activation memory for fewer decode calls](experiences/cuda/diffusion/minimax-h3-vae-stacked-tiling/README.md) | open_pr |
| ascend | [MiniMax-H3 SP head buckets reduce exposed All-to-All with quality still pending](experiences/ascend/diffusion/minimax-h3-sp-head-buckets/README.md) | open_pr |
| ascend | [MiniMax-H3 reference KV reuse must balance ownership after Ulysses](experiences/ascend/diffusion/minimax-h3-reference-kv-reuse/README.md) | open_pr |
| cuda | [MiniMax-H3 online CUDA MXFP8 is an opt-in approximate deployment lane](experiences/cuda/diffusion/minimax-h3-cuda-mxfp8/README.md) | open_pr |

## Retrieval and promotion

- Match repository/backend/scenario before applying device, software, task and workload conditions.
- The `common` entries provide measurement and production-integration methods for CUDA, ROCm and Ascend, with backend-native tooling and independent hardware qualification.
- This first import has concrete CUDA and Ascend MiniMax-H3 cases. It contains no claimed ROCm H3 optimization result. CUDA fusion/quantization and Ascend overlap cannot be promoted to ROCm by analogy.
- Preserve negative results: no distinguishable production-shape AdaLN speed gain, higher stacked-tiling memory, slower pre-Ulysses KV cache, zero TeaCache hits and MXFP8 loading OOM.
- An imported/open-PR record can inform a hypothesis. Only a linked remote experiment with exact source/environment/workload, appropriate quality gates and repeatable measurements can establish a local validated experience.
- Latency values in summaries are milliseconds; GiB/MiB and allocator versus device-residency memory are kept distinct. Small fixed-work sample counts do not establish production tail latency.
- OmniRSI's conceptual validation levels and upstream CI levels/readiness tracks are separate taxonomies.

## Additional reviewed candidates

The following sources were considered but not promoted into performance records:

- [#5819 static conditioning](https://github.com/vllm-project/vllm-omni/pull/5819), head `cae3cdb3c42a8604b61b89e3170278aa25168fff`: request-local projection/refiner/RoPE hoisting; CPU equivalence/call-count evidence, no full-generation latency claim.
- [#5765 offline example](https://github.com/vllm-project/vllm-omni/pull/5765), head `86a4dd4e3106058dd2ab7137402c41efd19ef049`: Ascend warmup/profiling example, one measured 2-step generation; not production repeated performance evidence.
- [#7470 RainFusion AllGather-KV](https://github.com/vllm-project/vllm-omni/pull/7470), head `6b9a54452c38fa14a7129aa68de4936cad8fd54e`: open/draft; only static checks reported, pytest/NPU multi-rank validation not run.
- [#7519 E2E acceleration roadmap](https://github.com/vllm-project/vllm-omni/pull/7519), head `7ba27060000028a1ab5fc86058271e1fb9f9da29`: open/draft split plan; regression failures and no full-model E2E qualification. A collection of proposed optimizations is not a cumulative speedup.
- [#8376 ROCm pinned host slabs](https://github.com/vllm-project/vllm-omni/pull/8376), head `4e8c4bdd2be83dddd8bae45be4b69aec1dff1bb2`: open; production offloader CI and separate diagnostic Flux2 evidence. It is not MiniMax-H3 performance/quality evidence, so it remains a future backend-wide import candidate.
- [#7543 model/hardware recipe separation](https://github.com/vllm-project/vllm-omni/pull/7543), head `d25a11b4faae491d6b3b54e4a8052b051872d1c5`: documentation design, explicitly establishes no new hardware performance or quality results.

Do not interpret a roadmap, documentation change, shared platform abstraction or successful unit test as a hardware deployment qualification.
