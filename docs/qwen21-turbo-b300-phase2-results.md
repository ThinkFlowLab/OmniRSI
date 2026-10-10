# Qwen Turbo / B300 x4 phase 2 results

**The 30% target passed: fresh original-source baseline 384.16 -> 266.60 ms E2E, a 30.60% reduction. Against the earliest historical baseline of385.07 ms, the reduction is30.76%.** All40 formal image checks passed with identical decoded RGB/alpha pixels. The source baseline remains original main `f69b1f2b`; it was not reset to phase 1's optimized candidate.

The first frozen phase 2 cohort achieved30.203% against history but only29.8098% against its fresh baseline, so its formal verdict remains FAIL. The H19 cohort below is a separate passing result, with the earlier records preserved.

| Final metric | Original source | H19 candidate |
| --- | --- | --- |
| Median of five trial means | 384.1617 ms | 266.6048 ms |
| Trial mean range | 377.6383-399.7588 ms | 261.8247-294.6473 ms |
| Measured requests | 15 | 15 |
| Image checks including first requests | 20 | 20 |
| RGB/alpha SSIM against original canonical | 1.0 | 1.0 |
| RGB/alpha PSNR | infinity | infinity |

Baseline means: `[384.1617,382.9615,377.6383,386.3696,399.7588]` ms. Candidate means: `[264.8543,268.7395,266.6048,261.8247,294.6473]` ms. The slower fifth candidate trial remains in the aggregation; no values were discarded. These are descriptive medians/ranges, without a statistical-significance or tail-latency claim. Historical385.0696 ms comparison is cross-cohort and reported separately.

## Frozen candidate and workload

Candidate `330d411d2dd567bc061cb3124f4e39d195a4be45`, with DCO. The earlier frozen candidate was `e873b3d0e9fed62775047d8471a5d9e0cb26e67f`. Original source baseline remains `f69b1f2b19cece03f6736c7da093ea25de0e24fd`. [Complete three-commit patch](../examples/patches/qwen21-turbo-b300-phase2.patch) applies cleanly to that baseline, including the original phase 1 changes. Latest main was fetched for inspection; unrelated newer changes are excluded from this frozen experiment.

TP1/Ulysses4/Ring1/VAE1, BF16, 1280x704, CFG1, native eight sigma steps, concurrency1, repeated prompts with new seeds, six warmups/server and full native RGBA PNG/base64 HTTP response. Model revision, runtime, assets, driver and frozen v4 numerical validator are unchanged. The first cohort used physical B300 IDs2/5/6/7. After foreign GPU5 activity interrupted follow-up, H19 and a fresh original-source baseline were separately qualified on IDs2/4/6/7, also NV18/NUMA0. Both sides of the new cohort use exactly those four cards. Cold/unseen-prompt checks are correctness checks, not a cold-prompt latency claim.

## Further hypotheses, in order

H9-H16 are the eight initial code rounds in phase 2. H17 and H19 add two code rounds; H18 is a launch-API feasibility probe, without production integration. Together with phase 1, this is 18 code hypotheses plus one investigation. Corrections/reruns are not extra rounds.

Values are exploratory medians of three group means under accumulating candidates. They are not independent additive gains or the final five-group result.

| Round | Optimization | Exploratory E2E | Accuracy and disposition |
| --- | --- | --- | --- |
| H9 | Residual gate multiply/add and SiLU/multiply fusion, preserving BF16 materialization | 331.45 ms | Identical decoded pixels; retained |
| H10 | RoPE-only fusion, matching the real-leading / imaginary-trailing CUDA complex FMA order | 327.15 ms | Identical decoded pixels; retained as the normalization fallback |
| H11 | Fuse Q/K cast/square and normalization/RoPE suffix around the original ATen mean | 300.94 ms | Identical decoded pixels; superseded by H13 |
| H12 | Reuse the existing exact Wan vector-norm/Triton epilogue for Qwen's equivalent native VAE norm | 280.25 ms | Identical decoded pixels; retained; no convolution/tiling changes |
| H13 | Fully fuse head128 Q/K norm/RoPE with ATen's four-values-per-lane reduction tree and both BF16 casts | 277.44 ms in the quiet trial | Identical decoded pixels; retained; scope restricted to the validated CUDA/BF16/PyTorch2.13 path |
| H14 | Broadcast decode-only scale/gate parameters instead of dense repeated rows | 283.61 ms | Accuracy passed, no demonstrated E2E benefit; removed |
| H15 | Direct strided staging + one symmetric-memory exchange + direct unpack for equal Q/K/V | 265.19 ms after removing H14 | Four-rank exact exchange and identical image pixels; retained |
| H16 | Skip concatenating the empty decode joint query | 271.52 ms | Accuracy passed, no demonstrated E2E benefit; reverted |
| H17 | Reorder eligibility checks and avoid repeated reads of the first tensor's metadata | 359.88 ms during a host load spike | Accuracy passed; no demonstrated benefit; reverted. Host load108 and outside CPU workers were observed, without assigning the timing difference a proven cause |
| H18 | Probe already-compiled Triton launch versus JIT dispatch, with fresh-pointer correctness | Standalone host enqueue6.41 -> 3.95 us; no E2E trial | Numeric/new-pointer probe passed; not integrated because a safe cache adds metadata work and the small isolated saving is not an E2E result |
| H19 | Validate native BF16 pointwise layout/type contracts once per denoise step, bypassing repeated checks through native unquantized linear/norm blocks | 266.47 ms on qualified GPUs2/4/6/7 | Identical decoded pixels; revised frozen candidate. Other modes retain guarded fallbacks |

H13's first integration attempt had an indentation error and never reached readiness. Its subsequent trial overlapped an owned Nsight diagnostic on another GPU and measured 342.14 ms; a later quiet run measured 277.44 ms. All attempts remain preserved. This discrepancy is not assigned a proven cause. Formal measurements run no owned GPU diagnostics on any card concurrently. Numeric corrections, tuning and reruns do not add independent rounds.

Unlike rejected H3, the surviving norm kernel matches the installed ATen reduction grouping and complex multiplication's actual FMA placement. CUDA operator checks alone did not promote it: complete native-eight-step images also passed. Unsupported dtype/head geometry, gradients, compilation and stack combinations retain eager arithmetic.

## Reprofile and KDA evidence

The second phase profiled H8 **before new edits** at the same image shape and two explicit sigma steps, then profiled the frozen final candidate. Each rank0 diagnostic covered three requests and 192 transformer blocks:

| Diagnostic count | H8 | Frozen phase 2 |
| --- | --- | --- |
| Symmetric-memory exchange kernels | 768 | 384 |
| CUDA kernel launches (`cudaLaunchKernel`) | 10,581 | 4,173 |
| Full norm/RoPE fused kernels | 0 | 192 |
| Batched QKV stage / unpack kernels | 0 / 0 | 192 / 192 |
| Exact VAE norm epilogues | 0 | 108 |

Communication call count halved; kernel launch count fell 60.56%. Aggregate communication kernel durations were 151.81 and 143.87 ms, including waits. The count reduction does not imply halving critical-path communication time. Inclusive profiled `diffuse` durations were 298.65 -> 272.05 ms/three requests and `_decode_latents` 193.39 -> 142.82 ms/three requests; these contain diagnostic synchronization/overhead and are not E2E results.

[KDA workflow and pinned skills](qwen21-turbo-b300-phase2-plan.md#linked-external-skills) informed the task contract, representative operator harnesses and promotion ledger. The MiniMax-H3 [Sol Engine kernel reference](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) motivated elementwise/norm/activation fusion; its approximate methods and published speedups are not local Qwen results.

Nsight Compute 2026.3 on B300 collected overview/PM-sampling and source reports. Representative original residual multiply `[1,880,4096]` measured 6.464 us, SM throughput9.82%, DRAM read throughput29.26% of sustained peak and 2.23 TB/s. Fused QK `[1,880,32,128]` measured 13.760 us, SM throughput43.72%, DRAM throughput14.12%, active warps83.18% and 23 registers/thread. Missing CTC metrics and the unavailable old `PmSampling_WarpStates` section are preserved; no unavailable counter is inferred. Four production shapes and all 65,536 BF16 SiLU values passed eager-equivalence checks.

Profile stop requests hit the existing API timeout after all four valid traces exported. Their diagnostic manifests remain failed. They provide bottleneck/call evidence and never become accepted latency/accuracy records.

H19's same-shape three-request/two-step profile confirmed pointwise eligibility calls fell from576 to6, taking0.064 ms inclusive in the new profile. Kernels, sampling and tensor values are not cached by this change. Its new four-rank traces are retained, together with the unchanged profiler-stop timeout record.

## First phase 2 formal cohort, preserved FAIL

Run `20261010T083057Z-fa0beb7b73`, candidate `e873b3d`, GPUs2/5/6/7:

- Original-source baseline means: `[420.1766,382.3601,400.2663,382.9154,381.2855]` ms, median382.9154 ms.
- Candidate means: `[268.7692,270.4295,265.0256,269.9344,267.6384]` ms, median268.7692 ms.
- Fresh comparison:29.8098% reduction, formal **FAIL**. Historical385.0696 ms comparison:30.203%, reported separately.
- All40 images passed the unchanged v4 validator, with identical decoded RGB/alpha pixels. Source integrity passed11 command checks; no foreign process overlap was observed on the selected cards.

Neither its values nor verdict were edited when proceeding to H19.

## Passing H19 formal cohort

Run `20261010T092145Z-61055bbc49`, candidate `330d411`, GPUs2/4/6/7. Five alternating AB/BA groups completed with11 passing source-integrity command checks. The unchanged v4 validator, invoked through the frozen directory adapter, covered all30 measured images and10 first requests; every decoded RGB/alpha pixel matched the original canonical. The 250 ms NVML observer found no foreign compute process on these selected cards during the successful cohort. GPU5's outside task was not terminated or included in the selected group.

Three unseen prompts/seeds142-144 plus their first request also passed with identical decoded RGB/alpha pixels (SSIM1, PSNR infinity). The reference is the preserved original-source held-out baseline; the candidate ran a fresh full E2E server/generation on the new fixed group. This verifies new conditioning and new seeds without claiming cold-prompt latency improvement. Owned server processes were cleaned up after every completed/aborted trial.

## Validation and artifacts

- Target CPU checks: 127 passed, 14 skipped, 24 deselected in the CPU invocation. Accelerator cases were separately selected rather than treating those skips as passes.
- Exact operator/VAE/FakeTensor checks: 19 passed in a CUDA-enabled invocation.
- Four-rank batched QKV exchange: exact for batch1/2 and local sequence16/32/880/913, including row-strided projected inputs.
- Relevant pre-commit passed, including Ruff, mypy3.10, test marks, SPDX and forbidden-import/CUDA checks. Candidate checkout is clean and DCO committed.
- Control CPU suite: 87 passed, no skips. The new adapter has path/repetition regression checks.
- After H19,109 relevant model CPU checks passed (23 accelerator cases deselected);10 pointwise/shared-modulation CUDA/CPU checks passed, including the verified-contract path. Its relevant pre-commit passed, including mypy3.10. Unchanged QK/VAE/exchange checks remain as above.
- The historical v4 validator is unchanged. A frozen directory adapter maps `RUN/validation/result.json` to the run root before invoking it, then requires ten complete reports and all40 image comparisons. It changes no images, similarity formula, threshold, source identity or legacy record.

Local evidence: `/home/zjy/code/david/tmp/qwen21-turbo-b300-rsi-20261010/artifacts/phase2/`, including every exploratory attempt, full four-rank traces, standalone NCU reports/metrics, commands, source snapshots, numerical checks, final five-group results and held-out outputs. Local artifacts are not GitHub-downloadable attachments. Generated image artifacts are not committed.

The passing run is `final-h19-runs/20261010T092145Z-61055bbc49`; complete frozen validation is its `validation/result.json`. `final-summary.json` records both denominators and the44 final/held-out comparisons. `heldout-h19/quality.json` retains the independent held-out report. The earlier failed cohort is `final-runs/20261010T083057Z-fa0beb7b73`.
