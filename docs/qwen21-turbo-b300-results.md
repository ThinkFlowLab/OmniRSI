# Qwen-Image-2.1-Turbo / local B300 × 4 results

**Measured E2E latency fell from 385.07 ms to 335.42 ms: 12.89%. The requested 20% target was not reached.** The eight declared code hypotheses are complete. All final and held-out image checks passed with identical decoded pixels. This is a measured improvement for the frozen workload, not a 20% success claim.

| Final metric | Baseline | Candidate |
| --- | --- | --- |
| Median of five trial means | 385.0696 ms | 335.4234 ms |
| Trial mean range | 379.9109–403.3866 ms | 333.7038–339.3807 ms |
| Full measured requests | 15 | 15 |
| Independent image checks including first requests | 20 | 20 |
| RGB / alpha SSIM against canonical | 1.0 | 1.0 |
| RGB / alpha PSNR against canonical | ∞ | ∞ |

The cohort alternated AB/BA and measured three complete synchronous requests per trial. Five baseline means were `[403.3866,379.9109,385.0696,385.1251,382.5984]` ms; candidate means were `[338.9145,334.5113,335.4234,339.3807,333.7038]` ms. These are descriptive medians/ranges, without a statistical-significance claim.

## Frozen scope

- Source baseline: main at preparation, `f69b1f2b19cece03f6736c7da093ea25de0e24fd`.
- Candidate: `66366303115242f1c2e35a601d37dd7ca4a81033`, DCO signed. [Applyable patch](../examples/patches/qwen21-turbo-b300.patch).
- Model: `Qwen/Qwen-Image-2.1-Turbo`, revision `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`; downloaded anonymously and frozen offline.
- Four physical B300s: IDs **2,5,6,7**, NV18 interconnect, NUMA 0, driver 610.43.02, 275040 MiB each. A 250 ms NVML observer found no external compute process on these devices during final or successful held-out runs.
- BF16, 1280×704, native eight sigmas, CFG1, TP1/Ulysses4, Ring1, VAE1/nontiled, FLASH_ATTN/FA4, full native RGBA PNG/base64 HTTP response, compression100, concurrency1.
- Torch 2.13.0+cu130, vLLM 0.31.0, Diffusers 0.40.0, Transformers 5.14.1, kernels 0.16.1, FA4 4.0.0b18; package versions/source paths and full model-file SHA256s are preserved locally.
- Fixed model-wide eager execution. Preparation found missing FA4, then default regional compilation failed with Inductor `NameError: s28`. Those failed settings are not performance baselines. This result does not compare against a working compiled deployment on another environment.
- Warm repeated prompts, new measured seeds, two warmup rounds / six full-shape requests per server as defined by the helper. Deterministic text-conditioning reuse is qualified; complete images and latents are never replayed. New prompt/seed correctness was checked separately.

## Profile evidence and configuration selection

The initial same-shape two-step profile preceded performance changes. On rank0, three requests produced 768 all-to-all calls and 227.25 ms aggregate NCCL SendRecv kernel duration, including wait. Cast/copy/layout kernels and repeated RMSNorm, RoPE and token metadata work were visible. Profile kernel sums are not E2E latency.

Three TP/Ulysses cells were measured in three full groups each. On the final device group, TP1/U4 passed quality at 384.15 ms search median; TP2/U2 and TP4/U1 failed the declared RGB similarity gate on some prompts and were ineligible. No threshold was relaxed. This establishes the best of these three tested cells, not a global optimum across other strategies/workloads.

A later diagnostic confirmed symmetric-memory communication actually executed: its byte-copy/permute kernel replaced NCCL SendRecv, with 108.54 ms aggregate kernel duration for the same three two-step requests. Normalization, layout handling and VAE/framework work still limited E2E improvement.

The profiling API's stop request timed out after all four ranks exported valid traces. Diagnostic manifests remain failed and never became performance/accuracy evidence; trace export, completed requests, source integrity and owned-process cleanup are documented separately.

## RSI iterations

Exploratory trial values below are not the final five-group result. H1 used the earlier GPU0–3 group; accepted-code probes were remeasured on IDs2,5,6,7. Source changes and every failed/cancelled attempt were retained.

| Hypothesis | Layer / change | Observed result | Decision |
| --- | --- | --- | --- |
| H1 | Pack equal Q/K/V into one 5D all-to-all | One trial 394.68 ms, quality PASS; no benefit against search baseline | Reverted |
| H2 | Bounded exact pure-T2I text-conditioning cache | Three-group median 369.54 ms; identical pixels | Retained |
| H3 | Fuse Q/K RMSNorm and complex RoPE | Unit tests passed after reduction-order fixes; complete eight-step RGB quality failed in both variants | Reverted; all frozen thresholds retained |
| H4 | Compute shared token selection, scale and gate tanh once per step instead of per block | Approximately 360 ms; identical pixels | Retained |
| H5 | Blackwell eager symmetric-memory Ulysses; fix conflicting toolkit/wheel headers | Approximately 347 ms; four-rank forward/reverse exchange exact; identical pixels | Retained |
| H6 | Graph original QKV projection/norm/RoPE without communication | Approximately 345 ms; no material E2E gain | Removed |
| H7 | Graph original post-attention residual/norm/FFN | Approximately 352 ms; copy/replay costs offset savings | Removed |
| H8 | Reuse static geometry/RoPE/mask metadata within one request's KV lifetime | Approximately 340 ms; identical pixels | Retained; final candidate above |

Retained code avoids repeated work and replaces layout communication with byte-preserving exchange. It leaves all original reduction/nonlinear arithmetic, sampling, RNG, model weights, decoder and encoding semantics intact. The text cache is limited to 16 entries/16 MiB, keyed by prompt/limit/role/model/processor/template/device/dtype, bypasses unsafe modes and clears on unload. Geometry storage belongs to a single generation/branch and does not cache newly sampled latents.

## Precision and verification

- All **30 timed images plus 10 first-request images** passed the unmodified frozen v4 standalone validator against the canonical reference. Minimum RGB and alpha SSIM=1.0, every PSNR=∞: decoded pixels are identical.
- Three unseen prompts/seeds142–144 plus their first request also passed with identical decoded pixels. This validates cache misses/new conditioning and new-seed behavior; it is not a repeated performance study of cold prompts.
- Target checks: 42 relevant CPU tests passed; four accelerator-only checks were explicitly skipped in that CPU invocation. Four-rank CUDA symmetric-memory equivalence passed for batches1/2 and sequences16/32/880. Exact shared-modulation checks passed. Rejected fusion and graph tests/results are separately retained.
- Relevant full pre-commit checks passed, including Ruff, mypy for Python3.10, SPDX, forbidden imports and CUDA-call checks. The target candidate checkout is clean and committed.
- Control-package CPU suite passed with the new runner-directory/root-resolution regressions; **85 tests, no skips** in the final CPU suite.

The initial aggregate validation command looked for `run.lock.json` under `RUN/validation`, causing a path-resolution failure. **Original frozen run/evaluation records were not edited.** Every one of the ten complete manifests was then independently checked by the same unmodified frozen v4 validator with explicit paths, covering all 40 images. The repository's aggregate path resolver was fixed prospectively and tested, without changing media metrics or thresholds. The formal 20% verdict remains **FAIL**, regardless of the infrastructure error, because measured gain was only12.89%.

## Reproduction and evidence

```bash
# The patch applies cleanly to the pinned source. Do not apply to an unrelated revision.
git checkout -b codex-qwen21-turbo-reproduce f69b1f2b19cece03f6736c7da093ea25de0e24fd
git apply /path/to/OmniRSI/examples/patches/qwen21-turbo-b300.patch
```

Follow the [active plan](qwen21-turbo-b300-test-plan.md) with newly frozen copies of the current helpers for a new run; the historical v4 guards/artifacts stay immutable. Keep exactly the same hardware/source/model/shape/schedule/environment on both sides.

Local campaign root: `/home/zjy/code/david/tmp/qwen21-turbo-b300-rsi-20261010`. It contains model/guard/asset hashes, all configurations, raw requests, accepted/rejected patches, four-rank traces and summaries, device observations, CPU/CUDA/pre-commit logs, held-out outputs and quality reports. The final frozen run is `artifacts/final-runs/20261010T063041Z-08b2b2f1a7`; independent complete quality coverage is `artifacts/final-quality-audit.json`, held-out coverage is `artifacts/heldout-quality.json`, and the final summary is `artifacts/final-summary.json`. Local paths are not downloadable GitHub attachments.

External Megatron jobs repeatedly occupied otherwise idle cards; overlapping attempts were cancelled and excluded. No outside process was terminated. Eight declared code hypotheses were exhausted. More work would need a new iteration budget and stable GPU window; a 20% claim requires a new passing five-group result, not selecting a favorable exploratory sample or weakening accuracy.
