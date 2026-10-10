# MiniMax-H3 stacked VAE tiling trades activation memory for fewer decode calls

An open stacked-tiling PR batches rank-local spatial tiles into one decoder call per temporal window; measured decode gains are exact for the reported latent but PP1 memory rises.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `open_pr`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `cuda`
- Scenario: `diffusion.video_generation`
- Hardware: NVIDIA B300 SXM6, SM103; PP4 uses GPUs 0–3

### Software and workload

- vLLM 0.28.0 benchmark environment
- Base dff28e686711add0e929ce0c5660b09ef26c8c88 + PR head b188874539903ecc04a8874b782d7d1a898d191c
- Exact decoder kernels from #6607; FP16 CUDA autocast
- MiniMax-H3 video VAE
- Fixed latent (1,24,62,48,84), 768×1344 decode, same latent broadcast to ranks
- One warmup + three measured decodes; PP latency is max rank wall-clock
- Remote VAE implements native stack_tiling; auto requires at least two global tiles per VAE rank

## Findings and evidence

- **upstream_reported**: PP1 sequential→stacked median decode: 4387→3838 ms, peak allocated 11.83→16.73 GiB. PP4: 1198→1055 ms, peak 12.49 GiB unchanged; reported torch.equal=True and max abs diff 0 for both comparisons.
- **source_inspected**: Default false keeps sequential behavior; auto checks global tile count relative to parallel size. The model's previous stack_tiling state is restored in finally.
- **counterexample**: Fewer decoder calls improve stage latency while PP1 activations consume about 4.90 GiB more. A decode microbenchmark does not establish complete generation or HTTP speedup.

## How to use this experience

- Keep the default conservative until target capacity permits tile batching.
- Evaluate sequential/stacked/auto at target PP sizes and shapes rather than assuming all parallel sizes scale.
- Prioritize this work only when VAE decode is a material E2E stage.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Run pinned-branch tiling, registry and CLI forwarding tests before using --vae-stack-tiling; the option may not exist on target main.
- Repeat decode using identical latent and record maximum-rank wall time and per-rank peaks.
- Compare decoded tensor identity/seams/order, then full same-seed video/audio and HTTP E2E.
- Qualify PP2/PP8, smaller tile grids and OOM/fallback cases separately.

## Limits

- Open/draft PR; branch/default/options may differ from current main.
- Clean current-branch PP2/PP8 latency is reported pending.
- Reported exactness applies to this latent, autocast mode and decoder revision; portability is unverified.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/7024) — `b188874539903ecc04a8874b782d7d1a898d191c` (open_draft); accessed 2026-10-09.
