# MiniMax-H3 reference KV reuse must balance ownership after Ulysses

An open reference-KV reuse PR fixes a slower pre-Ulysses prototype by caching full reference sequences per local head shard after All-to-All; reuse remains quality-sensitive and evidence is one-shot.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `open_pr`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `ascend`
- Scenario: `diffusion.video_generation`
- Hardware: 4× Ascend 910B3, 64 GiB HBM/device

### Software and workload

- Hardware environment vLLM 0.28.0, vllm-omni abbreviated validation commit 85d63868
- Current rebased PR head 6de9a18572ba3211fc33f47c8e5beaea8d895e98 targets vLLM 0.30 APIs; local rebase environment was incompatible
- MiniMax-H3
- USP4/Ring1, 1344×768, 5-second output, 24 FPS, 50 steps
- Tier2, refresh interval 8, ring size 2, unquantized BF16 host cache
- 4 images / 2-second reference video / 5-second reference video; one warmup then one measured request each

## Findings and evidence

- **counterexample**: Pre-Ulysses prototype worsened E2E: 4 images 878321→913491 ms; 2s video 477610→675020 ms; 5s video 925693→1270051 ms. Sequence shards gave uneven reference ownership/H2D critical paths.
- **upstream_reported**: Post-Ulysses E2E: 739566 ms (4 images), 441096 ms (2s video), 846829 ms (5s video), reported reductions 15.80%, 7.65%, 8.52% against disabled baselines. Peak HBM rises slightly.
- **source_inspected**: Post-All-to-All cache keeps the full reference sequence for each rank's local head shard; pinned host storage is staged through a 2–3-slot device ring with ready/consumed events.
- **quality_limit**: Outputs were manually inspected with no obvious difference on tested prompts/seeds; no objective quality metric or repeated variance. INT8/FP8 host-cache modes have unit coverage but no Ascend E2E benchmark.

## How to use this experience

- Treat periodic reference reuse as approximate unless timestep invariance is proven; BF16 storage alone does not make cached values exact.
- Audit token/head ownership before choosing cache placement. Balance H2D and cache synchronization across SP ranks.
- Separate unquantized reuse from INT8/FP8 host-cache quality/performance rows.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Use a dependency-compatible exact branch; do not mix the vLLM 0.28 hardware environment with rebased vLLM 0.30 APIs without revalidation.
- Run targeted reference-KV and parallel tests, including disabled, refresh, ring staging, fallback and SP boundary handling.
- Repeat each matched reference workload at least three times and capture per-rank H2D, memory, latency and balance.
- Compare trajectory/video/audio/temporal metrics, varied prompts/references and refresh policies before promotion.

## Limits

- Open PR; hardware validation source differs from rebased PR head.
- One-shot measurements after warmup cannot establish variance or production tail latency.
- Objective quality, host quantization E2E, remote reliability and cross-backend portability remain unverified.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/8072) — `6de9a18572ba3211fc33f47c8e5beaea8d895e98` (open); accessed 2026-10-09.
