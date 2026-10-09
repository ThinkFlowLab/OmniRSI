# Diffusion measurement and profiling discipline

Freeze the serving, single-request and benchmark commands; measure performance without diagnostic profiler overhead, then use targeted traces to choose one-variable experiments.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `guidance`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `common`
- Scenario: `diffusion.optimization`
- Hardware: Methodological guidance; no hardware validation is claimed

### Software and workload

- vllm-omni guidance snapshot ee81f33001922f8d91dd09da0a57ee1531a8df82; parser flags and model capabilities must be checked at the target revision
- Wan/Qwen/Flux-style image and video pipelines; adapt to other diffusion models
- Named device SKU/count/topology, model revision, assets, seed, shape, frames, schedule, dtype, attention backend and compile state are frozen
- Backend-native profiler and low-overhead stage timings are available or their absence is recorded

## Findings and evidence

- **guidance**: Performance baseline and operator/host-stack diagnostic traces serve different purposes. PyTorch profiler, shape recording and stack collection can distort measured latency.
- **guidance**: Compare client E2E, server inference, encode, denoise, decode and peak memory independently. A kernel or denoise win can be offset by decode, communication or transport.
- **guidance**: For CFG-enabled models, compare CFG2×USP(world/2) with USP(world); treat HSDP primarily as capacity until an A/B demonstrates speed; tune VAE patch parallel only where its stage share is material.
- **guidance**: Every optimization, including intended math-preserving layout/cache changes, needs artifact and numerical evidence. Quantization, sparse/approximate attention and reduced steps are quality-sensitive hypotheses.

## How to use this experience

- Freeze explicit CLI argument vectors for server startup, single request, benchmark and diagnostic collection separately; retain raw responses and exit statuses.
- Run an explicit warmup policy and at least three measured repetitions for shortlisted fixed-work candidates; retain every sample and report median/mean plus spread in milliseconds.
- Collect operator+shape and host-stack traces as separate diagnostic runs. Start at rank 0; expand when imbalance/communication hypotheses justify it.
- Choose candidates from measured dominant stages and trace evidence; include both a latency objective and memory/quality constraints.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Confirm A/B source, assets, random seed, schedule, attention backend, precision, compile warmup and resource placement differ only in the tested variable.
- Run unprofiled baseline and candidate, validate output artifacts, measure stage and client timings, and compare against baseline self-run variance.
- Validate per-rank device memory and host memory; do not derive p95/p99 from three fixed-work repeats.
- On ROCm/Ascend use their appropriate profiling workflow; CUDA kernel/profiler evidence does not establish another backend.

## Limits

- This is imported methodological guidance, not a performance result.
- Parallelism and compiler benefits depend on model implementation, shapes, topology and target software.
- No remote device run or upstream tool execution was performed by OmniRSI during this import.

## Sources

- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/diffusion-perf-opt/SKILL.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/diffusion-perf-opt/references/optimization-playbook.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
