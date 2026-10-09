# Production diffusion integration and compatibility gates

Separate Day-0 loading support from production readiness; qualify exact task/backend/hardware combinations through API parity, strict loading, quality, offload/cache lifecycle, cancellation and load stability.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `guidance`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `common`
- Scenario: `diffusion.integration`
- Hardware: Methodological guidance; no hardware validation is claimed

### Software and workload

- vllm-omni source snapshot ee81f33001922f8d91dd09da0a57ee1531a8df82; online FP8/DLO/cache/batching support must be qualified at target commit
- New or existing diffusion integrations, including MiniMax-H3
- The initial pipeline/registry/loader vertical slice already works
- Official implementation, checkpoint and target repository revisions are pinned independently

## Findings and evidence

- **guidance**: A compatibility row includes task, API/execution mode, shape/schedule, backend, packed layout, cache, quantization, offload, topology, hardware, dtype and output transport. Unproven combinations default to not tested.
- **guidance**: Strict loading must account for every retained tensor and all source shards of fused QKV/gate-up destinations. Shared layers must use their platform dispatcher, with explicit guarded fast paths and native fallbacks.
- **guidance**: Online quantization, pre-quantized/pruned/rotated checkpoints, fixed adapter fusion and DLO are distinct evidence lanes; resident and cold/materialization memory peaks must be recorded.
- **guidance**: Mutable scheduler, RNG, latent and cache state belongs to each request. Structural request-batch/step-execution interfaces do not establish model throughput, quality or cancellation safety.
- **guidance**: Function, Accuracy, Performance and Reliability are independent readiness tracks. They must not be equated automatically with OmniRSI's conceptual L0–L4 validation levels or upstream CI's numbered taxonomy.

## How to use this experience

- Normalize offline/sync/async task inputs through one validated contract and test invalid boundaries before engine submission or persistent side effects.
- Maintain scoped states validated/limited/unsupported/not tested in experiment evidence; keep imported guidance status separate from remote validation.
- Prefer shared fast primitives and preserve dtype/rounding/materialization boundaries; measure and qualify each vendor/card/topology independently.
- Build per-request cache keys and lifecycle tests covering success, error, abort, disconnect and the next uncached request.
- Retain exact deployment CLI arguments and generated environment snapshots; any upstream YAML examples remain source references rather than OmniRSI run input.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Compare the official dense reference with component, block, intermediate latent and final media outputs under fixed assets/seed/schedule.
- Run strict-load and negative-task tests, uneven packed samples, group nonmember tests, and one output per request.
- Validate FP8, cache, DLO, TP/SP and compile separately before testing combinations; measure cold peak, resident peak, per-stage memory and host PSS.
- Test cancellation at queue/encode/denoise/decode/output boundaries, late worker results, OOM and worker failure, then issue a known-good request.
- Run declared below/near/above-saturation arrival loads and report success/errors, queue time, resource slope and enough samples for claimed percentiles.

## Limits

- The source describes a target readiness process; it does not prove that every model or hardware satisfies the gates.
- CUDA/ROCm/Ascend support, sparse attention, HWR/mmap, quantization and offload are independent scoped qualifications.
- No local GPU/NPU, generated-media quality, real serving or soak validation was performed for this knowledge import.

## Sources

- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/production-add-diffusion-model/SKILL.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/production-add-diffusion-model/references/api-and-recipes.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/production-add-diffusion-model/references/feature-patterns.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/production-add-diffusion-model/references/performance-patterns.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
- [Pinned upstream guidance](https://github.com/vllm-project/vllm-omni/blob/ee81f33001922f8d91dd09da0a57ee1531a8df82/.claude/skills/production-add-diffusion-model/references/production-validation.md) — `ee81f33001922f8d91dd09da0a57ee1531a8df82`; accessed 2026-10-09.
