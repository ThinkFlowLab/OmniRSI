# MiniMax-H3 exact AdaLN schedule cache: capacity benefit can exceed latency benefit

An open PR precomputes finite-schedule AdaLN tables and can release projection weights; production-shape compiled measurements show capacity savings without a distinguishable speed gain.

This entry is an attributed summary of upstream material. It is `imported`, locally verified is `false`, and evidence level is `open_pr`. Search/show operations read it as evidence; they do not execute upstream instructions, scripts, patches or commands. Hardware results below are upstream reports, not OmniRSI test results.

## Scope

- Repository: `vllm_omni`
- Backend: `cuda`
- Scenario: `diffusion.video_generation`
- Hardware: 2× NVIDIA H20 98 GB for TP2; one H20 for the separately reported layerwise run

### Software and workload

- PR head/hardware commit 635215a96d3eaf6b5bac66020feae51b34daa5c9
- Python 3.12, PyTorch 2.11.0+cu128, vLLM 0.26.0, FlashAttention
- MiniMax-H3 FL2VA / T2VA
- Small eager row: 448×256, 29 requested/39 aligned frames, 50 sigma points/49 forwards, seed 42, TP2/text-encoder TP2, flow shifts 12/3
- Production regional-compile row: 1248×768, 209 frames, 50 steps, seed 2101, TP2
- Single request in flight; schedule hits and cold misses measured separately

## Findings and evidence

- **upstream_reported**: Small eager row: five steady requests after warmup; wall mean 15285.8 ms disabled, 14987.9 ms cache retained, 15029.4 ms cache+weight-offload. These are roughly 1.95% and 1.68% reductions for that row.
- **upstream_reported**: Per TP rank, 12.144 GiB projection weights replaced by 0.885 GiB tables yields 11.259 GiB net device-memory reduction by accounting. First-build peak still includes coexisting weights and tables.
- **counterexample**: Production compiled row: one measured steady sample per mode; retained cache changed client wall 1298497→1298456 ms, effectively no distinguishable speed benefit. Weight offload increased client wall to 1305133 ms (~0.51% slower) while lowering residency ~11.256 GiB/device.
- **upstream_reported**: Fixed steady outputs were reported byte-identical across the three modes. Schedule changes require restore/build/atomic replace/re-offload and can add substantial miss latency; cold-to-steady audio differences also exist in the disabled control.

## How to use this experience

- Treat weight release as a capacity optimization with a latency/memory frontier, not an additional automatic speedup.
- Key exact schedules and all dependent configuration/adapter/device/dtype identities; bound cache size and preserve distributed ownership.
- Reserve memory for cold build and schedule replacement; a lower steady footprint cannot solve an unfit first-build phase.

Use the exact linked source and inspect the target checkout before selecting an implementation or parser flag. An open PR's flags or code may not exist in upstream main. H200/B200, ROCm and Ascend results are not interchangeable.

## Remote validation plan

- Use the PR's pinned implementation or first verify an equivalent path exists in target source; it was open at import.
- A/B cache off, cache retained and cache+offload with fixed workload and compile state, at least three warm repeats.
- Measure cold miss, same-schedule hit and changed-schedule rebuild independently; capture NVML/device and allocator peaks.
- Compare steady output hashes plus decoded video/audio and test layerwise/HSDP/quantized ownership separately.

## Limits

- Source PR open; target main support is not implied.
- Production-shape deltas are one-sample observations and sub-1% values do not establish statistically meaningful speed changes.
- H20 and specific versions/schedules only; H200/B200 or another backend needs new validation.

## Sources

- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5783) — `635215a96d3eaf6b5bac66020feae51b34daa5c9` (open); accessed 2026-10-09.
- [Upstream PR / discussion](https://github.com/vllm-project/vllm-omni/pull/5783#issuecomment-5192067994) — `635215a96d3eaf6b5bac66020feae51b34daa5c9` (open); accessed 2026-10-09.
