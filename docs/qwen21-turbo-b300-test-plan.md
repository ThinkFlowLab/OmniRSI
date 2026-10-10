# Local Qwen-Image-2.1-Turbo / B300 × 4 performance plan

This replaces the active LTX-2.5 campaign. [Completed local results](qwen21-turbo-b300-results.md): 12.89% measured E2E improvement; the 20% goal was not reached. The old LTX plan and its blocked evidence remain historical. A plan is not a performance result.

## Goal, workload and non-goals

Reduce warm synchronous image-generation E2E latency by **at least 20% relative to a freshly measured, quality-passing best tested four-card configuration**, using the same configuration on baseline and candidate:

`median(candidate trial mean ms) / median(baseline trial mean ms) <= 0.80`

Five alternating AB/BA groups; each trial measures the mean of three fixed prompts/seeds. Startup, weight downloads, compilation/warmups and profiler overhead are reported separately. No pooling or selecting favorable prompts.

| Control | Frozen value |
| --- | --- |
| Source | Latest main at preparation: `f69b1f2b19cece03f6736c7da093ea25de0e24fd` (includes Turbo support #8699) |
| Model | `Qwen/Qwen-Image-2.1-Turbo`, revision `d65dbc9a7e8f6b5479e33dee6030eaab2a906509` |
| Hardware | Same local B300 IDs 2,5,6,7; NV18; NUMA 0 |
| Task | Text to image, concurrency 1, one native RGBA PNG per request |
| Resolution | 1280×704, 720p class, both dimensions divisible by 32 |
| Precision | BF16; native prefix KV cache; no quantization or approximate cache |
| Steps | Model's exact 8 sigmas: `[1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568]` |
| Guidance | CFG=1; no negative branch, CFG ranks=1 |
| Attention | FLASH_ATTN on both sides |
| Parallel search | TP/Ulysses = 1/4, 2/2, 4/1; Ring=1, VAE=1, no tiling |
| Compile | Fixed `--enforce-eager` for all sides/cells; default compilation failed startup with Inductor `NameError: s28` on this environment |
| Requests | Three fixed prompts and seeds 42/43/44 in the helper |
| Warmup | Two full-shape rounds, seeds 10042..10044 and 11042..11044 |
| E2E | POST `/v1/images/generations` until complete JSON/base64 PNG response downloaded |
| Diagnostic | Same shape, explicit 2-element sigma override; no performance metric or quality eligibility |

Changing `num_inference_steps` alone does **not** shorten Turbo: its saved sigma grid overrides it. Profiling uses explicit request `sigmas` instead. The official eight-element grid remains fixed for formal E2E and accuracy checks.

The workload measures repeated prompts with new seeds. Deterministic conditioning reuse may be qualified as an optimization, but final images/latents may never be replayed. Record cold/unseen-prompt behavior separately, and test cache misses, prompt changes, invalid limits and request isolation. A repeat-prompt gain must not be presented as a universal cold-prompt gain.

Non-goals: changing model, steps, precision, image size, CFG, noise/seed handling, PNG encoding, response scope or quality thresholds; image editing, throughput/batching, other SKUs and production-tail latency.

## Phase 0: prepare and freeze

Initial exploratory runs used GPUs 0–3; external Megatron jobs later occupied them. An attempted GPUs 4–7 window was interrupted by a foreign GPU4 job. Final qualification and comparisons use GPUs 2,5,6,7, with fresh reference/search/baseline data and a 250ms NVML observer. External GPU overlap interrupts only the owned helper and invalidates that trial. Earlier numbers are not the final 20% denominator. The observer keeps its logs outside trial directories. A preparation-only driver correction allows OmniRSI's pre-created trial directories while continuing to reject existing result files; freeze v4 before the new cohort.

Work under `/home/zjy/code/hsliu`, on `codex-` branches. Pull main and freeze the SHA before measurement. Keep artifacts/cache outside source. Use `/home/zjy/code/hsliu/venvs/codex-qwen21-turbo-b300/bin/python`; verify imports and `findmnt -T` for environment/model/cache.

Preparation retained both failed startup probes: missing FA4, then the default compile path failed with an Inductor `s28` name error. No valid performance/reference run occurred under those failed settings. The new protocol freezes eager execution before canonical runs; this preserves the full BF16 schedule and is not a measured speedup.

Preparation first found missing Blackwell FlashAttention support; install the pinned main extra `flash-attn-4[cu13]==4.0.0b18` into the task venv before freezing. FA2/FA3 are not a compatible substitute on B300.

The exact snapshot is public and downloaded anonymously (`token=False`); do not inherit the expired token from the old LTX campaign. Keep assets local and offline after preparation. Freeze all downloaded file SHA256s and their size/mtime; neither candidate nor benchmark may modify weights/configs. Freeze the trial driver, sibling lifecycle helper and independent quality script SHA256s before canonical runs. No dependency changes after freezing.

```bash
export LAB=/home/zjy/code/david/tmp/qwen21-turbo-b300-rsi-20261010
export PY=/home/zjy/code/hsliu/venvs/codex-qwen21-turbo-b300/bin/python
export BASE=/home/zjy/code/hsliu/qwen21-turbo-b300-baseline
export CAND=/home/zjy/code/hsliu/qwen21-turbo-b300-candidate
export MODEL=$LAB/cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/d65dbc9a7e8f6b5479e33dee6030eaab2a906509
export CUDA_VISIBLE_DEVICES=2,5,6,7
export OMP_NUM_THREADS=4
```

The checkpoint VAE emits four channels. Preserve native RGBA throughout encoding and validation; do not silently discard transparency. PNG output compression is explicitly fixed to 100 (lossless, lowest PNG compression).

For a new reproduction, create a new `guards-next` directory, copy the current `qwen21_turbo_b300.py`, `qwen21_turbo_quality.py` and `ltx25_b300.py` from `examples/`, record SHA256s and make them read-only **before a new canonical reference**. The historical `guards-v4` and measured manifests remain immutable. Current aggregate validation resolves `RUN/validation/result.json` correctly; never overwrite old references/result paths.

The control package stays inference-library-free. The independent PNG quality example uses the prepared environment's Pillow, NumPy and scikit-image; freeze their versions as well.

## Phase 1: feasibility, canonical reference and profile first

1. GPU-free probes: validate source/asset identities, imports, driver/device allocation and exact sigma/PNG contracts.
2. One full eight-step four-worker feasibility run (TP1/U4), with genuine `/health` readiness and full image completion. Preserve failure/startup logs.
3. Generate single-worker canonical reference and independent repeatability run. Require complete 1280×704 RGBA PNG, matching frozen metadata, SHA integrity, **SSIM≥0.99 and PSNR≥35 dB for every image's RGB and alpha channels separately**. If reference repeatability fails, stop without weakening thresholds.
4. Run initial TP1/U4 short-step profile before any performance code or configuration optimization. Distinguish text encoder, prefill/decode DiT, VAE, communication, copies and CPU/PNG/HTTP overhead. Diagnostic stage timers synchronize the GPU and are not E2E performance evidence.

```bash
"$PY" "$LAB/guards-next/qwen21_turbo_b300.py" \
  --source-repo "$BASE" --model-path "$MODEL" --cache-root "$LAB/cache" \
  --single-reference --tp 1 --ulysses 1 --output "$LAB/artifacts/reference-v4-gpu2-5-6-7/result.json"
# Repeat in a fresh reference-repeat directory, then validate it independently.
"$PY" "$LAB/guards-next/qwen21_turbo_quality.py" \
  --reference-manifest "$LAB/artifacts/reference-v4-gpu2-5-6-7/result.json" \
  --candidate-manifest "$LAB/artifacts/reference-repeat-v4-gpu2-5-6-7/result.json" \
  --output "$LAB/artifacts/reference-repeatability.json"

"$PY" "$LAB/guards-next/qwen21_turbo_b300.py" \
  --source-repo "$BASE" --model-path "$MODEL" --cache-root "$LAB/cache" \
  --tp 1 --ulysses 4 --diagnostic --profile-steps 2 \
  --output "$LAB/artifacts/initial-profile/result.json"
```

## Phase 2: qualify four-card configurations

Compare the three declared TP/Ulysses cells, all using four workers. Keep VAE1/nontiled decode fixed to avoid decoder changes at this small shape. Each cell uses one server, six full-shape warmups and three complete measured groups; all nine images must pass canonical quality. Preserve every failed configuration. Rank the median of three group means, then freeze the lowest-latency eligible cell as `best.json`.

```bash
"$PY" "$LAB/guards-next/qwen21_turbo_b300.py" \
  --source-repo "$BASE" --model-path "$MODEL" --cache-root "$LAB/cache" \
  --tp 1 --ulysses 4 --repetitions 3 --output "$LAB/artifacts/search/tp1-u4/result.json"
# Repeat for TP2/U2 and TP4/U1, with independent output directories.
# Run the independent quality script on each complete cell before selecting best.
```

This is the best of three tested cells for this exact workload, not a global optimum. TP and SP support are conditional on this model's actual output correctness. No eligible cell means stop.

## Phase 3: layered RSI code iterations

Re-profile the frozen winner if it differs from the initial configuration. For each iteration write a falsifiable hypothesis, isolated variable, fixed controls, evidence, acceptance and stop condition:

1. Pipeline stages: remove unnecessary encoder/conditioning work while preserving templates, masks, limits, new-seed behavior and image-conditioned fallback.
2. Distributed execution: remove redundant waits/copies and improve layout/overlap without changing TP/Ulysses or mathematical reductions.
3. Operators: equivalent FFN projection fusion, norm/RoPE/modulation and launch overhead, with numeric tests and existing non-CUDA fallbacks.
4. E2E: one complete eight-step candidate trial plus accuracy before claiming repeated A/B performance. Reject regressions; preserve rejected patches/profiles.

Protect sampling/noise/latent/decoder/serving semantics, model assets, metrics and validator. Prefer removing unnecessary work. Do not install dependencies during optimization. Maximum eight hypotheses / eight hours; clean only owned processes at a budget/interrupt boundary. Neither a lower-step profile nor a faster kernel proves a 20% E2E result.

## Phase 4: final five-group AB/BA and accuracy

Freeze candidate code before final runs. Baseline/candidate share winner tuple, environment, caches, model, protocol and guards. Each formal trial has fresh-server independent warmup; alternation controls order bias. Validate **all 15 baseline and 15 candidate images plus each server's first warmup image**, not just the final request, against the canonical reference and against the equivalent baseline output. Save medians/ranges and per-prompt values.

```bash
# Fill the frozen winner values from best.json.
export TP=1 ULYSSES=4
BASELINE_CMD="$PY $LAB/guards-next/qwen21_turbo_b300.py --source-repo $BASE --model-path $MODEL --cache-root $LAB/cache --tp $TP --ulysses $ULYSSES --output {result}"
CANDIDATE_CMD="$PY $LAB/guards-next/qwen21_turbo_b300.py --source-repo $CAND --model-path $MODEL --cache-root $LAB/cache --tp $TP --ulysses $ULYSSES --output {result}"
QUALITY_CMD="$PY $LAB/guards-next/qwen21_turbo_quality.py --reference-manifest $LAB/artifacts/reference-v4-gpu2-5-6-7/result.json --omnirsi-run --expected-parallel $TP,$ULYSSES --output {result}"
"$PY" -m omnirsi run \
  --repo "$BASE" --candidate-repo "$CAND" --repo-type vllm_omni \
  --backend cuda --device-model B300 --device-ids 2,5,6,7 --python "$PY" \
  --scenario diffusion.image_generation --mode code --agent codex \
  --model Qwen/Qwen-Image-2.1-Turbo --model-revision d65dbc9a7e8f6b5479e33dee6030eaab2a906509 \
  --workload-id qwen21-turbo-1280x704-native8-cfg1-seeds42-44 \
  --baseline-command "$BASELINE_CMD" --candidate-command "$CANDIDATE_CMD" \
  --validation-command "$QUALITY_CMD" --metric-key metrics.latency_ms --metric-unit ms \
  --comparison-scope synchronous_png_image_e2e --direction minimize \
  --repetitions 5 --min-improvement-pct 20 --max-minutes 240 \
  --output-dir "$LAB/artifacts/final-runs"
```

Additional unseen prompt/seed images must pass full E2E accuracy and must produce fresh seed-dependent outputs. Test the first request after readiness, especially for conditioning-cache candidates. Describe measured applicability and cold overhead. View image outputs alongside numerical checks; do not label unmeasured perceptual quality as certified.

Deliver raw requests, profile summaries/traces, all quality reports, frozen source/assets/guards/environment, accepted and rejected iterations, candidate patch/DCO commit and the OmniRSI report. Update PR #2 and attach the measured RSI iteration diagram to its comment. Preserve PASS / FAIL / INCONCLUSIVE; if 20% is not achieved, report it directly.

Sources: [pinned vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/f69b1f2b19cece03f6736c7da093ea25de0e24fd/recipes/Qwen/Qwen-Image-2.1.md), [fixed model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/d65dbc9a7e8f6b5479e33dee6030eaab2a906509).
