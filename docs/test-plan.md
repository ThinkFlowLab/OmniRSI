# Test plan

## B300 × 4 / LTX-2.5 remote Codex campaign

Follow the [complete CLI Test plan](ltx25-b300-test-plan.md) for pinned source/model preparation, canonical A/V reference, conditional nine-cell TP/Ulysses/VAE search, external Codex invocation and five-group AB/BA acceptance. The goal is at least 30% lower E2E latency relative to a freshly remeasured quality-passing best parallel configuration. This authoring session has no B300 inference evidence.

The included `examples/ltx25_b300.py` and `examples/ltx25_quality.py` preserve complete media, source/protocol/guard hashes and all trial quality evidence. CPU tests cover command/protocol contracts, drift, incomplete audio/video and owned Linux process cleanup; they do not certify LTX-2.5 parallel configurations.

## Local / CI contracts

```bash
python -m pip install .
python -m unittest discover -s tests -v
python tests/check_installed.py
python -m pip wheel --no-deps . -w dist
```

The suite uses real subprocesses and temporary signed-off Git fixtures. It checks fresh result paths, AB/BA order, quality gaps, invalid/nonfinite metrics, no-gain/regression thresholds, source mutations (including restored changes), command failure, deadlines, cancellation, owned descendants and report escaping. CLI/catalog tests check filtering, source attribution, context applicability, dry-run and status/report. No model inference is part of these tests.

Install FFmpeg/ffprobe to run the real CPU media integration check. Linux process-group tests require Linux; platform-specific skips are reported explicitly.

## Remote checks to run yourself

### 1. Installation and source identity

- Checkout the PR branch and install in a clean Python environment.
- Run `omnirsi --help`, `--version`, CPU smoke and installed-package knowledge searches.
- Use Git-root paths for baseline/candidate. Verify each commit and imported engine module path.
- Preserve dirty patches and untracked source separately, or commit them with `-s`. Fingerprints alone do not reconstruct source.

### 2. Accelerator inspection

```bash
omnirsi doctor --repo /work/vllm-omni --repo-type vllm_omni --backend cuda --device-ids 0,1 --python /envs/omni/bin/python
omnirsi doctor --repo /work/vllm-omni --repo-type vllm_omni --backend rocm --python /envs/omni/bin/python
omnirsi doctor --repo /work/vllm-omni --repo-type vllm_omni --backend ascend --python /envs/omni/bin/python
```

Run only the command for your hardware. Inspect raw output, actual device visibility, firmware/driver/runtime, communication stack and topology. CUDA raw IDs may differ from logical visible IDs; ROCm/Ascend selection is not certified by this bootstrap. `--device-ids` is descriptive metadata and does not isolate processes or set backend visibility.

### 3. Real baseline/candidate pair

- Follow Quickstart with your existing working engine and frozen workload.
- Start with the same configuration on both sides: no gain must fail the positive-gain gate.
- Next change one factor and use at least three alternating paired repetitions; compare raw values and variance, not just the median.
- Check every workload request completed. Failures/omissions cannot become speedups.
- Verify the quality script actually evaluates the affected model/task, seed/shape/steps, output modality, metrics and acceptance threshold.
- With no validation command, require INCONCLUSIVE. With a known bad quality result, require FAIL.
- An open upstream PR must be checked out deliberately at its recorded head; it is not assumed available in main.

### 4. Lifecycle and process cleanup on remote Linux

- Run a controlled long-lived benchmark, then Ctrl-C. Expect cancelled evidence and no owned server/child processes.
- Run it with a short `--max-minutes`; expect timeout/FAIL and no owned descendants.
- Check your model wrapper does not daemonize or move processes out of the runner's process group. Such independent services require caller-managed cleanup.
- During a controlled run, change a tracked source file. Expect INCONCLUSIVE source integrity.
- Verify missing/malformed JSON preserves logs and never returns PASS.
- Do not use device-global kill commands for this test; inspect only the trial's recorded PIDs/processes.

### 5. Knowledge-specific qualification

- Select only records matching your repo/backend, then manually check SKU, model/task, revision, topology and load conditions.
- Q/K and modulation fusion: compare outputs, rounding behavior, shape/dtype guards and non-CUDA fallback.
- AdaLN and reference KV: confirm request-local state, cache clearing/ownership, cancellation and memory footprint; retain slower results.
- VAE tiling: compare generated outputs and memory peaks as well as latency.
- TeaCache and MXFP8: explicitly qualify quality and loading/runtime memory; do not label approximate optimization lossless.
- Ascend SP head buckets: add whole-model quality and repeated E2E tests; raw video hash differences are not proof of quality equivalence.
- This import contains no concrete MiniMax-H3 ROCm performance result. Do not transfer a CUDA/Ascend gain to ROCm without measurement.

## Acceptance and evidence to return

Return the PR SHA, baseline/candidate commits, environment/visibility, model/asset/workload revisions, exact argv, raw result cohort, successful-request counts, validation output and `report.html`. Distinguish the measured scope from production capacity, SLO tails and long-stability claims.

Windows CPU tests, static checks and packaging are local verification. Actual CUDA/ROCm/Ascend, MiniMax-H3 quality/performance, remote Linux cleanup and production stability are **not validated by this PR authoring session**.
