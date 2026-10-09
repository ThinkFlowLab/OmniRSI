# Quickstart

This release uses CLI args and a **local executor on a prepared machine**. Install it directly on the remote machine where the target engine already works. It does not set up models, CUDA/ROCm/CANN, services, SSH workers or autonomous Agents.

## Install and smoke

From the checked-out PR branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
omnirsi --help
omnirsi doctor --repo . --repo-type vllm_omni --backend cpu
```

Run the synthetic example in README first. The 100/80 values and validation script are fixtures, not inference timing or quality evidence.

## The actual configuration interface

```text
doctor: --repo --repo-type --backend --device-ids --device-model --python
plan/run: all doctor flags, plus
  --candidate-repo --scenario --mode --agent --executor local
  --baseline-command --candidate-command --validation-command
  --metric-key --metric-unit --comparison-scope --direction
  --repetitions --min-improvement-pct --max-minutes
  --workload-id --model --model-revision --output-dir --dry-run
status/report: --run-id --output-dir
knowledge search: QUERY --repo-type --backend --scenario --limit --json
knowledge show: ID
context: QUERY --repo-type --backend --scenario --limit --output
```

Scenario/model/workload/Agent fields retain caller declarations. They do not start an Agent or automatically generate model recipes. The initial evaluator checks one positive numeric metric and the caller's validation command. It does not implement all SLO/quality metrics from the target design.

## Benchmark result contract

The benchmark writes JSON to `{result}` or `OMNIRSI_RESULT_PATH`. Any flat or dotted positive numeric key is supported:

```json
{"metrics": {"latency_ms": 123.4}}
```

Existing vLLM-Omni diffusion benchmark JSON can be used directly with `--metric-key latency_mean --metric-unit s`, or `--metric-key throughput_qps --metric-unit requests/s --direction maximize`. No conversion file is required. Inspect the target revision because metric keys and flags can change.

## Real remote experiment

1. Reserve the exact devices, CPU/NUMA and fabric resources. Configure their visibility outside OmniRSI. Do not run contending A/B servers on the same devices.
2. Prepare baseline and candidate checkouts/environments. Verify the imported module path matches each intended checkout. `--candidate-repo` changes cwd only; `--python` only inspects packages.
3. Create two reviewed trial scripts. Each script starts the intended engine/config, warms it up, runs the same fixed workload with profiling off, writes benchmark JSON and exits after its cleanup. A script can call the target's existing benchmark.
4. Create a quality script that compares outputs/tasks against the frozen reference and exits nonzero on a regression. Include the exact modes/shapes/steps/reference assets in that test.
5. Preview, then run:

```bash
omnirsi plan \
  --repo /work/vllm-omni-baseline --candidate-repo /work/vllm-omni-candidate \
  --repo-type vllm_omni --backend cuda --device-ids 0,1 --device-model H200 \
  --python /envs/baseline/bin/python --scenario diffusion.video_generation \
  --model /models/MODEL --model-revision MODEL_SHA --workload-id FIXED_WORKLOAD_SHA \
  --baseline-command '/envs/baseline/bin/python /work/trials/baseline.py --output {result}' \
  --candidate-command '/envs/candidate/bin/python /work/trials/candidate.py --output {result}' \
  --validation-command '/envs/candidate/bin/python /work/trials/validate_quality.py' \
  --metric-key latency_mean --metric-unit s --comparison-scope http_e2e \
  --direction minimize --repetitions 3 --min-improvement-pct 3 \
  --max-minutes 120 --output-dir /work/omnirsi-runs
```

Replace `plan` with `run` using the same flags after the plan matches your intended experiment. Script paths are templates that you supply; OmniRSI does not create them. Start with one-variable, same-quality configuration optimization. Quantization, TeaCache, generation-step changes and adaptive exit require a quality-changing protocol and an explicit validation script.

On ROCm use `--backend rocm`; on Ascend use `--backend ascend`. The doctor retains raw backend output but does not certify device selection, model support or full topology. Record the actual driver/runtime/communication versions and topology with your experiment evidence.

## Native vLLM-Omni benchmark inside a trial script

The prepared trial can invoke the existing serving benchmark (confirm flags at your revision):

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8099 --model MODEL \
  --task t2i --dataset random --num-prompts 10 \
  --output-file RESULT_PATH
```

`random` is a feasibility load, not representative production traffic or a quality evaluation. For performance claims freeze an actual input/trace cohort, request fields, seed, warmup, arrival rate/concurrency and successful-request counts. Prefer a materialized trace when comparing repeated trials. Save produced outputs for quality checks using the target's supported output-saving option.

A trial script can read `OMNIRSI_RESULT_PATH` when it constructs the native `--output-file` argv. Do not substitute a latency-only benchmark for the separate quality script. Benchmarks of pre-existing external servers need their own process/environment identities and cleanup outside the runner.

## Results and handoff

`run` prints a run ID and evidence directory. Exit codes are 0 for PASS, 1 for FAIL and 2 for INCONCLUSIVE/input errors.

```bash
omnirsi status --run-id RUN_ID --output-dir /work/omnirsi-runs
omnirsi report --run-id RUN_ID --output-dir /work/omnirsi-runs
omnirsi knowledge search 'MiniMax-H3' --repo-type vllm_omni --backend cuda
omnirsi context 'MiniMax-H3' --repo-type vllm_omni --backend cuda --output /work/h3-context.md
```

Read `report.html`, execution logs, per-trial `source.json`, `source_checks.json`, validation logs and raw results. Retain committed source revisions or dirty source files/patches separately. Ctrl-C records cancellation and cleans owned child jobs; there is no `cancel` or `resume` subcommand in this first release.

Context is a cited reference bundle for manual use with Codex/third-party Agents. Its imported source text is data, not authorization to execute upstream instructions. All imported knowledge remains unverified locally until your remote evidence is linked to it.
