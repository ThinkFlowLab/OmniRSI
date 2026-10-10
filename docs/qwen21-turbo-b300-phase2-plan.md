# Executable phase 2 plan and KDA links

The [draft](qwen21-turbo-b300-phase2-draft.md) fixes the task contract. The first-stage [plan](qwen21-turbo-b300-test-plan.md) and [results](qwen21-turbo-b300-results.md) remain historical; this extension supersedes only the target and iteration budget.

The first frozen phase 2 cohort (`e873b3d`, GPUs2/5/6/7) passed all40 image checks but achieved 29.8098% against its fresh original-source baseline, so its formal verdict remains FAIL. Follow-up H19 (`330d411`) uses a separately qualified GPUs2/4/6/7 cohort after foreign GPU5 occupancy. Both sides use that same fixed group; preserve the earlier cohort and report historical cross-cohort values separately.

## Linked external skills

Use KDA as reference material in a separate checkout, not an inference dependency. Read these exact resources before use:

- [KDA workflow](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/docs/agent-flow.md), [starter contract](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/prompts/basic-flow.md).
- [KernelWiki SKILL.md](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/SKILL.md): Blackwell kernel fusion/epilogues/launch overhead references. Knowledge cutoff 2026-04-27; verify version-sensitive claims locally.
- [ncu-report SKILL.md](https://github.com/mit-han-lab/ncu-report-skill/blob/d1887948c7d53690cfe6605f59c1329b8a1c6bb5/SKILL.md): representative harness, overview/source reports, metric-based diagnosis. This campaign uses B300/SM103; do not substitute B200 bandwidth/SM counts.

Local reference checkout: `/home/zjy/code/hsliu/kda-qwen21`, with KDA's recorded submodule revisions. Generated reports/harnesses go under the existing campaign's `artifacts/phase2/`; no generated image assets are committed.

## Execution and promotion

1. Run frozen `guards-v4/qwen21_turbo_b300.py` with H8 source, `--tp 1 --ulysses 4 --diagnostic --profile-steps 2`, monitored by `safely_run.py`. Preserve exported four-rank traces even if the known stop-profile API timeout recurs; failed diagnostics do not become E2E results.
2. Parse CPU/kernel/source summaries. Record task, shape, dtype/strides, eager rounding/reduction contract, hypothesis, isolated variable, test/evaluation command and reject conditions before edits.
3. Operator feasibility: exact checks for elementwise fusion and strict numerical checks for any reduction. NCU runs must use a new directory and actual production shapes. Record tools/permissions failures honestly.
4. Full E2E feasibility: unchanged driver, native eight steps, original canonical RGB/alpha gate (SSIM>=0.99, PSNR>=35 dB). Measure first request separately. No benefit or any failed image means revise/reject. A claimed A/B improvement requires at least three groups under fixed controls.
   Run no Nsight or other GPU experiment concurrently, including on other cards in the host. H13's concurrent diagnostic attempt was much slower than its later quiet trial despite no foreign process on selected cards; the discrepancy is retained and no causal claim or promotion is based on that attempt.
5. After collecting accepted changes, freeze source and run five fresh alternating AB/BA groups. Compare both to the 385.0696 ms historical baseline and to freshly measured original source. Require >=30% reduction and all 40 final images passing; additional held-out new prompts/seeds must pass.
6. Run relevant CPU/CUDA checks and pre-commit, commit with DCO, update PR #2 and post the measured iteration graph/table. Report FAIL/INCONCLUSIVE if the target or accuracy gate is not met.

## Historical v4 validation adapter

The phase 2 local cohort reuses the original frozen driver, canonical images and numerical validator. v4's aggregate CLI has the historical run-root bug. Freeze [the directory adapter](../examples/qwen21_turbo_validation_adapter.py) outside source before the cohort and use it as `--validation-command` with `--validator /path/to/guards-v4/qwen21_turbo_quality.py`, the original reference manifest, `--omnirsi-run --expected-parallel 1,4 --output {result}`. It invokes the unchanged validator, then requires ten complete reports containing all40 comparisons. It changes only the environment path used to find the run root. Current helpers resolve the root themselves; a new reproduction should freeze current helpers and generate a new canonical reference as specified in the first-stage plan.

No arbitrary eight-hypothesis cap carries over from the completed first stage. Continue evidence-backed iterations toward the new target; if GPU access or another external condition prevents work, retain actionable evidence and state what remains incomplete.
