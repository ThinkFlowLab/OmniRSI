# Qwen Turbo B300 phase 2 draft

Continue the first campaign, with a new **30% E2E latency reduction** target against the original source baseline `f69b1f2b19cece03f6736c7da093ea25de0e24fd`. Historical baseline 385.0696 ms implies 269.5487 ms. The denominator is not the H8 candidate. A final fresh five-group AB/BA must corroborate the improvement; report both historical and contemporaneous comparisons.

The model revision, environment, BF16, TP1/Ulysses4, 1280x704, native eight sigmas, CFG1, PNG settings, HTTP response scope, warmups, prompts/seeds and frozen v4 quality thresholds remain as in the first plan. The initial phase 2 cohort used GPUs2/5/6/7. After a foreign GPU5 job interrupted follow-up, a separately qualified cohort uses GPUs2/4/6/7 (also NV18/NUMA0), identically for baseline and candidate. Cohorts remain separate. No quantization, approximate attention/cache, final-image replay, reduced steps or altered decoder arithmetic. First-request and unseen-prompt correctness remain mandatory. Profiling at two steps is diagnostic only.

Implementation begins from H8 candidate `66366303115242f1c2e35a601d37dd7ca4a81033` on `codex-qwen21-turbo-b300-30pct`; artifacts remain outside source. Latest upstream was fetched for inspection, but the experiment source is intentionally pinned to preserve the original baseline. Newly fetched unrelated commits are not performance interventions.

1. Reprofile H8 with the frozen driver before source edits. Inspect stage times, kernel launch/copy chains, all-to-all and host synchronization. Export a task ledger with actual shapes and rounding contracts.
2. Review framework/pipeline: request metadata lifetime, unnecessary tensors/copies, scheduler synchronization and repeated eager dispatch. Avoid caching sampled states or changing serialized output.
3. Extract isolated operators: modulation/residual pointwise chains, RoPE-only layout/casts, QK normalization components, FFN activation and communication layout. Prefer fusion without changing reductions or BF16 rounding boundaries after H3's precision failure.
4. Build representative operator harnesses, compare eager and fused outputs, inspect Nsight Compute metrics, then evaluate each candidate in full eight-step E2E generation. Preserve failed variants.
5. Promote only full-image quality-passing candidates with repeated E2E benefit. Freeze and run final five-group original baseline/candidate AB/BA, first requests and held-out prompts.

Use [KDA](https://github.com/NVlabs/kda/tree/ef6ce617693ef0782b3ecb9f37e39bbf10226a90) as a workflow reference. Its exact KernelWiki and ncu-report submodules are pinned in the companion plan. Sources provide candidate techniques, not local performance evidence. The [MiniMax-H3 Sol Engine reference](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) motivates elementwise/norm/activation fusion; its approximate methods and reported speedups are not transferred to this campaign.
