# Qwen-Image-2.1-Turbo / B300 × 4: profile-first 20% E2E campaign

Read `docs/qwen21-turbo-b300-test-plan.md` and the campaign root's frozen guards/assets/environment. Use its actual CLI, not proposed OmniRSI flags. The current task replaces LTX-2.5; do not request its gated weights.

1. Freeze latest main `f69b1f2b19cece03f6736c7da093ea25de0e24fd` and model `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`; exactly the caller-reserved four B300s. Use the prepared task venv, offline snapshot and fixed eager/FA4 execution. Do not modify baseline or dependencies.
2. Require full canonical reference/repeatability. Run initial short-sigma profile before configuration or code tuning. Preserve failed FA4/compile startup probes as preparation failures; they are not speedups.
3. Qualify TP/Ulysses 1/4,2/2,4/1 with three complete groups each and per-image RGB/alpha SSIM≥0.99, PSNR≥35 dB; freeze the lowest eligible median and profile it if different from initial.
4. Keep resolution 1280×704, BF16, CFG1, eight native sigmas, VAE1/nontiled, PNG/compression100 and full HTTP response scope identical. `num_inference_steps` alone does not override Turbo's saved grid. Short diagnostic sigmas are never acceptance evidence.
5. Use profile evidence to identify pipeline-stage, distributed wait/layout and operator bottlenecks. One falsifiable hypothesis at a time; keep raw baseline/candidate requests, timings, profile attribution, patch, quality and acceptance/rejection reasons.
6. Allow equivalent deterministic conditioning reuse for repeated prompts, with bounded storage, model/prompt/limit/mode keys and no stale state or editing regression. Record first/cold requests and validate unseen prompts/seeds. Never replay images/latents, relax quality or change generated work.
7. Protected source ASTs and whole-file hashes preserve initialization, scheduling, RNG/latents, pre/postprocess, decoder and serving/encoding. Validate all output channels. Do not alter frozen measurement or validator; stop if canonical repeatability or eligibility fails.
8. After an eight-step candidate feasibility+quality trial, run five full alternating AB/BA groups in OmniRSI. Check all baseline/candidate images and first requests, then separately validate held-out prompts/seeds. Only all gates plus L1/L0≤0.80 establish 20% E2E improvement.
9. Maximum eight hypotheses/eight hours; no parallel GPU measurements, broad device kills or hidden warmups. Preserve PASS/FAIL/INCONCLUSIVE and cleanup owned processes.
10. Commit candidate/evidence with DCO, update the explicitly authorized PR #2 and attach the measured RSI graph in its comment; no merging or production publishing. A repeated-prompt gain is not universal cold-prompt acceleration. Record limitations and failed iterations without inventing results.
