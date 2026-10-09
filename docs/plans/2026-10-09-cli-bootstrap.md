# CLI-first RSI bootstrap implementation plan

**Goal:** Deliver an installable first OmniRSI package, a reproducible external-command experiment path, and source-grounded diffusion knowledge with a remote validation guide.

**Architecture:** CLI arguments form immutable run contracts. A local runner supervises explicit baseline/candidate/validation commands and records JSON evidence; an independent evaluator and HTML report preserve incomplete outcomes. Repository/backend metadata and imported experience live in package data so installed wheels have the same knowledge as a checkout.

**Tech stack:** Python 3.10+, standard library, setuptools; unittest for CPU-only tests. No mandatory torch, GPU runtime, YAML parser, database server or agent service.

## Scope and assumptions

- The reviewed plan remains the target architecture; this PR implements its first measurable slice.
- The user will install/run the CLI on their prepared remote accelerator machine. The first executor is local to that machine; control-host SSH orchestration is later work.
- The caller supplies actual benchmark/validation commands. No autonomous source modification, model installation or production release occurs.
- `--backend cpu` exists for synthetic contract/smoke tests; it makes no accelerator claim.
- CUDA/ROCm/Ascend inspection and repository metadata do not establish a supported model/function matrix.
- Imported upstream skills and PR reports are marked imported/upstream-reported, never locally validated.
- User input remains CLI args. JSON files are run snapshots, benchmark output and internal knowledge records.

## Tasks and verification

1. Package and contracts: create pyproject, CLI arguments and core records; verify import without torch and package data in an installed wheel.
2. Execution/evaluation: run isolated output paths with explicit commands and a global deadline; capture output, versions and evidence; verify failure, timeout, cancellation, missing/invalid metrics and quality gaps.
3. CLI/inspection: doctor, plan/run dry-run, run, status, report and knowledge search/show; verify meaningful CPU-only CLI end-to-end and argument errors.
4. Knowledge: paraphrase both requested skill sources and selected MiniMax-H3 PRs; pin sources, constraints and remote checks; verify catalog/schema/filtering and source identity.
5. Documentation: README, Quickstart, architecture and remote test plan; exercise every CPU example, and mark accelerator tests unexecuted.
6. Review/delivery: independent code self-review, signed-off commit, issue-linked PR with How to use and Test plan; attach the PR to this chat.

## Ownership

- Runtime contributor: contracts, execution, evaluation, report and their tests.
- Knowledge contributor: package-data experience entries and source documentation.
- Primary contributor: CLI, inspection, catalog, packaging, integration tests, docs and GitHub delivery.

## Completion criteria

- A real executable package, not empty adapters or fake phase success.
- The synthetic local experiment passes only with explicit successful validation.
- Missing evidence produces INCONCLUSIVE; command/validation failures are preserved.
- Knowledge search works in a clean installed package and filters repo/backend scope.
- PR documents the exact implemented commands and the remote GPU/NPU checks still required.

All commits use `git commit -s`. No merge or production activation is included.
