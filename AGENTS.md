# Development rules

- Follow plan -> code -> test -> review and keep changes focused.
- The public configuration interface is CLI arguments. JSON records are evidence and internal package data.
- Keep the control package importable without torch, inference engines or accelerator runtimes.
- Preserve PASS / FAIL / INCONCLUSIVE; missing evidence must not become PASS.
- Imported knowledge must keep sources, exact revisions, applicability and verification scope. Upstream reports are not local validation.
- Run `python -m unittest discover -s tests -v` for the CPU suite. Document remote accelerator checks that were not run.
- All Git commits must use `git commit -s`.
- Do not merge, publish packages or activate production systems as part of an ordinary code change.
