"""The user configuration surface is argparse, not task files."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from . import __version__
from .catalog import BACKENDS, REPO_TYPES
from .contracts import RunSpec


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=".", help="Prepared target Git checkout")
    parser.add_argument("--repo-type", choices=REPO_TYPES, default="vllm_omni")
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    parser.add_argument("--device-ids", default="", help="Comma-separated caller-assigned device IDs")
    parser.add_argument("--device-model", help="Expected CUDA device model; other probes retain raw output")
    parser.add_argument("--python", default=sys.executable, help="Python used for target package inspection")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="CLI-first reproducible experiments and tuning knowledge")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect a prepared local target without installing dependencies")
    _common(doctor)

    for name in ("plan", "run"):
        child = commands.add_parser(name, help="Freeze an external-command experiment" if name == "plan" else "Measure baseline and candidate commands")
        _common(child)
        child.add_argument("--candidate-repo", help="Candidate checkout; defaults to the baseline checkout")
        child.add_argument("--scenario", default="external_benchmark", help="Caller-declared scenario label")
        child.add_argument("--mode", choices=("config", "code", "semantic"), default="config")
        child.add_argument("--agent", choices=("manual", "codex", "external"), default="manual", help="Handoff label only; no Agent is launched")
        child.add_argument("--executor", choices=("local",), default="local", help="Install this CLI on the remote machine for this first release")
        child.add_argument("--baseline-command", required=True, help="Quoted argv command; {result} is replaced with a fresh output path")
        child.add_argument("--candidate-command", required=True)
        child.add_argument("--validation-command", help="Explicit quality/correctness command; missing evidence is INCONCLUSIVE")
        child.add_argument("--metric-key", default="metrics.latency_ms", help="Flat or dotted key in benchmark JSON")
        child.add_argument("--metric-unit", default="ms")
        child.add_argument("--comparison-scope", default="external_command")
        child.add_argument("--direction", choices=("minimize", "maximize"), default="minimize")
        child.add_argument("--repetitions", type=int, default=3)
        child.add_argument("--min-improvement-pct", type=float, default=0.0)
        child.add_argument("--max-minutes", type=float, default=120.0, help="Experiment deadline, at most seven days")
        child.add_argument("--workload-id", help="Caller-declared fixed workload identity")
        child.add_argument("--model", help="Model identity/path recorded in the experiment")
        child.add_argument("--model-revision")
        child.add_argument("--output-dir", default="runs")
        child.add_argument("--dry-run", action="store_true")

    for name in ("status", "report"):
        child = commands.add_parser(name, help="Read a recorded experiment" if name == "status" else "Regenerate its HTML report")
        child.add_argument("--run-id", required=True)
        child.add_argument("--output-dir", default="runs")

    knowledge = commands.add_parser("knowledge", help="Search or inspect source-grounded imported experience")
    actions = knowledge.add_subparsers(dest="knowledge_action", required=True)
    search = actions.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--repo-type", choices=REPO_TYPES)
    search.add_argument("--backend", choices=BACKENDS)
    search.add_argument("--scenario")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--json", action="store_true")
    show = actions.add_parser("show")
    show.add_argument("id")
    context = commands.add_parser("context", help="Export cited knowledge for manual Codex/Agent handoff")
    context.add_argument("query", nargs="?", default="")
    context.add_argument("--repo-type", choices=REPO_TYPES, required=True)
    context.add_argument("--backend", choices=BACKENDS, required=True)
    context.add_argument("--scenario")
    context.add_argument("--limit", type=int, default=5)
    context.add_argument("--output", help="Write a Markdown context file instead of stdout")
    return root


def device_ids(args: argparse.Namespace) -> tuple[str, ...]:
    values = tuple(value.strip() for value in args.device_ids.split(",") if value.strip())
    if len(set(values)) != len(values):
        raise ValueError("device IDs must be unique")
    return values


def make_spec(args: argparse.Namespace) -> RunSpec:
    def command(value: str | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        parsed = tuple(shlex.split(value, posix=True))
        if not parsed:
            raise ValueError("Commands cannot be empty")
        return parsed

    spec = RunSpec(
        repo=str(Path(args.repo).resolve()),
        candidate_repo=str(Path(args.candidate_repo).resolve()) if args.candidate_repo else None,
        repo_type=args.repo_type, backend=args.backend, scenario=args.scenario, mode=args.mode,
        baseline_command=command(args.baseline_command), candidate_command=command(args.candidate_command),
        validation_command=command(args.validation_command), metric_key=args.metric_key,
        metric_unit=args.metric_unit, comparison_scope=args.comparison_scope,
        direction=args.direction, repetitions=args.repetitions,
        min_improvement_pct=args.min_improvement_pct, max_minutes=args.max_minutes,
        workload_id=args.workload_id, model=args.model, model_revision=args.model_revision,
        agent=args.agent, device_ids=device_ids(args), output_dir=str(Path(args.output_dir).resolve()),
    )
    spec.validate()
    return spec
