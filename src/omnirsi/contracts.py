"""Immutable CLI contracts; no accelerator or Agent runtime dependencies."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    repo: str
    baseline_command: tuple[str, ...]
    candidate_command: tuple[str, ...]
    candidate_repo: str | None = None
    repo_type: str = "vllm"
    backend: str = "cpu"
    scenario: str = "external_benchmark"
    mode: str = "config"
    validation_command: tuple[str, ...] | None = None
    metric_key: str = "metrics.latency_ms"
    metric_unit: str = "ms"
    comparison_scope: str = "external_command"
    direction: str = "minimize"
    repetitions: int = 3
    min_improvement_pct: float = 0.0
    max_minutes: float = 120.0
    workload_id: str | None = None
    model: str | None = None
    model_revision: str | None = None
    agent: str = "manual"
    device_ids: tuple[str, ...] = ()
    output_dir: str = "runs"

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for label, value in (("repo", self.repo), ("candidate_repo", self.candidate_repo or self.repo)):
            if not Path(value).is_dir():
                raise ValueError(f"{label} must be an existing directory: {value}")
        for label, command in (("baseline_command", self.baseline_command),
                               ("candidate_command", self.candidate_command),
                               ("validation_command", self.validation_command)):
            if label == "validation_command" and command is None:
                continue
            if isinstance(command, str) or not command or not all(isinstance(arg, str) for arg in command):
                raise ValueError(f"{label} must be a nonempty argv sequence of strings")
            if not command[0]:
                raise ValueError(f"{label} executable cannot be empty")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be minimize or maximize")
        if self.mode not in {"config", "code", "semantic"}:
            raise ValueError("mode must be config, code or semantic")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int) or self.repetitions < 1:
            raise ValueError("repetitions must be a positive integer")
        for label, value in (("max_minutes", self.max_minutes), ("min_improvement_pct", self.min_improvement_pct)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.max_minutes <= 0 or self.min_improvement_pct < 0:
            raise ValueError("max_minutes must be positive and min_improvement_pct nonnegative")
        if self.max_minutes > 7 * 24 * 60:
            raise ValueError("max_minutes cannot exceed seven days in this release")
        if not self.metric_key or any(not part for part in self.metric_key.split(".")):
            raise ValueError("metric_key must be a nonempty dotted JSON key")
        if not self.metric_unit or not self.comparison_scope:
            raise ValueError("metric_unit and comparison_scope must be declared")
        output_root = Path(self.output_dir).resolve()
        for repo in (self.repo, self.candidate_repo or self.repo):
            source_root = Path(repo).resolve()
            if source_root == output_root or source_root.is_relative_to(output_root):
                raise ValueError("output_dir cannot equal or contain a source repository")
