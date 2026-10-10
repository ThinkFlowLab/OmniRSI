"""Evidence gates, independent from command execution and Agent assertions."""

import math

from .contracts import RunSpec


def read_metric(result: dict, key: str) -> float:
    value = result
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing metric: {key}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {key} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"metric {key} must be finite and positive")
    return float(value)


def _summary(trials: list[dict]) -> dict | None:
    values = [trial["metric"] for trial in trials if trial.get("metric") is not None]
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    middle = ordered[midpoint] if len(ordered) % 2 else ordered[midpoint - 1] + (ordered[midpoint] - ordered[midpoint - 1]) / 2
    spread = (max(values) - min(values)) / middle * 100
    return {"values": values, "median": middle, "min": min(values), "max": max(values),
            "range_pct_of_median": spread if math.isfinite(spread) else None}


def evaluate(spec: RunSpec, baseline: list[dict], candidate: list[dict],
             validation: dict | None, integrity: dict | None = None) -> dict:
    failed, incomplete = [], []
    for label, trials in (("baseline", baseline), ("candidate", candidate)):
        if len(trials) != spec.repetitions:
            incomplete.append(f"{label}: expected {spec.repetitions} trials, recorded {len(trials)}")
        for index, trial in enumerate(trials, 1):
            if trial.get("status") in {"failed", "timeout", "error"}:
                failed.append(f"{label} trial {index}: {trial.get('status')}")
            elif trial.get("status") != "completed":
                incomplete.append(f"{label} trial {index}: {trial.get('status')}")
            if trial.get("metric") is None:
                incomplete.append(f"{label} trial {index}: {trial.get('metric_error') or 'metric missing'}")
    if validation is None:
        incomplete.append("quality validation command was not supplied")
    elif validation.get("status") in {"failed", "timeout", "error"}:
        failed.append(f"quality validation: {validation.get('status')}")
    elif validation.get("status") != "completed":
        incomplete.append(f"quality validation: {validation.get('status')}")
    if integrity is None or not integrity.get("verified"):
        incomplete.append((integrity or {}).get("reason", "source integrity was not verified"))

    before, after = _summary(baseline), _summary(candidate)
    improvement = None
    all_metrics = (len(baseline) == len(candidate) == spec.repetitions and
                   all(t.get("status") == "completed" and t.get("metric") is not None
                       for t in baseline + candidate))
    if all_metrics:
        improvement = (after["median"] - before["median"]) / before["median"] * 100
        if spec.direction == "minimize":
            improvement *= -1
        if not math.isfinite(improvement):
            improvement = None
            incomplete.append("derived improvement is not finite; check metric scale and units")
        elif improvement <= 0 or improvement < spec.min_improvement_pct:
            failed.append(f"measured gain {improvement:.6g}% must be positive and meet "
                          f"{spec.min_improvement_pct:.6g}% threshold")
    verdict = "FAIL" if failed else "INCONCLUSIVE" if incomplete else "PASS"
    return {"verdict": verdict, "reasons": failed + incomplete,
            "metric_key": spec.metric_key, "metric_unit": spec.metric_unit,
            "comparison_scope": spec.comparison_scope, "direction": spec.direction,
            "min_improvement_pct": spec.min_improvement_pct,
            "baseline": before, "candidate": after, "improvement_pct": improvement,
            "measurement_method": "Alternating AB/BA sequential trials; median and range are descriptive only.",
            "quality_contract": "Exit zero covers only the supplied validation command; inspect its logs and coverage.",
            "source_integrity": integrity,
            "statistical_significance": "not established"}
