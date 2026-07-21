from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable


SCHEMA_VERSION = "tifa-evaluation-result.v2"
VALID_STATUSES = {"measured", "smoke", "incomplete", "placeholder", "aborted", "not_run", "invalid"}
INFRASTRUCTURE_FAILURES = {"transport", "timeout", "rate_limit", "auth", "invalid_response", "provider_schema"}
FAILURE_TAXONOMY = {
    "transport": "infrastructure", "timeout": "infrastructure", "rate_limit": "infrastructure", "auth": "provider", "invalid_response": "tool_schema", "provider_schema": "tool_schema",
    "invalid_arguments": "argument_generation", "loop_detected": "planning_loop", "step_limit_reached": "budget", "budget_exceeded": "budget",
    "policy_denied": "policy", "execution_failed": "execution", "verifier_failed": "verifier",
}


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return [max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)]


def stratified_sample(tasks: Iterable[Any], limit: int | None, seed: int) -> list[Any]:
    items = list(tasks)
    if limit is None or limit >= len(items):
        return items
    if limit <= 0:
        raise ValueError("limit must be positive")
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        category = getattr(item, "category", None)
        if category is None:
            category = item["category"]
        groups[str(category)].append(item)
    rng = random.Random(seed)
    for name in sorted(groups):
        group = groups[name]
        group.sort(key=lambda value: str(getattr(value, "task_id", value.get("task_id") if isinstance(value, dict) else value)))
        rng.shuffle(group)
    names = sorted(groups)
    selected: list[Any] = []
    cursor = 0
    while len(selected) < limit:
        name = names[cursor % len(names)]
        if groups[name]:
            selected.append(groups[name].pop())
        cursor += 1
    return selected


def _rates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(case.get("passed") is True for case in cases)
    infra = sum(case.get("failure_category") in INFRASTRUCTURE_FAILURES for case in cases)
    adjusted_total = len(cases) - infra
    recovery = [case for case in cases if case.get("category") == "tool_recovery"]
    repair = [case for case in cases if case.get("repair_feedback_triggered")]
    provider_retry = [case for case in cases if int(case.get("provider_retries", 0)) > 0]
    tool_calls = sum(int(case.get("tool_calls", case.get("tool_call_count", 0))) for case in cases)
    duplicates = sum(int(case.get("duplicate_side_effects", 0)) for case in cases)
    return {
        "strict_pass_rate": passed / len(cases) if cases else None,
        "strict_pass_rate_ci95": wilson_interval(passed, len(cases)),
        "infrastructure_adjusted_pass_rate": passed / adjusted_total if adjusted_total else None,
        "infrastructure_failure_rate": infra / len(cases) if cases else None,
        "duplicate_side_effect_rate": duplicates / max(1, tool_calls),
        "scenario_success_rate": sum(c.get("passed") is True for c in recovery) / len(recovery) if recovery else None,
        "repair_recovery_rate": sum(c.get("passed") is True for c in repair) / len(repair) if repair else None,
        "provider_retry_recovery_rate": sum(c.get("passed") is True for c in provider_retry) / len(provider_retry) if provider_retry else None,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rates = _rates(cases)
    categories: dict[str, Any] = {}
    for category in sorted({str(case.get("category", "unknown")) for case in cases}):
        selected = [case for case in cases if str(case.get("category", "unknown")) == category]
        category_passed = sum(case.get("passed") is True for case in selected)
        categories[category] = {
            "executed": len(selected), "passed": category_passed,
            "pass_rate": category_passed / len(selected),
            "ci95": wilson_interval(category_passed, len(selected)),
        }
    rates.update({
        "executed_case_count": len(cases),
        "failure_distribution": dict(Counter(str(c.get("failure_category")) for c in cases if c.get("passed") is not True)),
        "failure_taxonomy_distribution": dict(Counter(FAILURE_TAXONOMY.get(str(c.get("failure_category")), "unknown") for c in cases if c.get("passed") is not True)),
        "category_results": categories,
        "total_input_tokens": sum(int(c.get("input_tokens", 0)) for c in cases),
        "total_output_tokens": sum(int(c.get("output_tokens", 0)) for c in cases),
    })
    return rates


def release_decision(metrics: dict[str, Any], *, repetitions: int, cloud: bool, baseline_rate: float | None = None, require_comparable: bool = False, comparable: bool = True) -> dict[str, Any]:
    threshold = 0.8 if cloud else 0.5
    reasons = []
    if repetitions < 3:
        reasons.append("fewer_than_three_repetitions")
    if require_comparable and not comparable:
        reasons.append("INVALID_FOR_RELEASE_COMPARISON")
    if (metrics.get("strict_pass_rate") or 0.0) < threshold:
        reasons.append("pass_rate_below_threshold")
    if (metrics.get("duplicate_side_effect_rate") or 0.0) != 0:
        reasons.append("duplicate_side_effects_detected")
    if (metrics.get("infrastructure_failure_rate") or 0.0) != 0:
        reasons.append("provider_or_schema_failure")
    scenario = metrics.get("scenario_success_rate")
    if scenario is not None and scenario < 0.9:
        reasons.append("recovery_scenario_below_threshold")
    if baseline_rate is not None and (metrics.get("strict_pass_rate") or 0.0) < baseline_rate - 0.05:
        reasons.append("baseline_regression_over_5pp")
    return {"passed": not reasons, "status": "passed" if not reasons else "failed", "reasons": reasons, "thresholds": {"pass_rate": threshold, "max_regression": 0.05, "recovery": 0.9, "duplicate_side_effect_rate": 0.0, "infrastructure_failure_rate": 0.0}}


def make_evaluation_result(*, status: str, track: str, suite: dict[str, Any], provenance: dict[str, Any], execution: dict[str, Any], sampling: dict[str, Any], cases: list[dict[str, Any]], metrics: dict[str, Any], decision: dict[str, Any], limitations: list[str]) -> dict[str, Any]:
    result = {"schema_version": SCHEMA_VERSION, "status": status, "track": track, "suite": suite, "provenance": provenance, "execution": execution, "sampling": sampling, "cases": cases, "metrics": metrics, "decision": decision, "limitations": limitations, "generated_at": datetime.now(timezone.utc).isoformat()}
    result["manifest_digest"] = stable_digest({"track": track, "suite": suite, "provenance": provenance, "execution": execution, "sampling": sampling})
    validate_evaluation_result(result)
    return result


def validate_evaluation_result(result: dict[str, Any]) -> None:
    required = {"schema_version", "status", "track", "suite", "provenance", "execution", "sampling", "cases", "metrics", "decision", "limitations", "generated_at", "manifest_digest"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"evaluation result missing fields: {sorted(missing)}")
    if result["schema_version"] != SCHEMA_VERSION or result["status"] not in VALID_STATUSES:
        raise ValueError("invalid evaluation schema version or status")
    identities = [(case.get("task_id"), case.get("repetition")) for case in result["cases"]]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate evaluation case identity")
    if result["status"] in {"measured", "smoke", "incomplete"} and result["metrics"].get("executed_case_count") != len(result["cases"]):
        raise ValueError("executed case count does not match cases")


def load_evaluation_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == SCHEMA_VERSION:
        validate_evaluation_result(payload)
        if payload.get("status") in {"measured", "smoke", "incomplete"}:
            for case in payload.get("cases", []):
                ref = case.get("run_ref", {})
                artifact_dir = ref.get("artifact_dir")
                if not artifact_dir: continue
                directory = (path.parent / artifact_dir).resolve()
                try: directory.relative_to(path.parent.resolve())
                except ValueError as exc: raise ValueError("run_ref escapes evaluation artifact directory") from exc
                for name, expected in ref.get("digests", {}).items():
                    artifact = directory / name
                    if not artifact.is_file(): raise ValueError(f"measured run artifact missing: {case.get('task_id')}/{name}")
                    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    if actual != expected: raise ValueError(f"measured run artifact digest mismatch: {case.get('task_id')}/{name}")
        return payload
    if payload.get("schema_version") in {"tifa-provider-eval.v1", "tifa-live-eval.v1"}:
        return {"schema_version": payload["schema_version"], "status": "measured", "legacy": True, "source": str(path), "original": payload}
    raise ValueError("unsupported evaluation result schema")


def paired_comparison(left: dict[str, Any], right: dict[str, Any], seed: int = 20260721, samples: int = 5000) -> dict[str, Any]:
    def comparison_config(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") == SCHEMA_VERSION:
            return {"code_version": payload["provenance"].get("code_version"), "task_ids": payload["sampling"].get("selected_task_ids"), "repetitions": payload["sampling"].get("repetitions"), "execution": payload.get("execution")}
        return {"code_version": payload.get("code_version"), "task_ids": [case.get("task_id") for case in payload.get("cases", [])], "repetitions": payload.get("repetitions"), "execution": payload.get("environment")}
    comparable = comparison_config(left) == comparison_config(right)
    lcases = {(c["task_id"], c["repetition"]): bool(c.get("passed")) for c in left.get("cases", [])}
    rcases = {(c["task_id"], c["repetition"]): bool(c.get("passed")) for c in right.get("cases", [])}
    keys = sorted(lcases.keys() & rcases.keys())
    counts = Counter((lcases[key], rcases[key]) for key in keys)
    deltas = [int(rcases[key]) - int(lcases[key]) for key in keys]
    rng = random.Random(seed); boot = []
    if deltas:
        for _ in range(samples):
            boot.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
        boot.sort()
    return {"comparable": comparable, "status": "measured" if comparable else "INVALID_FOR_RELEASE_COMPARISON", "paired_cases": len(keys), "both_passed": counts[(True, True)], "right_only": counts[(False, True)], "left_only": counts[(True, False)], "both_failed": counts[(False, False)], "mean_pass_rate_delta": sum(deltas) / len(deltas) if deltas else None, "bootstrap_ci95": [boot[int(.025 * len(boot))], boot[min(len(boot) - 1, int(.975 * len(boot)))]] if boot else None}
