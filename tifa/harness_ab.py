from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .cases import tool_schema_digest
from .contracts import TaskContract
from .eval_suite import EvaluationTask, professional_tasks, select_evaluation_backend
from .evaluation import INFRASTRUCTURE_FAILURES, make_evaluation_result, release_decision, stable_digest, stratified_sample, summarize_cases
from .execution import ExecutionBackend
from .models import AgentResult, ModelClient
from .providers import ProviderError, create_model_client
from .runtime import HarnessControls, RunBudget, build_agent
from .stores import RunStore, now
from .tools import SPECS, ToolArgumentError, ToolRegistry, call_fingerprint
from .verifier import verify_contract
from .workspace import WorkspaceContext


@dataclass(frozen=True)
class HarnessVariant:
    name: str
    runner: str = "full"
    controls: HarnessControls = HarnessControls()

    def treatment_digest(self) -> str:
        return stable_digest(asdict(self))


MINIMAL = HarnessVariant("minimal", "minimal", HarnessControls(False, False, False, False))
FULL = HarnessVariant("full")
STANDARD_ABLATIONS = (
    HarnessVariant("ablation_no_immediate_verifier", controls=HarnessControls(immediate_verifier=False)),
    HarnessVariant("ablation_no_structured_recovery_path", controls=HarnessControls(structured_recovery=False)),
    HarnessVariant("ablation_no_repeat_side_effect_governance", controls=HarnessControls(side_effect_governance=False)),
    HarnessVariant("ablation_no_context_memory_checkpoint", controls=HarnessControls(context_memory_checkpoint=False)),
)
DEFAULT_BUDGET = RunBudget(max_model_calls=14, max_tool_calls=10, max_duration_seconds=300)


def _failure_category(exc: Exception) -> str:
    if isinstance(exc, ToolArgumentError): return "schema_error"
    if "path" in str(exc).lower(): return "path_error"
    if isinstance(exc, PermissionError): return "approval_denied"
    if isinstance(exc, (ValueError, KeyError, TypeError)): return "invalid_arguments"
    if isinstance(exc, TimeoutError): return "tool_timeout"
    return "environment_error"


class MinimalFunctionCallingRunner:
    """Safety-constrained function-calling substrate without Tifa treatments."""

    def __init__(self, root: Path, model_client: ModelClient, backend: ExecutionBackend, budget: RunBudget = DEFAULT_BUDGET) -> None:
        self.workspace = WorkspaceContext.build(root)
        self.workspace.normalize_paths = False
        self.model_client = model_client
        self.backend = backend
        self.budget = budget

    def run(self, contract: TaskContract) -> AgentResult:
        store = RunStore(self.workspace.repo_root)
        registry = ToolRegistry(
            self.workspace, "never", execution_backend=self.backend,
            allowed_tools=set(contract.allowed_tools), writable_paths=contract.writable_paths,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": contract.goal}]
        affected_paths: list[str] = []
        events: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        security_events: list[dict[str, Any]] = []
        usage_rows: list[dict[str, Any]] = []
        fingerprints: list[str] = []
        usage = {"model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        attempts = 0
        stop_reason = "retry_limit_reached"
        failure_category: str | None = None
        answer = ""
        started = time.perf_counter()
        store.append_trace("run_started", {"request": contract.goal, "variant": "minimal", "contract_id": contract.contract_id})
        while attempts < self.budget.max_model_calls and usage["tool_calls"] < self.budget.max_tool_calls:
            if time.perf_counter() - started > self.budget.max_duration_seconds:
                stop_reason, failure_category = "budget_exceeded", "budget_exceeded"; break
            attempts += 1
            try:
                response = self.model_client.complete(contract.goal, registry.schemas(), None, messages)
            except ProviderError as exc:
                stop_reason, failure_category, answer = "provider_error", exc.category, f"Minimal runner stopped: {exc.category}."
                store.append_trace("provider_error", {"category": exc.category, "attempt": attempts, "request_meta": getattr(self.model_client, "last_request_meta", {})})
                break
            usage["model_calls"] += 1
            usage["input_tokens"] += int(response.usage.get("input_tokens", response.usage.get("prompt_tokens", response.usage.get("input_estimate", 0))))
            usage["output_tokens"] += int(response.usage.get("output_tokens", response.usage.get("completion_tokens", response.usage.get("output_estimate", 0))))
            usage["cost_usd"] += float(response.usage.get("cost_usd", 0.0)); usage_rows.append(response.usage)
            store.append_trace("model_response", {"attempt": attempts, "text": response.text, "tool_calls": [asdict(call) for call in response.tool_calls], "usage": response.usage, "cache": response.cache})
            if not response.tool_calls:
                answer, stop_reason = response.text, "final_answer_returned"
                break
            remaining = self.budget.max_tool_calls - usage["tool_calls"]
            executable, truncated = response.tool_calls[:remaining], response.tool_calls[remaining:]
            messages.append({"role": "assistant", "content": response.text, "tool_calls": [asdict(call) for call in response.tool_calls]})
            for call in executable:
                fingerprint = call_fingerprint(call.name, call.arguments)
                try:
                    registry.last_execution = None
                    output, affected = registry.run(call.name, call.arguments)
                    ok = True
                    if registry.last_execution:
                        executions.append(asdict(registry.last_execution))
                        security_events.extend({"type": event, "tool": call.name} for event in registry.last_execution.security_events)
                except Exception as exc:
                    output, affected, ok = f"error: {_failure_category(exc)}: {exc}", [], False
                    failure_category = _failure_category(exc)
                usage["tool_calls"] += 1
                affected_paths = sorted(set(affected_paths + affected)); fingerprints.extend(fingerprint for _ in affected if affected)
                event = {"call_id": call.id, "name": call.name, "arguments": call.arguments, "output": output, "ok": ok, "affected_paths": affected, "fingerprint": fingerprint}
                events.append(event); store.append_trace("tool_commit", event)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            for call in truncated:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": "error: tool call skipped because no execution budget remained"})
                store.append_trace("tool_batch_truncated", {"call_id": call.id, "name": call.name})
            if truncated or usage["tool_calls"] >= self.budget.max_tool_calls:
                stop_reason, failure_category = "step_limit_reached", failure_category or "step_limit_reached"
                break
        verification = verify_contract(self.workspace.repo_root, contract.verifier, self.backend, affected_paths)
        if verification.get("passed") is True and stop_reason in {"step_limit_reached", "retry_limit_reached"}:
            stop_reason, failure_category = "verified_at_limit", None
        elif verification.get("passed") is True:
            failure_category = None
        report = {"run_id": store.run_id, "answer": answer, "stop_reason": stop_reason, "failure_category": failure_category or verification.get("failure_category"), "tool_steps": usage["tool_calls"], "attempts": attempts, "affected_paths": affected_paths, "verifier": verification, "variant": "minimal"}
        store.write("report.json", "tifa-report.v3", report)
        store.write("task_state.json", "tifa-task-state.v3", {"run_id": store.run_id, "status": "finished", **report})
        trace = [json.loads(line) for line in (store.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        evidence = {
            "run_id": store.run_id, "created_at": now(), "task_contract": asdict(contract), "events": trace,
            "checkpoints": [], "verifier": verification,
            "repo_snapshot": {"workspace_digest": self.workspace.fingerprint(), "tool_schema_digest": tool_schema_digest(registry.schemas())},
            "context_manifest": {"policy": "disabled_minimal_baseline", "memory_enabled": False, "selected_items": [], "dropped_items": []},
            "metrics": {"provider_usage": usage_rows, "budget_usage": usage, "execution_events": executions, "security_events": security_events, "harness": {"variant": "minimal", "verified_after_tool": 0, "tool_batch_truncated": len(truncated) if 'truncated' in locals() else 0, "normalized_paths": self.workspace.normalized_path_count}},
            "provenance": {"provider": self.model_client.provider, "model": self.model_client.model, "temperature": getattr(self.model_client, "temperature", None)},
        }
        store.write("evidence_bundle.json", "evidence-bundle.v3", evidence)
        store.write("metrics.json", "tifa-run-metrics.v1", {"run_id": store.run_id, "stop_reason": stop_reason, "failure_category": report["failure_category"], "budget_usage": usage, "duplicate_side_effects": len(fingerprints) - len(set(fingerprints))})
        store.append_trace("report", report); store.append_trace("verifier", verification)
        return AgentResult(answer, store.run_id, "minimal", stop_reason, usage["tool_calls"], attempts, str(store.run_dir))


def _write_fixture(root: Path, task: EvaluationTask) -> None:
    for relative, content in task.files.items():
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")


def _persist_run(result: AgentResult, destination: Path, relative_to: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in ("report.json", "evidence_bundle.json", "trace.jsonl", "metrics.json", "task_state.json"):
        source = Path(result.run_dir) / name
        if source.is_file():
            target = destination / name; shutil.copy2(source, target); digests[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"run_id": result.run_id, "artifact_dir": destination.relative_to(relative_to).as_posix(), "digests": digests}


def _case_from_run(task: EvaluationTask, repetition: int, variant: HarnessVariant, result: AgentResult, duration_ms: float, run_ref: dict[str, Any]) -> dict[str, Any]:
    report = json.loads((Path(result.run_dir) / "report.json").read_text(encoding="utf-8"))
    evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    commits = [event["payload"] for event in evidence.get("events", []) if event.get("type") == "tool_commit"]
    effects = [commit["fingerprint"] for commit in commits if commit.get("affected_paths")]
    budget = evidence["metrics"]["budget_usage"]; harness = evidence["metrics"].get("harness", {})
    security_events = evidence["metrics"].get("security_events", [])
    violation_markers = ("denied", "violation", "escape", "repeated_tool_failure", "duplicate_tool_call_id")
    security_violations = sum(any(marker in str(event.get("type", "")).lower() for marker in violation_markers) for event in security_events)
    return {
        "task_id": task.task_id, "category": task.category, "repetition": repetition, "variant": variant.name,
        "passed": report["verifier"]["passed"], "failure_category": report.get("failure_category"), "stop_reason": result.stop_reason,
        "duration_ms": duration_ms, "model_calls": int(budget.get("model_calls", 0)), "tool_calls": int(budget.get("tool_calls", 0)),
        "input_tokens": int(budget.get("input_tokens", 0)), "output_tokens": int(budget.get("output_tokens", 0)),
        "duplicate_side_effects": len(effects) - len(set(effects)), "repair_feedback_triggered": any(event.get("type") == "repair_feedback" for event in evidence.get("events", [])),
        "verified_after_tool": bool(harness.get("verified_after_tool")), "security_event_count": len(security_events), "security_violation_count": security_violations,
        "initial_snapshot_digest": stable_digest(task.files), "run_ref": run_ref,
    }


def _exact_mcnemar(full_only: int, minimal_only: int) -> float:
    total = full_only + minimal_only
    if total == 0: return 1.0
    tail = sum(math.comb(total, index) for index in range(0, min(full_only, minimal_only) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def paired_variant_order(task_id: str, repetition: int, seed: int) -> list[HarnessVariant]:
    variants = [MINIMAL, FULL]
    random.Random(f"{seed}:{task_id}:{repetition}").shuffle(variants)
    return variants


def _efficiency(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("input_tokens", "output_tokens", "duration_ms", "model_calls", "tool_calls"):
        values = sorted(float(case.get(metric, 0)) for case in cases)
        result[metric] = {
            "mean": statistics.mean(values) if values else None,
            "p50": statistics.median(values) if values else None,
            "p95": values[max(0, int(len(values) * .95) - 1)] if values else None,
        }
    return result


def causal_paired_comparison(minimal: dict[str, Any], full: dict[str, Any], *, seed: int = 20260721, samples: int = 5000) -> dict[str, Any]:
    left = {(case["task_id"], case["repetition"]): case for case in minimal.get("cases", [])}
    right = {(case["task_id"], case["repetition"]): case for case in full.get("cases", [])}
    invariant_matches = bool(minimal.get("invariant_config_digest")) and minimal.get("invariant_config_digest") == full.get("invariant_config_digest")
    identities_match = set(left) == set(right)
    snapshots_match = identities_match and all(left[key].get("initial_snapshot_digest") == right[key].get("initial_snapshot_digest") for key in left)
    comparable = invariant_matches and identities_match and snapshots_match
    mismatch_reasons = []
    if not invariant_matches: mismatch_reasons.append("invariant_config_mismatch")
    if not identities_match: mismatch_reasons.append("paired_case_identity_mismatch")
    if not snapshots_match: mismatch_reasons.append("initial_snapshot_mismatch")
    keys = sorted(set(left) & set(right)); excluded = [key for key in keys if left[key].get("failure_category") in INFRASTRUCTURE_FAILURES or right[key].get("failure_category") in INFRASTRUCTURE_FAILURES]
    evaluable = [key for key in keys if key not in excluded]
    both = sum(left[key].get("passed") is True and right[key].get("passed") is True for key in evaluable)
    full_only = sum(left[key].get("passed") is not True and right[key].get("passed") is True for key in evaluable)
    minimal_only = sum(left[key].get("passed") is True and right[key].get("passed") is not True for key in evaluable)
    neither = len(evaluable) - both - full_only - minimal_only
    clusters: dict[str, list[float]] = {}
    for key in evaluable: clusters.setdefault(key[0], []).append(float(right[key].get("passed") is True) - float(left[key].get("passed") is True))
    cluster_values = [sum(values) / len(values) for values in clusters.values()]
    rng = random.Random(seed); boot = []
    if cluster_values:
        for _ in range(samples): boot.append(sum(rng.choice(cluster_values) for _ in cluster_values) / len(cluster_values))
        boot.sort()
    delta = (full_only - minimal_only) / len(evaluable) if evaluable else None
    return {
        "schema_version": "tifa-harness-comparison.v1", "comparable": comparable,
        "status": "measured" if comparable else "INVALID_FOR_CAUSAL_COMPARISON", "mismatch_reasons": mismatch_reasons, "paired_cases": len(keys), "evaluable_pairs": len(evaluable),
        "excluded_infrastructure_pairs": [{"task_id": task, "repetition": repetition} for task, repetition in excluded],
        "both_passed": both, "full_only": full_only, "minimal_only": minimal_only, "both_failed": neither,
        "minimal_pass_rate": (both + minimal_only) / len(evaluable) if evaluable else None,
        "full_pass_rate": (both + full_only) / len(evaluable) if evaluable else None,
        "full_minus_minimal": delta, "cluster_bootstrap_ci95": [boot[int(.025 * len(boot))], boot[min(len(boot) - 1, int(.975 * len(boot)))]] if boot else None,
        "mcnemar_exact_p": _exact_mcnemar(full_only, minimal_only),
        "safety": {"minimal_duplicate_side_effects": sum(c.get("duplicate_side_effects", 0) for c in left.values()), "full_duplicate_side_effects": sum(c.get("duplicate_side_effects", 0) for c in right.values()), "minimal_security_violations": sum(c.get("security_violation_count", 0) for c in left.values()), "full_security_violations": sum(c.get("security_violation_count", 0) for c in right.values()), "minimal_security_events_observed": sum(c.get("security_event_count", 0) for c in left.values()), "full_security_events_observed": sum(c.get("security_event_count", 0) for c in right.values())},
        "efficiency": {"minimal": _efficiency(list(left.values())), "full": _efficiency(list(right.values()))},
    }


def _variant_result(variant: HarnessVariant, cases: list[dict[str, Any]], invariant: dict[str, Any], task_manifest_digest: str, backend_manifest: dict[str, Any], repetitions: int, seed: int, code_version: str, dirty: bool | None) -> dict[str, Any]:
    metrics = summarize_cases(cases)
    durations = sorted(float(case["duration_ms"]) for case in cases)
    metrics.update({"duration_mean_ms": statistics.mean(durations) if durations else None, "duration_p50_ms": statistics.median(durations) if durations else None, "duration_p95_ms": durations[max(0, int(len(durations) * .95) - 1)] if durations else None, "mean_model_calls": statistics.mean(case["model_calls"] for case in cases) if cases else None, "mean_tool_calls": statistics.mean(case["tool_calls"] for case in cases) if cases else None})
    status = "smoke" if len({case["task_id"] for case in cases}) < 100 else "measured" if repetitions >= 3 and backend_manifest["isolation_level"] == "container_strong" else "incomplete"
    result = make_evaluation_result(
        status=status, track="harness_ab", suite={"name": "professional", "version": "v2", "task_count": len({case["task_id"] for case in cases}), "task_manifest_digest": task_manifest_digest},
        provenance={"project": "Tifa", "code_version": code_version, "dirty_worktree": dirty, "provider": invariant["provider"], "model": invariant["model"], "variant": variant.name, "treatment_config_digest": variant.treatment_digest()},
        execution={**backend_manifest, "temperature": 0, "budget": invariant["budget"], "invariant_config_digest": stable_digest(invariant)},
        sampling={"seed": seed, "repetitions": repetitions, "selected_task_ids": sorted({case["task_id"] for case in cases})}, cases=cases, metrics=metrics,
        decision=release_decision(metrics, repetitions=repetitions, cloud=True), limitations=["Synthetic tasks measure harness behavior, not real-repository success.", "Infrastructure-failed pairs are excluded only from causal capability deltas and remain reported.", "local_degraded evidence is not release-grade." if backend_manifest["isolation_level"] == "local_degraded" else "container_strong verifier execution."],
    )
    result["invariant_config_digest"] = stable_digest(invariant); result["treatment_config_digest"] = variant.treatment_digest()
    return result


def evaluate_harness_ab(provider: str, model: str, output: Path, *, case_count: int = 30, repetitions: int = 1, seed: int = 20260721, ablations: str = "none") -> dict[str, Any]:
    if case_count not in {1, 30, 100}: raise ValueError("cases must be 1, 30, or 100")
    if repetitions not in {1, 3}: raise ValueError("repetitions must be 1 or 3")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY is required for harness A/B evaluation")
    output.mkdir(parents=True, exist_ok=True)
    tasks = stratified_sample(professional_tasks(), case_count, seed)
    backend, backend_manifest = select_evaluation_backend()
    budget_dict = {"model_calls": DEFAULT_BUDGET.max_model_calls, "tool_calls": DEFAULT_BUDGET.max_tool_calls, "duration_seconds": DEFAULT_BUDGET.max_duration_seconds}
    task_manifest = [{"task_id": task.task_id, "category": task.category, "prompt": task.prompt, "files": task.files, "contract": asdict(task.contract)} for task in tasks]
    task_digest = stable_digest(task_manifest)
    schema_digest = tool_schema_digest([SPECS[name] for name in tasks[0].contract.allowed_tools])
    invariant = {"provider": provider, "model": model, "temperature": 0, "budget": budget_dict, "task_manifest_digest": task_digest, "tool_schema_digest": schema_digest, "backend": backend_manifest}
    try: code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception: code_version = "unknown"
    try: dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL, timeout=3).strip())
    except Exception: dirty = None
    variants = [MINIMAL, FULL]
    if ablations == "standard": variants.extend(STANDARD_ABLATIONS)
    ablation_task_ids = {task.task_id for task in stratified_sample(professional_tasks(), min(case_count, 30), seed)}
    cases_by_variant: dict[str, list[dict[str, Any]]] = {variant.name: [] for variant in variants}
    order_manifest: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tifa-harness-ab-") as folder:
        base = Path(folder)
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                pair = paired_variant_order(task.task_id, repetition, seed)
                scheduled = pair + (variants[2:] if repetition == 1 and task.task_id in ablation_task_ids else [])
                order_manifest.append({"task_id": task.task_id, "repetition": repetition, "order": [variant.name for variant in scheduled]})
                for variant in scheduled:
                    root = base / variant.name / f"{task.task_id}-r{repetition}"; root.mkdir(parents=True); _write_fixture(root, task)
                    started = time.perf_counter(); client = create_model_client(provider, model)
                    if variant.runner == "minimal": result = MinimalFunctionCallingRunner(root, client, backend).run(task.contract)
                    else: result = build_agent(root, client, approval_policy="never", max_steps=10, max_attempts=14, execution_backend=backend, run_budget=DEFAULT_BUDGET, harness_controls=variant.controls).ask(task.prompt, contract=task.contract)
                    duration_ms = (time.perf_counter() - started) * 1000
                    destination = output / f"{variant.name}.runs" / f"{task.task_id}-r{repetition}-{result.run_id}"
                    run_ref = _persist_run(result, destination, output)
                    cases_by_variant[variant.name].append(_case_from_run(task, repetition, variant, result, duration_ms, run_ref))
    results: dict[str, dict[str, Any]] = {}
    invariants_by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_repetitions = 1 if variant.name.startswith("ablation_") else repetitions
        variant_manifest = [item for item in task_manifest if not variant.name.startswith("ablation_") or item["task_id"] in ablation_task_ids]
        variant_task_digest = stable_digest(variant_manifest)
        variant_invariant = {**invariant, "task_manifest_digest": variant_task_digest}
        invariants_by_variant[variant.name] = variant_invariant
        result = _variant_result(variant, cases_by_variant[variant.name], variant_invariant, variant_task_digest, backend_manifest, variant_repetitions, seed, code_version, dirty)
        results[variant.name] = result
        (output / f"{variant.name}.result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / f"{variant.name}.cases.json").write_text(json.dumps(result["cases"], ensure_ascii=False, indent=2), encoding="utf-8")
        (output / f"{variant.name}.manifest.json").write_text(json.dumps({"suite": result["suite"], "provenance": result["provenance"], "execution": result["execution"], "sampling": result["sampling"], "order": order_manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / f"{variant.name}.decision.json").write_text(json.dumps(result["decision"], ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = causal_paired_comparison(results["minimal"], results["full"], seed=seed)
    comparison["invariant_config_digest"] = stable_digest(invariant)
    evidence_status = "smoke" if case_count < 100 or repetitions < 3 or backend_manifest["isolation_level"] != "container_strong" else "measured"
    comparison["status"] = evidence_status if comparison["comparable"] else comparison["status"]
    ablation_results = {}
    for name, result in results.items():
        if not name.startswith("ablation_"): continue
        full_subset = {**results["full"], "invariant_config_digest": stable_digest(invariants_by_variant[name]), "cases": [case for case in results["full"]["cases"] if case["task_id"] in ablation_task_ids and case["repetition"] == 1]}
        ablation_results[name] = {"metrics": result["metrics"], "treatment_config_digest": result["treatment_config_digest"], "full_minus_ablation": causal_paired_comparison(result, full_subset, seed=seed)}
    (output / "ablation-results.json").write_text(json.dumps(ablation_results, ensure_ascii=False, indent=2), encoding="utf-8")
    ci = comparison.get("cluster_bootstrap_ci95")
    infrastructure_ok = results["minimal"]["metrics"].get("infrastructure_failure_rate", 0) <= .02 and results["full"]["metrics"].get("infrastructure_failure_rate", 0) <= .02
    safety_ok = comparison["safety"]["full_duplicate_side_effects"] <= comparison["safety"]["minimal_duplicate_side_effects"] and comparison["safety"]["full_security_violations"] <= comparison["safety"]["minimal_security_violations"]
    comparison["claim_gate"] = {"infrastructure_failure_at_most_2pct": infrastructure_ok, "no_duplicate_or_security_regression": safety_ok, "positive_delta": bool(comparison.get("full_minus_minimal") is not None and comparison["full_minus_minimal"] > 0), "ci_lower_above_zero": bool(ci and ci[0] > 0)}
    (output / "paired-comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    conclusion = "INVALID_FOR_CAUSAL_COMPARISON" if not comparison["comparable"] else "statistically_supported_harness_gain" if all(comparison["claim_gate"].values()) else "directional_but_statistically_uncertain" if comparison.get("full_minus_minimal", 0) > 0 and infrastructure_ok and safety_ok else "claim_gate_failed" if comparison.get("full_minus_minimal", 0) > 0 else "no_observed_harness_gain"
    report = {"schema_version": "tifa-harness-ab-report.v1", "status": evidence_status, "conclusion": conclusion, "provenance": {"code_version": code_version, "dirty_worktree": dirty, "provider": provider, "model": model, "invariant_config_digest": stable_digest(invariant), "backend": backend_manifest}, "comparison": comparison, "variants": {name: result["metrics"] for name, result in results.items()}, "ablations": ablation_results, "order": order_manifest, "limitations": ["Historical artifacts are descriptive only and are not used as the minimal baseline.", "This result is not release-grade when status is smoke or backend is local_degraded."]}
    (output / "harness-ab-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
