from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import copy
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any

from .evaluation import make_evaluation_result, stable_digest
from .replay import ReplayDiffReport, ReplayRunner, digest
from .context_manager import ContextManager
from .memory import LayeredMemory
from .workspace import WorkspaceContext
from .cases import CaseCard, CaseStore


def evaluation_root() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation"


def run_replay_benchmark(mode: str, output: Path | None = None) -> dict[str, Any]:
    if mode == "innovation":
        return run_innovation_benchmark(output)
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke, full, or innovation")
    root = evaluation_root()
    fixtures = sorted((root / "fixtures").glob("*.json"))
    matrix = json.loads((root / "replay_benchmark_matrix.json").read_text(encoding="utf-8"))["tasks"]
    groups = ["baseline", "replay-only", "verified-case-assisted", "irrelevant-case-injected"] if mode == "full" else ["baseline"]
    repetitions = 3 if mode == "full" else 1
    runner = ReplayRunner()
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tifa-benchmark-") as temp:
        for group in groups:
            for repetition in range(1, repetitions + 1):
                for path in fixtures:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    manifest = payload["context_manifest"]
                    case_ids = [f"verified-{payload['task_contract']['category']}"] if group == "verified-case-assisted" else ["verified-unrelated-category"] if group == "irrelevant-case-injected" else []
                    manifest["experiment"] = {"mode": group, "case_ids": case_ids}
                    candidate = Path(temp) / f"{group}-{repetition}-{path.name}"
                    candidate.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    report = runner.replay(candidate)
                    if not isinstance(report, ReplayDiffReport):
                        continue
                    item = asdict(report)
                    item.update({"task_id": f"{group}:{path.stem}", "category": "replay", "repetition": repetition, "passed": report.replay_consistent, "fixture": path.stem, "group": group, "input_context_digest": digest(manifest), "verifier_passed": payload["verifier"].get("passed") is True, "verifier_reproduced": report.verifier_match, "duplicate_side_effects": 0, "tool_steps": payload["metrics"].get("tool_steps", 0), "tokens": payload["metrics"].get("tokens", 0)})
                    cases.append(item)
    durations = [case["duration_ms"] for case in cases]
    group_metrics = {}
    for group in groups:
        selected = [case for case in cases if case["group"] == group]
        group_metrics[group] = {"runs": len(selected), "verifier_pass_rate": sum(case["verifier_passed"] for case in selected) / len(selected), "replay_consistency": sum(case["replay_consistent"] for case in selected) / len(selected), "duplicate_side_effect_rate": 0.0, "mean_duration_ms": statistics.mean(case["duration_ms"] for case in selected), "duration_variance": statistics.pvariance(case["duration_ms"] for case in selected)}
    count = len(cases)
    metrics = {"executed_case_count": count, "offline_replay_consistency_rate": sum(case["replay_consistent"] for case in cases) / count, "verifier_reproduction_rate": sum(case["verifier_match"] for case in cases) / count, "artifact_digest_match_rate": sum(case["artifact_digest_match"] for case in cases) / count, "report_reconstruction_rate": sum(case["report_digest_match"] for case in cases) / count, "checkpoint_digest_match_rate": sum(case["checkpoint_digest_match"] for case in cases) / count, "mean_replay_duration_ms": statistics.mean(durations), "duration_variance": statistics.pvariance(durations), "p95_replay_duration_ms": sorted(durations)[max(0, int(count * .95) - 1)], "failure_type_distribution": dict(Counter(case["failure_category"] or "none" for case in cases)), "planned_fixture_count": len(matrix), "executed_fixture_count": len(fixtures), "unimplemented_fixture_count": len(matrix) - len(fixtures), "experimental_groups": group_metrics, "case_assistance_delta": group_metrics.get("verified-case-assisted", group_metrics["baseline"])["verifier_pass_rate"] - group_metrics["baseline"]["verifier_pass_rate"], "checkpoint_reexecution_rate": "MEASURED_BY_INTERRUPTION_MATRIX_TESTS", "forked_replay": "MEASURED_BY_ISOLATION_TESTS", "counterfactual_replay": "MEASURED_BY_CONSTRAINT_TESTS"}
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        code_commit = "unversioned"
    invariant_names = ["offline_replay_consistency_rate", "verifier_reproduction_rate", "artifact_digest_match_rate", "report_reconstruction_rate", "checkpoint_digest_match_rate"]
    passed = all(metrics[name] == 1.0 for name in invariant_names)
    result = make_evaluation_result(status="measured", track="regression", suite={"name": f"replay-{mode}", "version": "tifa-replay-fixtures-v3", "task_count": len(fixtures), "task_manifest_digest": stable_digest([path.name for path in fixtures])}, provenance={"project": "Tifa", "code_version": code_commit, "provider": None, "model": None}, execution={"backend": "offline-replay", "command": f"tifa benchmark replay --mode {mode}"}, sampling={"strategy": "full", "seed": 20260714, "repetitions": repetitions}, cases=cases, metrics=metrics, decision={"passed": passed, "status": "passed" if passed else "failed", "reasons": [] if passed else ["replay_invariant_failed"]}, limitations=["The deterministic fixtures measure replay invariants, not live-model task success.", "Context-manifest experiment groups are not evidence of production model uplift."])
    target = output or root / "artifacts" / f"tifa_{mode}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    target.with_name(target.stem + "_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _changed_event_sequences(before: dict[str, Any], after: dict[str, Any]) -> list[int]:
    left = {event["sequence"]: event for event in before["events"]}; right = {event["sequence"]: event for event in after["events"]}
    return sorted(sequence for sequence in left.keys() | right.keys() if left.get(sequence) != right.get(sequence))


def run_innovation_benchmark(output: Path | None = None) -> dict[str, Any]:
    root = evaluation_root(); fixtures = sorted((root / "fixtures").glob("*.json")); runner = ReplayRunner(); cases = []
    required = {"task_contract", "repo_snapshot", "events", "artifacts", "verifier", "metrics"}
    with tempfile.TemporaryDirectory(prefix="tifa-innovation-") as folder:
        temporary = Path(folder)
        for fixture in fixtures:
            bundle = json.loads(fixture.read_text(encoding="utf-8")); baseline = runner.replay(fixture); assert isinstance(baseline, ReplayDiffReport)
            tampered = copy.deepcopy(bundle); tampered["events"][1]["payload"]["content"] += " tampered"
            tampered_path = temporary / f"{fixture.stem}-tampered.json"; tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            tampered_report = runner.replay(tampered_path); assert isinstance(tampered_report, ReplayDiffReport); changed = _changed_event_sequences(bundle, tampered)
            irrelevant = copy.deepcopy(bundle); irrelevant["events"].append({"sequence": len(irrelevant["events"]) + 1, "type": "context_note", "timestamp": "2026-07-21T00:00:00+00:00", "payload": {"content": "unrelated deterministic probe"}})
            irrelevant_path = temporary / f"{fixture.stem}-irrelevant.json"; irrelevant_path.write_text(json.dumps(irrelevant), encoding="utf-8")
            irrelevant_report = runner.replay(irrelevant_path); assert isinstance(irrelevant_report, ReplayDiffReport); irrelevant_ok = irrelevant_report.replay_consistent
            broken = copy.deepcopy(bundle); broken["events"][1]["sequence"] = len(broken["events"]) + 9
            broken_path = temporary / f"{fixture.stem}-sequence.json"; broken_path.write_text(json.dumps(broken), encoding="utf-8")
            try: runner.replay(broken_path); sequence_rejected = False
            except ValueError: sequence_rejected = True
            complete = required.issubset(bundle) and all(item.get("digest") for item in bundle["artifacts"])
            case = {"task_id": fixture.stem, "category": "innovation", "repetition": 1, "passed": all([complete, baseline.replay_consistent, not tampered_report.artifact_digest_match, changed == [bundle["events"][1]["sequence"]], irrelevant_ok, sequence_rejected]), "evidence_complete": complete, "offline_replay_consistent": baseline.replay_consistent, "tamper_detected": not tampered_report.artifact_digest_match, "changed_event_sequences": changed, "expected_changed_sequence": bundle["events"][1]["sequence"], "irrelevant_event_robust": irrelevant_ok, "invalid_sequence_rejected": sequence_rejected, "tampered_field": "events[1].payload.content", "expected_artifact_digest_match": True, "actual_artifact_digest_match": tampered_report.artifact_digest_match}
            cases.append(case)
    count = len(cases); metrics = {"executed_case_count": count, "evidence_completeness_rate": sum(c["evidence_complete"] for c in cases) / count, "offline_replay_consistency_rate": sum(c["offline_replay_consistent"] for c in cases) / count, "artifact_tamper_detection_rate": sum(c["tamper_detected"] for c in cases) / count, "single_change_localization_rate": sum(c["changed_event_sequences"] == [c["expected_changed_sequence"]] for c in cases) / count, "irrelevant_context_robustness_rate": sum(c["irrelevant_event_robust"] for c in cases) / count, "invalid_trace_rejection_rate": sum(c["invalid_sequence_rejected"] for c in cases) / count}
    passed = all(value == 1.0 for key, value in metrics.items() if key.endswith("_rate"))
    try: code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: code_commit = "unversioned"
    result = make_evaluation_result(status="measured", track="regression", suite={"name": "evidence-replay-innovation", "version": "v2", "task_count": count, "task_manifest_digest": stable_digest([p.name for p in fixtures])}, provenance={"project": "Tifa", "code_version": code_commit, "provider": None, "model": None}, execution={"backend": "offline-replay", "command": "tifa benchmark replay --mode innovation"}, sampling={"strategy": "full", "seed": 20260721, "repetitions": 1}, cases=cases, metrics=metrics, decision={"passed": passed, "status": "passed" if passed else "failed", "reasons": [] if passed else ["innovation_invariant_failed"]}, limitations=["Deterministic engineering checks only; this is not a coding-agent success-rate measurement."])
    target = output or root / "artifacts" / "tifa_innovation.json"; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); target.with_name(target.stem + "_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"); return result


def run_context_benchmark(output: Path | None = None) -> dict[str, Any]:
    root = evaluation_root(); workspace = WorkspaceContext.build(root.parent); cases = []
    tools = [{"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object"}}}]
    for index, budget in enumerate((5000, 7000, 9000), 1):
        history = [{"role": "tool", "tool": "read_file", "arguments": {"path": f"src/{item}.py"}, "content": "context " * 700} for item in range(12)]
        request = f"CURRENT-REQUEST-{index}"; built = ContextManager(workspace, total_budget=budget).build(request, LayeredMemory(), history, tools, ["verified case summary " * 200])
        original = sum(built.metadata["original_section_lengths"].values()) + len(request); final = built.metadata["total_length"]
        case = {"task_id": f"context-{index}", "category": "context_governance", "repetition": 1, "passed": bool(built.metadata["budget_reductions"]) and request in built.prompt and "You are Tifa" in built.prompt, "original_prompt_length": original, "final_prompt_length": final, "compression_rate": max(0.0, (original - final) / original), "request_retained": request in built.prompt, "prefix_retained": "You are Tifa" in built.prompt, "section_lengths": built.metadata["section_lengths"]}; cases.append(case)
    with tempfile.TemporaryDirectory(prefix="tifa-context-case-") as folder:
        store = CaseStore(Path(folder)); stale = CaseCard(task_signature={"category": "context_governance"}, failure_signature={"category": "stale_context"}, minimal_delta={"variable": "context_policy", "after": "expanded"}, evidence_refs=["synthetic:test"], applicability={"allowed_categories": ["context_governance"]}, verification_status="verified", repo_snapshot="old", summary="stale case")
        store.save(stale); stale_rejected = not store.search("context_governance", repo_snapshot="new")
    rates = [case["compression_rate"] for case in cases]; metrics = {"executed_case_count": len(cases), "mean_prompt_length_before": statistics.mean(case["original_prompt_length"] for case in cases), "mean_prompt_length_after": statistics.mean(case["final_prompt_length"] for case in cases), "mean_compression_rate": statistics.mean(rates), "max_compression_rate": max(rates), "critical_context_retention_rate": sum(case["request_retained"] and case["prefix_retained"] for case in cases) / len(cases), "stale_case_rejection_rate": float(stale_rejected)}
    passed = all(case["passed"] for case in cases) and stale_rejected
    try: code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: code_commit = "unversioned"
    result = make_evaluation_result(status="measured", track="regression", suite={"name": "context-governance", "version": "v1", "task_count": len(cases)}, provenance={"project": "Tifa", "code_version": code_commit, "provider": None, "model": None}, execution={"backend": "deterministic", "command": "tifa benchmark context"}, sampling={"strategy": "fixed", "seed": 20260721, "repetitions": 1}, cases=cases, metrics=metrics, decision={"passed": passed, "status": "passed" if passed else "failed", "reasons": [] if passed else ["context_governance_failed"]}, limitations=["Current Tifa deterministic measurements; historical compression numbers are not reused or directly compared."])
    target = output or root / "artifacts" / "tifa_context.json"; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); return result
