from __future__ import annotations

import json
from pathlib import Path

import pytest

from tifa.eval_suite import professional_tasks, select_evaluation_backend
from tifa.evaluation import (
    load_evaluation_result,
    make_evaluation_result,
    paired_comparison,
    release_decision,
    stratified_sample,
    summarize_cases,
    validate_evaluation_result,
)


def _result(cases: list[dict], repetitions: int = 3) -> dict:
    metrics = summarize_cases(cases)
    return make_evaluation_result(
        status="measured", track="capability", suite={"name": "test", "version": "v2"},
        provenance={"project": "Tifa", "code_version": "abc", "provider": "openai", "model": "model"},
        execution={"backend": "docker", "runner_image_digest": "sha256:test"},
        sampling={"seed": 1, "repetitions": repetitions}, cases=cases, metrics=metrics,
        decision=release_decision(metrics, repetitions=repetitions, cloud=True), limitations=[])


def test_v2_contract_rejects_missing_duplicate_and_bad_count():
    cases = [{"task_id": "a", "repetition": 1, "category": "tool_recovery", "passed": True}]
    result = _result(cases)
    validate_evaluation_result(result)
    missing = dict(result); missing.pop("provenance")
    with pytest.raises(ValueError, match="missing fields"): validate_evaluation_result(missing)
    duplicate = dict(result); duplicate["cases"] = cases * 2; duplicate["metrics"] = {**result["metrics"], "executed_case_count": 2}
    with pytest.raises(ValueError, match="duplicate"): validate_evaluation_result(duplicate)
    bad_count = dict(result); bad_count["metrics"] = {**result["metrics"], "executed_case_count": 2}
    with pytest.raises(ValueError, match="count"): validate_evaluation_result(bad_count)


def test_smoke_status_is_valid_but_not_release_measured():
    cases = [{"task_id": "a", "repetition": 1, "category": "single_file", "passed": True}]
    result = _result(cases, repetitions=1)
    result["status"] = "smoke"
    validate_evaluation_result(result)
    assert result["decision"]["passed"] is False


def test_stratified_sample_is_reproducible_and_balanced():
    first = stratified_sample(professional_tasks(), 18, 77)
    second = stratified_sample(list(reversed(professional_tasks())), 18, 77)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    counts = {category: sum(task.category == category for task in first) for category in {task.category for task in first}}
    assert set(counts.values()) == {2}


def test_recovery_metrics_are_distinct_and_strict_gate_counts_transport():
    cases = [
        {"task_id": "r", "repetition": 1, "category": "tool_recovery", "passed": True, "repair_feedback_triggered": False, "provider_retries": 0},
        {"task_id": "x", "repetition": 1, "category": "single_file", "passed": False, "failure_category": "transport", "provider_retries": 0},
    ]
    metrics = summarize_cases(cases)
    assert metrics["strict_pass_rate"] == .5 and metrics["infrastructure_adjusted_pass_rate"] == 1.0
    assert metrics["scenario_success_rate"] == 1.0 and metrics["repair_recovery_rate"] is None and metrics["provider_retry_recovery_rate"] is None
    assert "provider_or_schema_failure" in release_decision(metrics, repetitions=3, cloud=False)["reasons"]


def test_legacy_deepseek_artifact_preserves_measured_result():
    path = Path(__file__).parents[1] / "evaluation" / "artifacts" / "measured-live" / "professional-deepseek-v4-flash-30-smoke.json"
    if not path.exists(): pytest.skip("local measured artifact is not checked in")
    loaded = load_evaluation_result(path)
    original = loaded["original"]
    assert loaded["legacy"] is True and original["verifier_pass_rate"] == pytest.approx(19 / 30)
    assert original["failure_distribution"] == {"budget_exceeded": 2, "invalid_arguments": 5, "loop_detected": 1, "step_limit_reached": 2, "transport": 1}


def test_paired_comparison_refuses_different_commits_but_keeps_descriptive_delta():
    left = {"code_version": "a", "task_count": 2, "repetitions": 1, "cases": [{"task_id": "1", "repetition": 1, "passed": False}, {"task_id": "2", "repetition": 1, "passed": True}]}
    right = {"code_version": "b", "task_count": 2, "repetitions": 1, "cases": [{"task_id": "1", "repetition": 1, "passed": True}, {"task_id": "2", "repetition": 1, "passed": True}]}
    result = paired_comparison(left, right, samples=100)
    assert result["status"] == "INVALID_FOR_RELEASE_COMPARISON" and result["right_only"] == 1 and result["mean_pass_rate_delta"] == .5


def test_real_repo_track_is_explicitly_not_run():
    path = Path(__file__).parents[1] / "evaluation" / "real_repo_suite.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "not_run" and payload["tasks"] == []


def test_measured_run_ref_requires_present_matching_artifact(tmp_path: Path):
    case = {"task_id": "a", "repetition": 1, "category": "single_file", "passed": True, "run_ref": {"artifact_dir": "runs/a", "digests": {"report.json": "missing"}}}
    result = _result([case])
    path = tmp_path / "result.json"; path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact missing"):
        load_evaluation_result(path)


def test_evaluation_backend_falls_back_explicitly(monkeypatch):
    monkeypatch.setattr("tifa.eval_suite.DockerExecutionBackend.available", lambda self: False)
    backend, manifest = select_evaluation_backend()
    assert backend.name == "local" and manifest["isolation_level"] == "local_degraded"
