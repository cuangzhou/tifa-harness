from __future__ import annotations

import json
from pathlib import Path

import pytest

from tifa.benchmark import run_context_benchmark, run_innovation_benchmark
from tifa.cases import CaseCard, CaseStore, suggest_override
from tifa.replay import ReplayRunner
from tifa import FakeModelClient, TaskContract, build_agent
from tifa.cases import tool_schema_digest
from tifa.tools import ToolRegistry
from tifa.workspace import WorkspaceContext


def test_innovation_profile_measures_all_six_invariants(tmp_path: Path):
    result = run_innovation_benchmark(tmp_path / "innovation.json")
    expected = {"evidence_completeness_rate", "offline_replay_consistency_rate", "artifact_tamper_detection_rate", "single_change_localization_rate", "irrelevant_context_robustness_rate", "invalid_trace_rejection_rate"}
    assert expected <= result["metrics"].keys() and all(result["metrics"][key] == 1.0 for key in expected)
    assert len(result["cases"]) == 24 and all(case["changed_event_sequences"] == [case["expected_changed_sequence"]] for case in result["cases"])


def test_context_benchmark_is_new_measurement_and_rejects_stale_case(tmp_path: Path):
    result = run_context_benchmark(tmp_path / "context.json")
    assert result["decision"]["passed"] is True and result["metrics"]["stale_case_rejection_rate"] == 1.0
    assert "historical" in result["limitations"][0].lower()


def test_case_v1_migrates_and_freshness_reasons_are_explicit(tmp_path: Path):
    store = CaseStore(tmp_path)
    legacy = {"schema_version": "case-card.v1", "case_id": "legacy", "task_signature": {"category": "bug"}, "failure_signature": {"category": "loop_detected"}, "minimal_delta": {"variable": "memory_enabled", "after": False}, "evidence_refs": ["run"], "applicability": {"allowed_categories": ["bug"]}, "verification_status": "verified", "repo_snapshot": "old"}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    card = store.load("legacy"); fresh, reasons = store.assess_freshness(card, repo_snapshot="new")
    assert card.schema_version == "case-card.v2" and card.review_status == "legacy_review_required"
    assert fresh is False and reasons == ["repo_snapshot_changed"]


@pytest.mark.parametrize(("category", "variable"), [("transport", "provider"), ("irrelevant_context", "context_policy"), ("loop_detected", "memory_enabled"), ("unknown", None)])
def test_candidate_override_mapping(category: str, variable: str | None):
    override, review = suggest_override(category)
    assert override.get("variable") == variable
    assert review == ("needs_review" if variable is None else "ready")


def test_promote_requires_recorded_clean_single_variable_replay(tmp_path: Path):
    store = CaseStore(tmp_path); card = CaseCard(task_signature={"category": "bug"}, failure_signature={"category": "loop_detected"}, minimal_delta={"variable": "memory_enabled", "after": False}, evidence_refs=["run"], applicability={"allowed_categories": ["bug"]}, freshness_status="fresh")
    store.save(card)
    replay = {"spec": {"overrides": {"memory_enabled": False}}, "report": {"confounded": False, "source_unchanged": True}, "replay_bundle": {"verifier": {"passed": True}}, "same_task_contract": True, "same_snapshot": True, "replay_run_id": "verified-run"}
    store.record_verification(card.case_id, replay); promoted = store.promote(store.load(card.case_id))
    assert promoted.verification_status == "verified" and promoted.verification_run_id == "verified-run"
    bad = CaseCard(task_signature={"category": "other"}, failure_signature={"category": "loop_detected"}, minimal_delta={"variable": "memory_enabled", "after": False}, evidence_refs=["run"], applicability={"allowed_categories": ["other"]}, freshness_status="fresh")
    store.save(bad); replay["report"]["confounded"] = True; store.record_verification(bad.case_id, replay)
    with pytest.raises(ValueError, match="verification gate"): store.promote(store.load(bad.case_id))


def test_structured_diff_redacts_content_and_localizes_event():
    original = {"events": [{"sequence": 1, "type": "tool_commit", "payload": {"name": "write_file", "arguments": {"content": "secret"}, "affected_paths": ["a.py"]}}], "context_manifest": {}, "artifacts": [], "verifier": {}, "metrics": {}, "checkpoints": []}
    replay = json.loads(json.dumps(original)); replay["events"][0]["payload"]["arguments"]["content"] = "changed"
    diff = ReplayRunner.diff(original, replay)
    assert diff["events"]["changed_sequences"] == [1]
    assert diff["events"]["changes"][0]["original"]["payload"]["arguments"]["content"]["redacted"] is True


def test_failed_run_automatically_creates_one_candidate(tmp_path: Path):
    contract = TaskContract("must create result", ["write_file"], ["result.txt"], {"files": [{"path": "result.txt"}]}, max_repairs=0)
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask(contract.goal, contract=contract)
    store = CaseStore(tmp_path / ".tifa" / "cases"); cards = store.list()
    assert result.stop_reason == "completion_gate_failed" and len(cards) == 1 and cards[0].source_run_id == result.run_id
    assert store.propose_from_run(tmp_path, result.run_id).case_id == cards[0].case_id


def test_verified_fresh_case_is_opt_in_and_recorded_in_context_manifest(tmp_path: Path):
    context = WorkspaceContext.build(tmp_path); registry = ToolRegistry(context, approval_policy="never", delegate=lambda *_: "")
    store = CaseStore(tmp_path / ".tifa" / "cases")
    card = CaseCard(task_signature={"category": "bug"}, failure_signature={"category": "loop_detected"}, minimal_delta={"variable": "memory_enabled", "after": False}, evidence_refs=["evidence:verified"], applicability={"allowed_categories": ["bug"]}, summary="Use the verified minimal repair.", verification_status="verified", freshness_status="fresh", repo_snapshot=context.fingerprint(), tool_schema_digest=tool_schema_digest(registry.schemas()), context_policy_version="layered-budget-v2")
    store.save(card)
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask("done", case_store=store, case_category="bug")
    bundle = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    assert bundle["context_manifest"]["selected_items"][0]["selected_case_ids"] == [card.case_id]
