from __future__ import annotations

import json
from pathlib import Path
import re

from tifa import FakeModelClient, ModelResponse, TaskContract, ToolCall
from tifa.execution import LocalExecutionBackend
from tifa.harness_ab import MinimalFunctionCallingRunner, causal_paired_comparison, evaluate_harness_ab, paired_variant_order
from tifa.evaluation import load_evaluation_result
from tifa.runtime import HarnessControls, RunBudget, build_agent


def _contract(max_steps: int = 3) -> TaskContract:
    return TaskContract(
        "write READY to result.txt",
        ["read_file", "write_file"],
        ["result.txt"],
        {"files": [{"path": "result.txt", "equals": "READY"}]},
        max_steps=max_steps,
        max_repairs=0,
    )


def test_minimal_runner_scores_only_at_end_and_has_no_harness_state(tmp_path: Path):
    client = FakeModelClient([
        ModelResponse(tool_calls=[ToolCall("ready", "write_file", {"path": "result.txt", "content": "READY"})]),
        ModelResponse(tool_calls=[ToolCall("break", "write_file", {"path": "result.txt", "content": "BROKEN"})]),
        "done",
    ])
    result = MinimalFunctionCallingRunner(tmp_path, client, LocalExecutionBackend()).run(_contract())
    evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    assert result.stop_reason == "final_answer_returned"
    assert evidence["verifier"]["passed"] is False
    assert evidence["checkpoints"] == []
    assert evidence["context_manifest"]["policy"] == "disabled_minimal_baseline"
    assert not any(message["role"] == "user" for message in client.calls[1]["messages"][1:])


def test_minimal_runner_keeps_strict_paths_and_atomic_budget(tmp_path: Path):
    batch = ModelResponse(tool_calls=[
        ToolCall("bad", "write_file", {"path": "/workspace/result.txt", "content": "READY"}),
        ToolCall("extra", "write_file", {"path": "result.txt", "content": "READY"}),
    ])
    result = MinimalFunctionCallingRunner(tmp_path, FakeModelClient([batch]), LocalExecutionBackend(), RunBudget(max_model_calls=2, max_tool_calls=1)).run(_contract(max_steps=1))
    evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    commits = [event for event in evidence["events"] if event["type"] == "tool_commit"]
    assert len(commits) == 1 and commits[0]["payload"]["ok"] is False
    assert not (tmp_path / "result.txt").exists()
    assert evidence["metrics"]["budget_usage"]["tool_calls"] == 1


def test_full_ablation_disables_immediate_verifier(tmp_path: Path):
    client = FakeModelClient([
        ModelResponse(tool_calls=[ToolCall("ready", "write_file", {"path": "result.txt", "content": "READY"})]),
        ModelResponse(tool_calls=[ToolCall("break", "write_file", {"path": "result.txt", "content": "BROKEN"})]),
        "<final>done</final>",
    ])
    result = build_agent(tmp_path, client, approval_policy="never", harness_controls=HarnessControls(immediate_verifier=False)).ask("ignored", contract=_contract())
    report = json.loads((Path(result.run_dir) / "report.json").read_text(encoding="utf-8"))
    assert result.stop_reason == "completion_gate_failed"
    assert report["verifier"]["passed"] is False


def test_context_memory_checkpoint_ablation_records_no_checkpoints(tmp_path: Path):
    client = FakeModelClient([ModelResponse(tool_calls=[ToolCall("ready", "write_file", {"path": "result.txt", "content": "READY"})])])
    result = build_agent(tmp_path, client, approval_policy="never", harness_controls=HarnessControls(context_memory_checkpoint=False)).ask("ignored", contract=_contract())
    evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    assert result.stop_reason == "verified_after_tool"
    assert evidence["checkpoints"] == []
    assert evidence["context_manifest"]["selected_items"][0]["policy"] == "disabled_ablation"


def test_recovery_and_side_effect_ablations_do_not_add_tifa_interventions(tmp_path: Path):
    repeated = [ModelResponse(tool_calls=[ToolCall(f"w{index}", "write_file", {"path": "result.txt", "content": "READY"})]) for index in range(3)]
    controls = HarnessControls(immediate_verifier=False, structured_recovery=False, side_effect_governance=False)
    result = build_agent(tmp_path, FakeModelClient([*repeated, "<final>done</final>"]), approval_policy="never", harness_controls=controls).ask("ignored", contract=_contract())
    evidence = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    commits = [event for event in evidence["events"] if event["type"] == "tool_commit"]
    assert len(commits) == 3 and all(event["payload"]["output"].startswith("wrote") for event in commits)
    assert not any(event["type"] in {"loop_detected", "repair_feedback"} for event in evidence["events"])


def _comparison_result(digest: str, cases: list[dict]) -> dict:
    return {"invariant_config_digest": digest, "cases": cases}


def test_causal_comparison_counts_pairs_and_exact_mcnemar():
    minimal = _comparison_result("same", [
        {"task_id": "a", "repetition": 1, "passed": False, "initial_snapshot_digest": "a"},
        {"task_id": "b", "repetition": 1, "passed": True, "initial_snapshot_digest": "b"},
        {"task_id": "c", "repetition": 1, "passed": False, "initial_snapshot_digest": "c"},
    ])
    full = _comparison_result("same", [
        {"task_id": "a", "repetition": 1, "passed": True, "initial_snapshot_digest": "a"},
        {"task_id": "b", "repetition": 1, "passed": True, "initial_snapshot_digest": "b"},
        {"task_id": "c", "repetition": 1, "passed": False, "initial_snapshot_digest": "c"},
    ])
    result = causal_paired_comparison(minimal, full, samples=100)
    assert result["comparable"] is True
    assert (result["both_passed"], result["full_only"], result["minimal_only"], result["both_failed"]) == (1, 1, 0, 1)
    assert result["full_minus_minimal"] == 1 / 3 and result["mcnemar_exact_p"] == 1.0


def test_causal_comparison_rejects_invariant_or_snapshot_mismatch():
    case = {"task_id": "a", "repetition": 1, "passed": True, "initial_snapshot_digest": "one"}
    other = {**case, "initial_snapshot_digest": "two"}
    result = causal_paired_comparison(_comparison_result("left", [case]), _comparison_result("right", [other]), samples=10)
    assert result["status"] == "INVALID_FOR_CAUSAL_COMPARISON"
    assert "invariant_config_mismatch" in result["mismatch_reasons"] and "initial_snapshot_mismatch" in result["mismatch_reasons"]


def test_pair_order_is_seeded_per_identity_not_iteration_order():
    first = [variant.name for variant in paired_variant_order("task-a", 1, 77)]
    assert first == [variant.name for variant in paired_variant_order("task-a", 1, 77)]
    orders = {tuple(variant.name for variant in paired_variant_order(f"task-{index}", 1, 77)) for index in range(10)}
    assert orders == {("minimal", "full"), ("full", "minimal")}


def test_harness_ab_writes_paired_persistent_artifacts(tmp_path: Path, monkeypatch):
    class CompletingClient:
        provider = "openai-compatible"
        model = "deepseek-v4-flash"
        temperature = 0

        def complete(self, prompt, tools, cache_key=None, messages=None):
            marker = re.search(r"TIFA-[A-Za-z0-9_-]+-OK", prompt).group(0)
            if not any(message.get("role") == "tool" for message in messages or []):
                return ModelResponse(tool_calls=[ToolCall("write", "write_file", {"path": "result.txt", "content": marker})])
            return ModelResponse("done")

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("tifa.harness_ab.create_model_client", lambda *_: CompletingClient())
    output = tmp_path / "ab"
    report = evaluate_harness_ab("openai", "deepseek-v4-flash", output, case_count=1, repetitions=1)
    assert report["comparison"]["comparable"] is True
    assert [item["order"] for item in report["order"]] == [[variant.name for variant in paired_variant_order(report["order"][0]["task_id"], 1, 20260721)]]
    for variant in ("minimal", "full"):
        result = load_evaluation_result(output / f"{variant}.result.json")
        assert result["cases"][0]["passed"] is True
        run_dir = output / result["cases"][0]["run_ref"]["artifact_dir"]
        assert (run_dir / "trace.jsonl").is_file()
    assert (output / "paired-comparison.json").is_file() and (output / "harness-ab-report.json").is_file()
