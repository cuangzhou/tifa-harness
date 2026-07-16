from __future__ import annotations

import json
from pathlib import Path

import pytest

from tifa import FakeModelClient, ModelResponse, SemanticIndex, TaskContract, ToolCall, build_agent, evaluate_case_assistance
from tifa.eval_suite import CATEGORY_COUNTS, professional_tasks, suite_manifest
from tifa.operations import RunLock, collect_garbage, continuation_lineage, migrate_artifact
from tifa.reporting import load_run, render_report
from tifa.verifier import verify_contract
from tifa.workspace import WorkspaceContext
from tifa.cli import main


def test_completion_gate_repairs_before_success(tmp_path: Path):
    outputs = ["<final>premature</final>", ModelResponse(tool_calls=[ToolCall("w1", "write_file", {"path": "result.txt", "content": "READY"})]), "<final>done</final>"]
    contract = TaskContract("create result", ["write_file"], ["result.txt"], {"files": [{"path": "result.txt", "equals": "READY"}]}, max_repairs=2)
    result = build_agent(tmp_path, FakeModelClient(outputs), approval_policy="never").ask(contract.goal, contract=contract)
    state = json.loads((Path(result.run_dir) / "task_state.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (Path(result.run_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result.stop_reason == "final_answer_returned" and state["task_phase"] == "COMPLETE" and state["repairs"] == 1
    assert any(event["type"] == "repair_feedback" for event in events)


def test_completion_gate_fails_without_required_verifier(tmp_path: Path):
    contract = TaskContract("finish", require_verifier=True, max_repairs=0)
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask(contract.goal, contract=contract)
    assert result.stop_reason == "completion_gate_failed"


def test_contract_write_allowlist(tmp_path: Path):
    call = ModelResponse(tool_calls=[ToolCall("w1", "write_file", {"path": "blocked.txt", "content": "x"})])
    contract = TaskContract("write", ["write_file"], ["allowed"], {"files": [{"path": "allowed/result.txt"}]}, max_repairs=0)
    result = build_agent(tmp_path, FakeModelClient([call, "<final>done</final>"]), approval_policy="never").ask(contract.goal, contract=contract)
    assert not (tmp_path / "blocked.txt").exists() and result.stop_reason == "completion_gate_failed"


def test_loop_feedback_recovers_within_repair_budget(tmp_path: Path):
    (tmp_path / "source.txt").write_text("read me", encoding="utf-8")
    repeated = [ModelResponse(tool_calls=[ToolCall(f"r{index}", "read_file", {"path": "source.txt"})]) for index in range(3)]
    write = ModelResponse(tool_calls=[ToolCall("w1", "write_file", {"path": "result.txt", "content": "READY"})])
    contract = TaskContract("recover", ["read_file", "write_file"], ["result.txt"], {"files": [{"path": "result.txt", "equals": "READY"}]}, max_repairs=2)
    result = build_agent(tmp_path, FakeModelClient([*repeated, write, "<final>done</final>"]), approval_policy="never").ask(contract.goal, contract=contract)
    assert result.stop_reason == "final_answer_returned" and (tmp_path / "result.txt").read_text() == "READY"


def test_advanced_verifier(tmp_path: Path):
    (tmp_path / "module.py").write_text("def ready():\n    return 'READY'\n", encoding="utf-8")
    spec = {"files": [{"path": "module.py", "contains": "READY", "regex": "def\\s+ready"}], "ast": [{"path": "module.py", "kind": "function", "name": "ready"}], "allowed_changed_paths": ["module.py"], "max_changed_files": 1, "coverage": [{"metric": "line", "actual": 91, "minimum": 85}], "performance": [{"metric": "p95", "actual": 100, "maximum": 200}]}
    result = verify_contract(tmp_path, spec, affected_paths=["module.py"])
    assert result["passed"] is True and result["summary"] == "all checks passed"


def test_semantic_index_queries_and_incremental_status(tmp_path: Path):
    (tmp_path / "src").mkdir(); (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text("import json\ndef calculate(value):\n    return value + 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_service.py").write_text("from src.service import calculate\nassert calculate(1) == 2\n", encoding="utf-8")
    (tmp_path / "web.ts").write_text("export function render() { return 1 }\n", encoding="utf-8")
    index = SemanticIndex(WorkspaceContext.build(tmp_path)); built = index.build()
    assert built["symbol_count"] >= 2 and index.search_symbols("calculate")[0].path == "src/service.py"
    assert index.find_references("calculate") and "tests/test_service.py" in index.related_tests("src/service.py")
    assert "json" in index.dependency_context("src/service.py")["direct"]


def test_operations_reporting_and_migration(tmp_path: Path):
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask("done")
    assert load_run(tmp_path, result.run_id)["report"]["stop_reason"] == "final_answer_returned"
    assert render_report(tmp_path, result.run_id, tmp_path / "report.html", "html").exists()
    assert continuation_lineage(tmp_path, result.run_id)[0]["run_id"] == result.run_id
    assert collect_garbage(tmp_path)["mode"] == "dry-run"
    old = tmp_path / "old.json"; old.write_text(json.dumps({"schema_version": "tifa-report.v2"}), encoding="utf-8")
    assert migrate_artifact(old)["schema_version"] == "tifa-report.v3"
    lock = RunLock(tmp_path / "run.lock").acquire()
    with pytest.raises(RuntimeError): RunLock(tmp_path / "run.lock").acquire()
    lock.release()


def test_professional_suite_contract_counts():
    tasks = professional_tasks(); manifest = suite_manifest()
    assert len(tasks) == 100 == manifest["task_count"] and sum(CATEGORY_COUNTS.values()) == 100
    assert len({task.task_id for task in tasks}) == 100 and all(task.contract.verifier for task in tasks)


def test_case_assistance_remains_opt_in_without_clean_gain():
    result = evaluate_case_assistance([{"passed": True}, {"passed": False}], [{"passed": False}, {"passed": True}])
    assert result["decision"] == "remain_opt_in" and result["regressions"] == 1


def test_professional_cli_surfaces(tmp_path: Path):
    contract = tmp_path / "contract.json"; contract.write_text(json.dumps({"goal": "finish", "require_verifier": False}), encoding="utf-8")
    assert main(["run", "--contract", str(contract), "--cwd", str(tmp_path)]) == 0
    run_id = next((tmp_path / ".tifa" / "runs").iterdir()).name
    assert main(["inspect", "run", run_id, "--cwd", str(tmp_path), "--lineage"]) == 0
    assert main(["index", "build", "--cwd", str(tmp_path)]) == 0
    assert main(["index", "status", "--cwd", str(tmp_path)]) == 0
    assert main(["gc", "--cwd", str(tmp_path), "--dry-run"]) == 0
    assert main(["report", run_id, "--cwd", str(tmp_path), "--format", "json", "--output", str(tmp_path / "run.json")]) == 0
