import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tifa import FakeModelClient, ModelResponse, ToolCall, Tifa, build_agent
from tifa.cases import CaseCard, CaseStore
from dataclasses import asdict
from tifa.replay import ReplayResult, ReplayRunner, ReplaySpec, workspace_digest
from tifa.runtime import ResumeMismatch


def test_structured_tool_call_and_unconfigured_verifier(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    response = ModelResponse(tool_calls=[ToolCall("call-1", "read_file", {"path": "a.txt"})])
    result = build_agent(tmp_path, FakeModelClient([response, "<final>done</final>"]), approval_policy="never").ask("read")
    bundle = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "evidence-bundle.v2"
    assert bundle["verifier"]["status"] == "not_configured" and bundle["verifier"]["passed"] is None
    assert bundle["checkpoints"]


def test_verifier_file_and_assertion(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask("x", verifier={"files": [{"path": "ok.txt"}], "assertions": [{"actual": 1, "equals": 1}]})
    report = json.loads((Path(result.run_dir) / "report.json").read_text(encoding="utf-8"))
    assert report["verifier"]["passed"] is True


def test_resume_from_committed_checkpoint_does_not_repeat_write(tmp_path: Path):
    call = ModelResponse(tool_calls=[ToolCall("write-1", "write_file", {"path": "out.txt", "content": "once"})])
    agent = build_agent(tmp_path, FakeModelClient([call]), approval_policy="never")
    with pytest.raises(InterruptedError, match="TOOL_COMMITTED"):
        agent.ask("write", interrupt_at="TOOL_COMMITTED")
    run_id = next((tmp_path / ".tifa" / "runs").iterdir()).name
    resumed = Tifa.resume_run(tmp_path, FakeModelClient(["<final>done</final>"]), run_id, approval_policy="never")
    result = resumed.ask("continue")
    assert (tmp_path / "out.txt").read_text() == "once"
    assert json.loads((Path(result.run_dir) / "report.json").read_text())["parent_run_id"] == run_id


@pytest.mark.parametrize("phase", ["MODEL_PENDING", "TOOL_PENDING", "TOOL_RUNNING", "TOOL_COMMITTED", "VERIFIER_PENDING", "FINALIZING"])
def test_interruption_matrix_has_no_duplicate_side_effect(tmp_path: Path, phase: str):
    call = ModelResponse(tool_calls=[ToolCall("write-1", "write_file", {"path": "matrix.txt", "content": "once"})])
    agent = build_agent(tmp_path, FakeModelClient([call, "<final>done</final>"]), approval_policy="never")
    with pytest.raises(InterruptedError): agent.ask("write once", interrupt_at=phase)
    run_id = next((tmp_path / ".tifa" / "runs").iterdir()).name
    outputs = [call, "<final>done</final>"] if phase in {"MODEL_PENDING", "TOOL_PENDING", "TOOL_RUNNING"} else ["<final>done</final>"]
    resumed = Tifa.resume_run(tmp_path, FakeModelClient(outputs), run_id, approval_policy="never")
    resumed.ask("continue")
    assert (tmp_path / "matrix.txt").read_text() == "once"
    commits = 0
    for trace in (tmp_path / ".tifa" / "runs").glob("*/trace.jsonl"):
        commits += sum(bool(json.loads(line)["type"] == "tool_commit" and json.loads(line)["payload"].get("after_digest")) for line in trace.read_text().splitlines())
    assert commits == 1


def test_checkpoint_tamper_rejected(tmp_path: Path):
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask("x")
    checkpoint = next((Path(result.run_dir) / "checkpoints").glob("*.json")); payload = json.loads(checkpoint.read_text()); payload["state"]["request"] = "tampered"; checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ResumeMismatch, match="digest"):
        Tifa.resume_run(tmp_path, FakeModelClient(), result.run_id, checkpoint.stem, approval_policy="never")


def test_forked_replay_never_writes_source(tmp_path: Path):
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    source = build_agent(tmp_path, FakeModelClient(["<final>source complete</final>"]), approval_policy="never").ask("source task")
    before = workspace_digest(tmp_path); bundle = Path(source.run_dir) / "evidence_bundle.json"
    result = ReplayRunner().replay(bundle, spec=ReplaySpec(source.run_id, "forked", "snapshot_copy", expected_source_digest=before), workspace=tmp_path, model_client=FakeModelClient(["<final>replayed</final>"]))
    assert isinstance(result, ReplayResult) and result.report.source_unchanged and result.replay_run_id and result.replay_bundle
    assert (tmp_path / "source.txt").read_text() == "source"


def test_counterfactual_requires_one_override(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        ReplaySpec("x", "counterfactual", "snapshot_copy", overrides={}).validate()
    ReplaySpec("x", "counterfactual", "snapshot_copy", overrides={"memory_enabled": False}).validate()


@pytest.mark.parametrize("variable,value", [("memory_enabled", False), ("context_policy", "minimal"), ("provider", "alternate")])
def test_counterfactual_override_is_applied(tmp_path: Path, variable: str, value):
    source = build_agent(tmp_path, FakeModelClient(["<final>source</final>"]), approval_policy="never").ask("counterfactual task", verifier={"assertions": [{"actual": 1, "equals": 1}]})
    client = FakeModelClient(["<final>replayed</final>"])
    if variable == "provider": client.provider = "alternate"
    spec = ReplaySpec(source.run_id, "counterfactual", "snapshot_copy", overrides={variable: value}, expected_source_digest=workspace_digest(tmp_path))
    result = ReplayRunner().replay(Path(source.run_dir) / "evidence_bundle.json", spec=spec, workspace=tmp_path, model_client=client)
    assert isinstance(result, ReplayResult) and not result.report.confounded and result.applied_overrides == {variable: value}
    if variable == "memory_enabled": assert result.replay_bundle["context_manifest"]["memory_enabled"] is False
    if variable == "context_policy": assert result.replay_bundle["context_manifest"]["policy"] == "minimal"
    if variable == "provider": assert result.replay_bundle["provenance"]["provider"] == "alternate"


def test_case_promotion_and_search(tmp_path: Path):
    store = CaseStore(tmp_path / "cases")
    source = build_agent(tmp_path, FakeModelClient(["<final>source</final>"]), approval_policy="never").ask("case task", verifier={"assertions": [{"actual": 1, "equals": 1}]})
    replay = ReplayRunner().replay(Path(source.run_dir) / "evidence_bundle.json", spec=ReplaySpec(source.run_id, "counterfactual", "snapshot_copy", overrides={"memory_enabled": False}, expected_source_digest=workspace_digest(tmp_path)), workspace=tmp_path, model_client=FakeModelClient(["<final>fixed</final>"]))
    card = CaseCard({"category": "bugfix", "tool_pattern": ["read_file"]}, {"category": "tool_error", "stop_reason": "failed"}, {"variable": "memory_enabled", "before": True, "after": False}, [f"run:{source.run_id}", f"run:{replay.replay_run_id}"], {"allowed_categories": ["bugfix"], "excluded_conditions": []}, "Use bounded memory")
    store.save(card)
    promoted = store.promote(card, asdict(replay))
    assert promoted.verification_status == "verified" and store.search("bugfix")[0].case_id == card.case_id
    store.reject(card.case_id); assert not store.search("bugfix")


def test_v2_bundle_schema(tmp_path: Path):
    result = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never").ask("x")
    bundle = json.loads((Path(result.run_dir) / "evidence_bundle.json").read_text())
    schema = json.loads((Path(__file__).parents[1] / "evaluation" / "evidence_bundle.schema.json").read_text())
    Draft202012Validator(schema).validate(bundle)
