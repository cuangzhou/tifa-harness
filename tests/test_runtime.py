import json
from pathlib import Path

import pytest

from tifa import FakeModelClient, Tifa, build_agent
from tifa.replay import ReplayDiffReport, ReplayRunner
from tifa.runtime import ResumeMismatch, parse


def test_public_api_and_parser():
    assert parse('<final>done</final>') == ("final", "done")
    assert parse('<tool>{"name":"read_file","arguments":{"path":"a"}}</tool>')[0] == "tool"
    assert parse('```json\n{"name":"write_file","arguments":{"path":"a.txt","content":"ok"}}\n```')[0] == "tool"
    assert parse("bad")[0] == "retry"


def test_fake_loop_writes_versioned_artifacts(tmp_path: Path):
    agent = build_agent(tmp_path, FakeModelClient(["<final>done</final>"]), approval_policy="never")
    result = agent.ask("finish")
    assert result.answer == "done"
    run = Path(result.run_dir)
    for name in ("task_state.json", "trace.jsonl", "checkpoints", "report.json", "evidence_bundle.json"):
        assert (run / name).exists()
    assert json.loads((run / "task_state.json").read_text())["schema_version"] == "tifa-task-state.v2"
    replay = ReplayRunner().replay(run / "evidence_bundle.json")
    assert isinstance(replay, ReplayDiffReport) and replay.replay_consistent


def test_tool_loop_and_duplicate_guard(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    call = '<tool>{"name":"read_file","arguments":{"path":"a.txt"}}</tool>'
    agent = build_agent(tmp_path, FakeModelClient([call, call, "<final>done</final>"]), approval_policy="never")
    result = agent.ask("read")
    assert result.tool_steps == 2
    trace = (Path(result.run_dir) / "trace.jsonl").read_text(encoding="utf-8")
    assert "reused committed tool result" in trace


def test_retry_limit(tmp_path: Path):
    result = build_agent(tmp_path, FakeModelClient(["bad"] * 3), max_attempts=2, approval_policy="never").ask("x")
    assert result.stop_reason == "retry_limit_reached"


def test_resume_round_trip_and_mismatch(tmp_path: Path):
    first = build_agent(tmp_path, FakeModelClient(["<final>one</final>"]), approval_policy="never").ask("one")
    resumed = Tifa.from_session(tmp_path, FakeModelClient(["<final>two</final>"]), first.session_id, approval_policy="never")
    assert resumed.ask("two").answer == "two"
    with pytest.raises(ResumeMismatch):
        Tifa.from_session(tmp_path, FakeModelClient(model="different"), first.session_id, approval_policy="never")
