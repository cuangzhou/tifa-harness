import json
from pathlib import Path

import pytest

from tifa.benchmark import evaluation_root, run_replay_benchmark
from tifa.replay import ReplayRunner
from tifa.result_contract import export_measured_metrics, make_result, write_result


def test_all_24_fixtures_replay():
    fixtures = sorted((evaluation_root() / "fixtures").glob("*.json"))
    assert len(fixtures) == 24
    assert all(ReplayRunner().replay(path).replay_consistent for path in fixtures)


def test_tamper_and_unknown_schema(tmp_path: Path):
    source = evaluation_root() / "fixtures" / "doc_01.json"
    payload = json.loads(source.read_text(encoding="utf-8")); payload["events"][1]["payload"]["content"] = "tampered"
    target = tmp_path / "tampered.json"; target.write_text(json.dumps(payload), encoding="utf-8")
    assert not ReplayRunner().replay(target).artifact_digest_match
    payload["schema_version"] = "unknown"; target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        ReplayRunner().replay(target)


def test_forked_replay_requires_workspace():
    source = evaluation_root() / "fixtures" / "doc_01.json"
    with pytest.raises(ValueError, match="workspace"):
        ReplayRunner().replay(source, "forked")


def test_placeholder_export_refused(tmp_path: Path):
    result = make_result(project="Tifa", benchmark="test", dataset_version="v1", result_kind="placeholder", implementation_status="design", git_commit="none", command="test", case_count=1, repetitions=1, seed=1, metrics={}, limitations=[])
    source = tmp_path / "x_PLACEHOLDER_.json"; write_result(result, source)
    with pytest.raises(ValueError, match="REFUSED"):
        export_measured_metrics(source, tmp_path / "out.json")


def test_smoke_reports_planned_vs_executed(tmp_path: Path):
    result = run_replay_benchmark("smoke", tmp_path / "result.json")
    assert result["metrics"]["planned_fixture_count"] == 24
    assert result["metrics"]["executed_fixture_count"] == 24
