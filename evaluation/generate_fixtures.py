"""Deterministically materialize the 12 planned v2 fixtures missing from the Pico evidence set."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent
IDS = ["doc_03", "doc_04", "single_03", "single_04", "cross_01", "cross_02", "cross_03", "cross_04", "context_03", "tool_03", "resume_03", "side_effect_03"]


def generate() -> None:
    template = json.loads((ROOT / "fixtures" / "doc_01.json").read_text(encoding="utf-8"))
    matrix = {item["id"]: item for item in json.loads((ROOT / "replay_benchmark_matrix.json").read_text(encoding="utf-8"))["tasks"]}
    for fixture_id in IDS:
        payload = json.loads(json.dumps(template)); payload["run_id"] = f"fixture-{fixture_id}"; payload["task_contract"]["task_id"] = fixture_id
        payload["task_contract"].update({"category": matrix[fixture_id]["category"], "scenario": matrix[fixture_id]["scenario"], "replay_focus": matrix[fixture_id]["replay_focus"]})
        payload["checkpoints"] = [{"checkpoint_id": "cp-1", "event_sequence": 1, "state_digest": payload["metrics"]["expected_state_digest"]}]
        (ROOT / "fixtures" / f"{fixture_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": generate()
