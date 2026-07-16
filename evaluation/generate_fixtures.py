"""Generate the complete deterministic Tifa replay suite from scenario definitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate() -> None:
    tasks = json.loads((ROOT / "replay_benchmark_matrix.json").read_text(encoding="utf-8"))["tasks"]
    for index, task in enumerate(tasks, 1):
        fixture_id = task["id"]; artifact_path = f"artifacts/{fixture_id}.txt"; content = f"Tifa fixture {fixture_id}: {task['scenario']}"
        failure = "tool_failure" if fixture_id in {"tool_02", "tool_03"} else "interrupted_partial_state" if fixture_id == "resume_02" else "policy_denied" if fixture_id == "side_effect_02" else None
        state = {"fixture_id": fixture_id, "category": task["category"], "status": "failed" if failure else "completed", "replay_focus": task["replay_focus"], "ordinal": index}
        report = {"fixture_id": fixture_id, "scenario": task["scenario"], "stop_reason": failure or "final_answer_returned", "affected_paths": [artifact_path]}
        verifier = {"status": "failed" if failure else "passed", "passed": failure is None, "failure_category": failure}
        events = [
            {"sequence": 1, "type": "state_patch", "timestamp": "2026-07-16T00:00:00+00:00", "payload": state},
            {"sequence": 2, "type": "artifact", "timestamp": "2026-07-16T00:00:01+00:00", "payload": {"path": artifact_path, "content": content}},
            {"sequence": 3, "type": "context_manifest", "timestamp": "2026-07-16T00:00:02+00:00", "payload": {"policy": "layered-budget-v2", "selected_items": [fixture_id], "token_estimate": 100 + index}},
            {"sequence": 4, "type": "report", "timestamp": "2026-07-16T00:00:03+00:00", "payload": report},
            {"sequence": 5, "type": "verifier", "timestamp": "2026-07-16T00:00:04+00:00", "payload": verifier},
        ]
        payload = {
            "schema_version": "evidence-bundle.v2", "run_id": f"fixture-{fixture_id}", "created_at": "2026-07-16T00:00:00+00:00",
            "task_contract": {"task_id": fixture_id, "prompt": f"Execute Tifa scenario: {task['scenario']}", "allowed_tools": ["read_file", "patch_file"], "step_budget": 4, "expected_artifact": artifact_path, "category": task["category"], "scenario": task["scenario"], "replay_focus": task["replay_focus"], "verifier": {"files": [{"path": artifact_path}]}},
            "repo_snapshot": {"snapshot_id": f"snapshot-{fixture_id}", "commit": None, "dirty": False, "workspace_digest": digest({"fixture": fixture_id})},
            "context_manifest": {"policy": "layered-budget-v2", "selected_items": [{"fixture_id": fixture_id}], "dropped_items": [], "token_estimate": 100 + index},
            "events": events, "checkpoints": [{"checkpoint_id": f"cp-{fixture_id}", "event_sequence": 1, "state_digest": digest(state)}],
            "artifacts": [{"path": artifact_path, "digest": digest(content), "status": "modified"}], "verifier": verifier,
            "metrics": {"expected_state_digest": digest(state), "expected_report_digest": digest(report), "tool_steps": 1, "tokens": 100 + index},
            "provenance": {"provider": "deterministic", "model": "fixture-runtime", "code_version": "0.3.0"},
        }
        (ROOT / "fixtures" / f"{fixture_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": generate()
