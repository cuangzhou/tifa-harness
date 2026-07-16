from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


NOT_IMPLEMENTED = {"status": "NOT_IMPLEMENTED", "supported_modes": ["offline"], "message": "Forked and Counterfactual Replay are intentionally not implemented."}


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ReplayDiffReport:
    run_id: str
    events_replayed: int
    state_digest_match: bool
    report_digest_match: bool
    artifact_digest_match: bool
    verifier_match: bool
    replay_consistent: bool
    failure_category: str | None
    duration_ms: float


class ReplayRunner:
    def replay(self, bundle_path: Path, mode: str = "offline") -> ReplayDiffReport | dict[str, Any]:
        if mode != "offline": return {**NOT_IMPLEMENTED, "requested_mode": mode}
        started = time.perf_counter()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("schema_version") != "evidence-bundle.v1": raise ValueError("unsupported EvidenceBundle schema")
        events = sorted(bundle["events"], key=lambda event: event["sequence"])
        if [e["sequence"] for e in events] != list(range(1, len(events) + 1)): raise ValueError("trace event sequences must be contiguous from 1")
        state: dict[str, Any] = {}; report: dict[str, Any] = {}; artifacts: dict[str, Any] = {}
        verifier_passed = False; failure_category = None
        for event in events:
            kind, payload = event["type"], event.get("payload", {})
            if kind == "state_patch": state.update(payload)
            elif kind == "artifact": artifacts[payload["path"]] = payload.get("content")
            elif kind == "verifier": verifier_passed, failure_category = bool(payload.get("passed")), payload.get("failure_category")
            elif kind == "report": report = payload
        state_ok = digest(state) == bundle["metrics"]["expected_state_digest"]
        report_ok = digest(report) == bundle["metrics"]["expected_report_digest"]
        artifact_ok = all(digest(artifacts.get(item["path"])) == item["digest"] for item in bundle["artifacts"])
        verifier_ok = verifier_passed == bool(bundle["verifier"]["passed"])
        consistent = state_ok and report_ok and artifact_ok and verifier_ok
        return ReplayDiffReport(bundle["run_id"], len(events), state_ok, report_ok, artifact_ok, verifier_ok, consistent, failure_category, (time.perf_counter() - started) * 1000)

    def replay_to_file(self, bundle_path: Path, output: Path, mode: str = "offline") -> ReplayDiffReport | dict[str, Any]:
        report = self.replay(bundle_path, mode)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report) if isinstance(report, ReplayDiffReport) else report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
