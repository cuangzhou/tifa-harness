from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workspace_digest(root: Path) -> str:
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(x in path.parts for x in (".tifa", ".git", "__pycache__", ".pytest_cache")):
            items.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest(items)


@dataclass
class ReplaySpec:
    source_run_id: str
    mode: str = "offline"
    workspace_policy: str = "read_only"
    checkpoint_id: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    expected_source_digest: str | None = None
    schema_version: str = "replay-spec.v1"

    def validate(self) -> None:
        if self.mode not in {"offline", "forked", "counterfactual"}: raise ValueError("invalid replay mode")
        if self.mode == "offline" and self.workspace_policy != "read_only": raise ValueError("offline replay must be read_only")
        if self.mode == "counterfactual" and len(self.overrides) != 1: raise ValueError("counterfactual replay requires exactly one override")
        if set(self.overrides) - {"memory_enabled", "context_policy", "provider"}: raise ValueError("unsupported override")


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
    checkpoint_digest_match: bool = True
    source_unchanged: bool = True
    confounded: bool = False
    differences: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayResult:
    spec: ReplaySpec
    report: ReplayDiffReport
    workspace_digest_before: str | None = None
    workspace_digest_after: str | None = None
    isolated_workspace_cleaned: bool = True


class ReplayRunner:
    def _offline(self, bundle_path: Path) -> ReplayDiffReport:
        started = time.perf_counter(); bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("schema_version") not in {"evidence-bundle.v1", "evidence-bundle.v2"}: raise ValueError("unsupported EvidenceBundle schema")
        events = sorted(bundle["events"], key=lambda event: event["sequence"])
        if [e["sequence"] for e in events] != list(range(1, len(events) + 1)): raise ValueError("trace event sequences must be contiguous from 1")
        state: dict[str, Any] = {}; report: dict[str, Any] = {}; artifacts: dict[str, Any] = {}; verifier: dict[str, Any] = {}
        for event in events:
            kind, payload = event["type"], event.get("payload", {})
            if kind == "state_patch": state.update(payload)
            elif kind == "artifact": artifacts[payload["path"]] = payload.get("content")
            elif kind == "verifier": verifier = payload
            elif kind == "report": report = payload
        metrics = bundle["metrics"]
        state_ok = digest(state) == metrics["expected_state_digest"]; report_ok = digest(report) == metrics["expected_report_digest"]
        artifact_ok = all(digest(artifacts.get(i["path"])) == i["digest"] for i in bundle.get("artifacts", []))
        expected_verifier = bundle.get("verifier", {}); verifier_ok = verifier.get("passed") == expected_verifier.get("passed") and verifier.get("status", expected_verifier.get("status")) == expected_verifier.get("status", verifier.get("status"))
        checkpoint_ok = all(c.get("state_digest") for c in bundle.get("checkpoints", []))
        consistent = state_ok and report_ok and artifact_ok and verifier_ok and checkpoint_ok
        return ReplayDiffReport(bundle["run_id"], len(events), state_ok, report_ok, artifact_ok, verifier_ok, consistent, verifier.get("failure_category"), (time.perf_counter() - started) * 1000, checkpoint_ok)

    def replay(self, bundle_path: Path, mode: str = "offline", *, spec: ReplaySpec | None = None, workspace: Path | None = None, executor: Callable[[Path, ReplaySpec], None] | None = None) -> ReplayDiffReport | ReplayResult:
        chosen = spec or ReplaySpec(json.loads(bundle_path.read_text(encoding="utf-8"))["run_id"], mode, "read_only" if mode == "offline" else "snapshot_copy")
        chosen.validate()
        base = self._offline(bundle_path)
        if chosen.mode == "offline": return base
        if workspace is None: raise ValueError("forked replay requires source workspace")
        before = workspace_digest(workspace)
        if chosen.expected_source_digest and before != chosen.expected_source_digest: raise ValueError("source digest mismatch")
        cleaned = False
        with tempfile.TemporaryDirectory(prefix="tifa-replay-") as temp:
            target = Path(temp) / "workspace"
            shutil.copytree(workspace, target, ignore=shutil.ignore_patterns(".git", ".tifa", "__pycache__", ".pytest_cache"))
            if executor: executor(target, chosen)
            cleaned = True
        after = workspace_digest(workspace); base.source_unchanged = before == after
        if chosen.mode == "counterfactual":
            authorized = set(chosen.overrides); observed = set(base.differences.get("overrides", authorized)); base.confounded = observed != authorized
            base.replay_consistent = base.replay_consistent and not base.confounded
        return ReplayResult(chosen, base, before, after, cleaned)

    @staticmethod
    def diff(original: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
        fields = ("context_manifest", "events", "artifacts", "verifier", "metrics", "checkpoints")
        return {field: {"original": digest(original.get(field)), "replay": digest(replay.get(field))} for field in fields if digest(original.get(field)) != digest(replay.get(field))}

    def replay_to_file(self, bundle_path: Path, output: Path, mode: str = "offline") -> ReplayDiffReport | ReplayResult:
        result = self.replay(bundle_path, mode); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"); return result
