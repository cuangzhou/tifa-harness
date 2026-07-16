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
    replay_run_id: str | None = None
    replay_bundle: dict[str, Any] | None = None
    applied_overrides: dict[str, Any] = field(default_factory=dict)
    same_task_contract: bool = True
    same_snapshot: bool = True


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

    def replay(self, bundle_path: Path, mode: str = "offline", *, spec: ReplaySpec | None = None, workspace: Path | None = None, model_client: Any | None = None) -> ReplayDiffReport | ReplayResult:
        chosen = spec or ReplaySpec(json.loads(bundle_path.read_text(encoding="utf-8"))["run_id"], mode, "read_only" if mode == "offline" else "snapshot_copy")
        chosen.validate()
        base = self._offline(bundle_path)
        if chosen.mode == "offline": return base
        if workspace is None: raise ValueError("forked replay requires source workspace")
        if model_client is None: raise ValueError("forked replay requires an explicit model client")
        source_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if chosen.source_run_id != source_bundle["run_id"]: raise ValueError("ReplaySpec source_run_id mismatch")
        source_run = workspace / ".tifa" / "runs" / chosen.source_run_id
        if not source_run.is_dir(): raise ValueError("source run is not available in workspace")
        if chosen.checkpoint_id and not (source_run / "checkpoints" / f"{chosen.checkpoint_id}.json").is_file(): raise ValueError("checkpoint is not available")
        before = workspace_digest(workspace)
        if chosen.expected_source_digest and before != chosen.expected_source_digest: raise ValueError("source digest mismatch")
        cleaned = False; replay_bundle = None; replay_run_id = None
        with tempfile.TemporaryDirectory(prefix="tifa-replay-") as temp:
            target = Path(temp) / "workspace"
            shutil.copytree(workspace, target, ignore=shutil.ignore_patterns(".git", ".tifa", "__pycache__", ".pytest_cache"))
            target_run = target / ".tifa" / "runs" / chosen.source_run_id
            target_run.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(source_run, target_run)
            from .memory import LayeredMemory
            from .runtime import Tifa
            overrides = chosen.overrides
            if "provider" in overrides and model_client.provider != overrides["provider"]: raise ValueError("provider override does not match supplied model client")
            agent = Tifa.resume_run(target, model_client, chosen.source_run_id, chosen.checkpoint_id, allow_snapshot_copy=True, runtime_overrides={"provider", "model"} if "provider" in overrides else set(), approval_policy="never", context_policy=overrides.get("context_policy", "layered-budget-v2"), memory_enabled=overrides.get("memory_enabled", True))
            if overrides.get("memory_enabled") is False: agent.memory = LayeredMemory()
            contract = source_bundle["task_contract"]
            result = agent.ask(contract["prompt"], verifier=contract.get("verifier"))
            replay_run_id = result.run_id; replay_path = Path(result.run_dir) / "evidence_bundle.json"; replay_bundle = json.loads(replay_path.read_text(encoding="utf-8"))
            cleaned = True
        after = workspace_digest(workspace); base.source_unchanged = before == after
        differences = self.diff(source_bundle, replay_bundle or {}); base.differences = differences
        original_contract = {k: v for k, v in source_bundle["task_contract"].items() if k != "task_id"}
        replay_contract = {k: v for k, v in (replay_bundle or {}).get("task_contract", {}).items() if k != "task_id"}
        same_contract = original_contract == replay_contract
        if chosen.mode == "counterfactual":
            unauthorized = not same_contract
            allowed_provider_change = "provider" in chosen.overrides
            provider_changed = source_bundle.get("provenance", {}).get("provider") != (replay_bundle or {}).get("provenance", {}).get("provider")
            base.confounded = unauthorized or (provider_changed and not allowed_provider_change)
            base.replay_consistent = base.replay_consistent and not base.confounded
        return ReplayResult(chosen, base, before, after, cleaned, replay_run_id, replay_bundle, dict(chosen.overrides), same_contract, before == after)

    @staticmethod
    def diff(original: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
        fields = ("context_manifest", "events", "artifacts", "verifier", "metrics", "checkpoints")
        return {field: {"original": digest(original.get(field)), "replay": digest(replay.get(field))} for field in fields if digest(original.get(field)) != digest(replay.get(field))}

    def replay_to_file(self, bundle_path: Path, output: Path, mode: str = "offline") -> ReplayDiffReport | ReplayResult:
        result = self.replay(bundle_path, mode); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"); return result
