from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from .contracts import TaskContract


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workspace_digest(root: Path) -> str:
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(x in path.parts for x in (".tifa", ".git", "__pycache__", ".pytest_cache")):
            items.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest(items)


SENSITIVE_KEYS = {"content", "text", "answer", "api_key", "authorization", "token", "secret", "password"}


def _safe_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ({"redacted": True, "digest": digest(item), "type": type(item).__name__} if key.lower() in SENSITIVE_KEYS else _safe_summary(item)) for key, item in value.items()}
    if isinstance(value, list): return [_safe_summary(item) for item in value[:20]]
    if isinstance(value, str) and len(value) > 160: return {"redacted": True, "digest": digest(value), "type": "str", "length": len(value)}
    return value


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
        if bundle.get("schema_version") not in {"evidence-bundle.v1", "evidence-bundle.v2", "evidence-bundle.v3"}: raise ValueError("unsupported EvidenceBundle schema")
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
            prompt = contract.get("goal", contract.get("prompt", "Continue replay task"))
            source_contract = TaskContract.from_dict(contract) if source_bundle.get("schema_version") == "evidence-bundle.v3" else None
            result = agent.ask(prompt, verifier=contract.get("verifier"), contract=source_contract)
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
        result: dict[str, Any] = {field: {"original_digest": digest(original.get(field)), "replay_digest": digest(replay.get(field))} for field in fields if digest(original.get(field)) != digest(replay.get(field))}
        left_events = {event.get("sequence"): event for event in original.get("events", [])}; right_events = {event.get("sequence"): event for event in replay.get("events", [])}
        changed_sequences = sorted(sequence for sequence in left_events.keys() | right_events.keys() if left_events.get(sequence) != right_events.get(sequence))
        if changed_sequences:
            result.setdefault("events", {}).update({"changed_sequences": changed_sequences, "changes": [{"sequence": sequence, "original": _safe_summary(left_events.get(sequence)), "replay": _safe_summary(right_events.get(sequence))} for sequence in changed_sequences[:20]]})
        left_context = original.get("context_manifest", {}); right_context = replay.get("context_manifest", {})
        if digest(left_context) != digest(right_context):
            result.setdefault("context_manifest", {}).update({"original_selected_ids": [item.get("case_id", item.get("id")) for item in left_context.get("selected_items", []) if isinstance(item, dict)], "replay_selected_ids": [item.get("case_id", item.get("id")) for item in right_context.get("selected_items", []) if isinstance(item, dict)], "original_token_estimate": left_context.get("token_estimate"), "replay_token_estimate": right_context.get("token_estimate")})
        for label, payload in (("original", original), ("replay", replay)):
            result.setdefault("summary", {})[label] = {"affected_paths": sorted({path for event in payload.get("events", []) for path in event.get("payload", {}).get("affected_paths", [])}), "verifier": _safe_summary(payload.get("verifier", {})), "provider_usage": _safe_summary(payload.get("metrics", {}).get("provider_usage", [])), "security_events": _safe_summary(payload.get("metrics", {}).get("security_events", [])), "duplicate_side_effects": payload.get("metrics", {}).get("duplicate_side_effects", 0)}
        return result

    def replay_to_file(self, bundle_path: Path, output: Path, mode: str = "offline") -> ReplayDiffReport | ReplayResult:
        result = self.replay(bundle_path, mode); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"); return result
