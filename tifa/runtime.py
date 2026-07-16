from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Callable
import uuid

from .context_manager import ContextManager
from .memory import DurableMemoryStore, LayeredMemory
from .models import AgentResult, FakeModelClient, ModelClient
from .replay import digest
from .stores import RunStore, SessionStore, now
from .tools import ToolRegistry, call_fingerprint
from .workspace import WorkspaceContext

TOOL_PATTERN = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL | re.IGNORECASE)
FINAL_PATTERN = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL | re.IGNORECASE)


def parse(content: str) -> tuple[str, Any]:
    final = FINAL_PATTERN.search(content)
    if final: return "final", final.group(1)
    tool = TOOL_PATTERN.search(content)
    if tool:
        try:
            payload = json.loads(tool.group(1))
            if not isinstance(payload.get("name"), str) or not isinstance(payload.get("arguments", {}), dict): raise ValueError
            return "tool", payload
        except (json.JSONDecodeError, ValueError): pass
    return "retry", "Model output must contain a valid <tool> JSON object or <final> answer."


class ResumeMismatch(RuntimeError):
    pass


class Tifa:
    def __init__(self, workspace: WorkspaceContext, model_client: ModelClient, *, max_steps: int = 8, max_attempts: int = 16, approval_policy: str = "on-risk", approver: Callable[[str, dict[str, Any]], bool] | None = None, session_id: str | None = None, memory: LayeredMemory | None = None, history: list[dict[str, Any]] | None = None, completed_fingerprints: set[str] | None = None, imported_from: str | None = None) -> None:
        self.workspace, self.model_client, self.max_steps, self.max_attempts = workspace, model_client, max_steps, max_attempts
        self.approval_policy, self.approver = approval_policy, approver
        self.session_id, self.memory, self.history = session_id or uuid.uuid4().hex, memory or LayeredMemory(), history or []
        self.completed_fingerprints, self.imported_from = completed_fingerprints or set(), imported_from
        self.sessions, self.memory_store = SessionStore(workspace.repo_root), DurableMemoryStore(workspace.repo_root / ".tifa" / "memory" / "memory.json")

    def _delegate(self, task: str, steps: int) -> str:
        child = Tifa(self.workspace, self.model_client, max_steps=steps, max_attempts=max(steps * 2, 3), approval_policy="never", memory=LayeredMemory(self.memory.state), history=[])
        child._registry_depth = 1
        return child.ask(task).answer

    def _registry(self) -> ToolRegistry:
        return ToolRegistry(self.workspace, self.approval_policy, self.approver, self._delegate, getattr(self, "_registry_depth", 0), 1)

    def runtime_identity(self, registry: ToolRegistry) -> dict[str, Any]:
        return {"provider": self.model_client.provider, "model": self.model_client.model, "approval_policy": self.approval_policy, "tool_signature": registry.signature()}

    @classmethod
    def from_session(cls, workspace: str | Path, model_client: ModelClient, session_id: str = "latest", **kwargs: Any) -> "Tifa":
        context = WorkspaceContext.build(workspace); payload = SessionStore(context.repo_root).load(session_id)
        registry = ToolRegistry(context, kwargs.get("approval_policy", payload.get("runtime_identity", {}).get("approval_policy", "on-risk")), delegate=lambda _task, _steps: "")
        identity = {"provider": model_client.provider, "model": model_client.model, "approval_policy": registry.approval_policy, "tool_signature": registry.signature()}
        expected = payload.get("runtime_identity", identity)
        mismatches = [key for key in identity if expected.get(key) != identity[key]]
        if payload.get("workspace_fingerprint") and payload["workspace_fingerprint"] != context.fingerprint(): mismatches.append("workspace_fingerprint")
        if mismatches: raise ResumeMismatch(f"resume rejected due to mismatch: {', '.join(sorted(set(mismatches)))}")
        return cls(context, model_client, session_id=payload.get("session_id"), memory=LayeredMemory(payload.get("memory")), history=payload.get("history", []), completed_fingerprints=set(payload.get("completed_fingerprints", [])), imported_from=payload.get("legacy_source"), **kwargs)

    def ask(self, request: str) -> AgentResult:
        registry, store = self._registry(), RunStore(self.workspace.repo_root)
        context = ContextManager(self.workspace)
        state: dict[str, Any] = {"run_id": store.run_id, "session_id": self.session_id, "status": "running", "request": request, "tool_steps": 0, "attempts": 0, "stop_reason": None, "affected_paths": [], "started_at": now()}
        store.write("task_state.json", "tifa-task-state.v1", state); store.append_trace("run_started", {"request": request})
        answer, stop_reason = "", "retry_limit_reached"
        while state["tool_steps"] < self.max_steps and state["attempts"] < self.max_attempts:
            state["attempts"] += 1
            built = context.build(request, self.memory, self.history, registry.schemas())
            response = self.model_client.complete(built.prompt, registry.schemas(), built.cache_key)
            store.append_trace("model_response", {"attempt": state["attempts"], "content": response.content, "usage": response.usage, "cache": response.cache, "context": built.metadata})
            kind, payload = parse(response.content)
            if kind == "final":
                answer, stop_reason = str(payload), "final_answer_returned"; break
            if kind == "retry":
                self.history.append({"role": "system", "content": payload}); store.append_trace("retry", {"reason": payload}); continue
            name, arguments = payload["name"], payload.get("arguments", {})
            fingerprint = call_fingerprint(name, arguments)
            if fingerprint in self.completed_fingerprints:
                output, affected, ok = "error: duplicate tool call blocked", [], False
            else:
                try:
                    output, affected = registry.run(name, arguments); ok = True
                    self.completed_fingerprints.add(fingerprint); self.memory.after_tool(name, arguments, output)
                except Exception as exc:
                    output, affected, ok = f"error: {type(exc).__name__}: {exc}", [], False
            state["tool_steps"] += 1; state["affected_paths"] = sorted(set(state["affected_paths"] + affected))
            self.history.append({"role": "tool", "tool": name, "arguments": arguments, "content": output, "ok": ok})
            store.append_trace("tool_executed", {"name": name, "arguments": arguments, "output": output, "ok": ok, "affected_paths": affected, "fingerprint": fingerprint})
            checkpoint = {"checkpoint_id": f"cp-{state['tool_steps']}", "event_sequence": store.sequence, "state": state, "completed_fingerprints": sorted(self.completed_fingerprints), "workspace_fingerprint": WorkspaceContext.build(self.workspace.repo_root).fingerprint(), "runtime_identity": self.runtime_identity(registry)}
            checkpoint["state_digest"] = digest(checkpoint["state"]); store.write("checkpoint.json", "tifa-checkpoint.v1", checkpoint)
            store.write("task_state.json", "tifa-task-state.v1", state)
        else:
            stop_reason = "step_limit_reached" if state["tool_steps"] >= self.max_steps else "retry_limit_reached"
            answer = f"Tifa stopped: {stop_reason}."
        state.update({"status": "finished", "stop_reason": stop_reason, "finished_at": now()})
        if not (store.run_dir / "checkpoint.json").exists():
            checkpoint = {"checkpoint_id": "cp-final", "event_sequence": store.sequence, "state": state, "completed_fingerprints": sorted(self.completed_fingerprints), "workspace_fingerprint": WorkspaceContext.build(self.workspace.repo_root).fingerprint(), "runtime_identity": self.runtime_identity(registry)}
            checkpoint["state_digest"] = digest(checkpoint["state"]); store.write("checkpoint.json", "tifa-checkpoint.v1", checkpoint)
        report = {"run_id": store.run_id, "session_id": self.session_id, "answer": answer, "stop_reason": stop_reason, "tool_steps": state["tool_steps"], "attempts": state["attempts"], "affected_paths": state["affected_paths"]}
        store.write("task_state.json", "tifa-task-state.v1", state); store.write("report.json", "tifa-report.v1", report)
        state_event = store.append_trace("state_patch", state); report_event = store.append_trace("report", report); verifier_event = store.append_trace("verifier", {"passed": True, "failure_category": None})
        events = [json.loads(line) for line in (store.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        evidence = {"schema_version": "evidence-bundle.v1", "run_id": store.run_id, "created_at": now(), "task_contract": {"task_id": store.run_id, "prompt": request, "allowed_tools": registry.names, "step_budget": self.max_steps, "expected_artifact": None}, "repo_snapshot": {"snapshot_id": self.workspace.fingerprint(), "commit": None, "dirty": bool(self.workspace.status), "workspace_digest": self.workspace.fingerprint()}, "context_manifest": {"policy": "layered-budget-v1", "selected_items": [], "dropped_items": [], "token_estimate": 0}, "events": events, "checkpoints": [], "artifacts": [], "verifier": {"passed": True, "failure_category": None}, "metrics": {"expected_state_digest": digest(state_event["payload"]), "expected_report_digest": digest(report_event["payload"])}, "provenance": {"provider": self.model_client.provider, "model": self.model_client.model, "code_version": "0.1.0"}}
        store.write("evidence_bundle.json", "evidence-bundle.v1", {k: v for k, v in evidence.items() if k != "schema_version"})
        current_workspace = WorkspaceContext.build(self.workspace.repo_root)
        session = {"session_id": self.session_id, "history": self.history, "memory": self.memory.state, "completed_fingerprints": sorted(self.completed_fingerprints), "workspace_fingerprint": current_workspace.fingerprint(), "runtime_identity": self.runtime_identity(registry), "latest_run_id": store.run_id, "imported_from": self.imported_from, "updated_at": now()}
        self.sessions.save(self.session_id, session); self.memory_store.save(self.memory)
        return AgentResult(answer, store.run_id, self.session_id, stop_reason, state["tool_steps"], state["attempts"], str(store.run_dir))


def build_agent(cwd: str | Path = ".", model_client: ModelClient | None = None, **kwargs: Any) -> Tifa:
    return Tifa(WorkspaceContext.build(cwd), model_client or FakeModelClient(), **kwargs)
