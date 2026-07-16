from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable
import uuid

from .context_manager import ContextManager
from .cases import CaseStore
from .memory import DurableMemoryStore, LayeredMemory
from .models import AgentResult, FakeModelClient, ModelClient, ToolCall
from .providers import ProviderError
from .replay import digest
from .stores import RunStore, SessionStore, now
from .tools import ToolRegistry, call_fingerprint
from .verifier import file_digest, verify_contract
from .workspace import WorkspaceContext

TOOL_PATTERN = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL | re.IGNORECASE)
FINAL_PATTERN = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL | re.IGNORECASE)
PHASES = {"MODEL_PENDING", "TOOL_PENDING", "TOOL_RUNNING", "TOOL_COMMITTED", "FINALIZING"}


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
    return "retry", "Model output must contain a valid tool call or <final> answer."


class ResumeMismatch(RuntimeError): pass


class Tifa:
    def __init__(self, workspace: WorkspaceContext, model_client: ModelClient, *, max_steps: int = 8, max_attempts: int = 16, approval_policy: str = "on-risk", approver: Callable[[str, dict[str, Any]], bool] | None = None, session_id: str | None = None, memory: LayeredMemory | None = None, history: list[dict[str, Any]] | None = None, messages: list[dict[str, Any]] | None = None, completed_fingerprints: set[str] | None = None, imported_from: str | None = None, parent_run_id: str | None = None, source_checkpoint_id: str | None = None, context_policy: str = "layered-budget-v2", memory_enabled: bool = True) -> None:
        self.workspace, self.model_client, self.max_steps, self.max_attempts = workspace, model_client, max_steps, max_attempts
        self.approval_policy, self.approver = approval_policy, approver
        self.session_id, self.memory, self.history = session_id or uuid.uuid4().hex, memory or LayeredMemory(), history or []
        self.messages, self.completed_fingerprints = messages or [], completed_fingerprints or set()
        self.imported_from, self.parent_run_id, self.source_checkpoint_id = imported_from, parent_run_id, source_checkpoint_id
        self.context_policy, self.memory_enabled = context_policy, memory_enabled
        self.sessions = SessionStore(workspace.repo_root); self.memory_store = DurableMemoryStore(workspace.repo_root / ".tifa" / "memory" / "memory.json")

    def _delegate(self, task: str, steps: int) -> str:
        child = Tifa(self.workspace, self.model_client, max_steps=steps, max_attempts=max(steps * 2, 3), approval_policy="never", memory=LayeredMemory(self.memory.state)); child._registry_depth = 1
        return child.ask(task).answer

    def _registry(self) -> ToolRegistry:
        return ToolRegistry(self.workspace, self.approval_policy, self.approver, self._delegate, getattr(self, "_registry_depth", 0), 1)

    def runtime_identity(self, registry: ToolRegistry) -> dict[str, Any]:
        return {"provider": self.model_client.provider, "model": self.model_client.model, "approval_policy": self.approval_policy, "tool_signature": registry.signature()}

    @classmethod
    def from_session(cls, workspace: str | Path, model_client: ModelClient, session_id: str = "latest", **kwargs: Any) -> "Tifa":
        context = WorkspaceContext.build(workspace); payload = SessionStore(context.repo_root).load(session_id)
        registry = ToolRegistry(context, kwargs.get("approval_policy", payload.get("runtime_identity", {}).get("approval_policy", "on-risk")), delegate=lambda *_: "")
        identity = {"provider": model_client.provider, "model": model_client.model, "approval_policy": registry.approval_policy, "tool_signature": registry.signature()}; expected = payload.get("runtime_identity", identity)
        mismatches = [k for k in identity if expected.get(k) != identity[k]]
        if payload.get("workspace_fingerprint") and payload["workspace_fingerprint"] != context.fingerprint(): mismatches.append("workspace_fingerprint")
        if mismatches: raise ResumeMismatch(f"resume rejected due to mismatch: {', '.join(sorted(set(mismatches)))}")
        return cls(context, model_client, session_id=payload.get("session_id"), memory=LayeredMemory(payload.get("memory")), history=payload.get("history", []), messages=payload.get("messages", []), completed_fingerprints=set(payload.get("completed_fingerprints", [])), imported_from=payload.get("legacy_source"), **kwargs)

    @classmethod
    def resume_run(cls, workspace: str | Path, model_client: ModelClient, run_id: str, checkpoint_id: str | None = None, *, allow_snapshot_copy: bool = False, runtime_overrides: set[str] | None = None, **kwargs: Any) -> "Tifa":
        context = WorkspaceContext.build(workspace); cp_dir = context.repo_root / ".tifa" / "runs" / run_id / "checkpoints"
        candidates = sorted(cp_dir.glob("*.json"))
        if not candidates: raise FileNotFoundError("no checkpoint found")
        if checkpoint_id:
            path = cp_dir / f"{checkpoint_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["state"].get("phase") not in {"MODEL_PENDING", "TOOL_COMMITTED"}: raise ResumeMismatch("checkpoint is not a safe commit point")
        else:
            safe = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in candidates]
            safe = [(p, v) for p, v in safe if v["state"].get("phase") in {"MODEL_PENDING", "TOOL_COMMITTED"}]
            if not safe: raise ResumeMismatch("no safe checkpoint found")
            path, payload = safe[-1]
        state = payload["state"]
        if payload.get("state_digest") != digest(state): raise ResumeMismatch("checkpoint digest mismatch")
        registry = ToolRegistry(context, kwargs.get("approval_policy", payload["runtime_identity"]["approval_policy"]), delegate=lambda *_: "")
        current = {"provider": model_client.provider, "model": model_client.model, "approval_policy": registry.approval_policy, "tool_signature": registry.signature()}
        overrides = runtime_overrides or set()
        mismatches = [key for key in current if current[key] != payload["runtime_identity"].get(key) and key not in overrides]
        if mismatches: raise ResumeMismatch(f"runtime identity mismatch: {', '.join(mismatches)}")
        if not allow_snapshot_copy and payload.get("workspace_fingerprint") != context.fingerprint(): raise ResumeMismatch("workspace fingerprint mismatch")
        return cls(context, model_client, session_id=state["session_id"], memory=LayeredMemory(payload.get("memory")), history=payload.get("history", []), messages=payload.get("messages", []), completed_fingerprints=set(payload.get("completed_fingerprints", [])), parent_run_id=run_id, source_checkpoint_id=payload["checkpoint_id"], **kwargs)

    def _checkpoint(self, store: RunStore, state: dict[str, Any], registry: ToolRegistry, parent: str | None = None) -> dict[str, Any]:
        checkpoint_id = f"cp-{store.sequence:06d}-{state['phase'].lower()}"
        payload = {"checkpoint_id": checkpoint_id, "parent_checkpoint_id": parent, "event_sequence": store.sequence, "state": state, "messages": self.messages, "history": self.history, "memory": self.memory.state, "completed_fingerprints": sorted(self.completed_fingerprints), "workspace_fingerprint": WorkspaceContext.build(self.workspace.repo_root).fingerprint(), "runtime_identity": self.runtime_identity(registry)}
        payload["state_digest"] = digest(payload["state"]); store.write(f"checkpoints/{checkpoint_id}.json", "tifa-checkpoint.v2", payload); return payload

    def ask(self, request: str, *, verifier: dict[str, Any] | None = None, interrupt_at: str | None = None, case_store: CaseStore | None = None, case_category: str | None = None) -> AgentResult:
        budget = 6000 if self.context_policy == "minimal" else 20000 if self.context_policy == "expanded" else 12000
        registry, store, context = self._registry(), RunStore(self.workspace.repo_root), ContextManager(self.workspace, total_budget=budget)
        state: dict[str, Any] = {"run_id": store.run_id, "session_id": self.session_id, "parent_run_id": self.parent_run_id, "source_checkpoint_id": self.source_checkpoint_id, "status": "running", "phase": "MODEL_PENDING", "request": request, "tool_steps": 0, "attempts": 0, "stop_reason": None, "affected_paths": [], "started_at": now()}
        if not self.messages: self.messages.append({"role": "user", "content": request})
        store.write("task_state.json", "tifa-task-state.v2", state); store.append_trace("run_started", {"request": request, "parent_run_id": self.parent_run_id, "source_checkpoint_id": self.source_checkpoint_id})
        checkpoints: list[dict[str, Any]] = []; last_cp = None
        cp = self._checkpoint(store, state, registry, last_cp); checkpoints.append(cp); last_cp = cp["checkpoint_id"]
        if interrupt_at == "MODEL_PENDING": raise InterruptedError("injected MODEL_PENDING interruption")
        answer, stop_reason, failure_category = "", "retry_limit_reached", None
        if not self.memory_enabled: self.memory = LayeredMemory()
        selected_cases = case_store.search(case_category) if case_store and case_category else []
        relevant_cases = [c.summary for c in selected_cases if c.summary]
        context_manifests: list[dict[str, Any]] = []; usage: list[dict[str, Any]] = []; artifacts: dict[str, dict[str, Any]] = {}; security_events: list[dict[str, Any]] = []; seen_call_ids: set[str] = set()
        while state["tool_steps"] < self.max_steps and state["attempts"] < self.max_attempts:
            state["attempts"] += 1; built = context.build(request, self.memory, self.history, registry.schemas(), relevant_cases); context_manifests.append({**built.metadata, "case_ids": [c.case_id for c in selected_cases]})
            try: response = self.model_client.complete(built.prompt, registry.schemas(), built.cache_key, self.messages)
            except ProviderError as exc: failure_category = exc.category; stop_reason = "provider_error"; answer = f"Tifa stopped: {exc.category}."; break
            usage.append(response.usage); store.append_trace("model_response", {"attempt": state["attempts"], "text": response.text, "tool_calls": [asdict(c) for c in response.tool_calls], "usage": response.usage, "cache": response.cache, "raw_response_ref": response.raw_response_ref})
            calls = response.tool_calls
            if not calls:
                kind, payload = parse(response.text)
                if kind == "final": answer, stop_reason = str(payload), "final_answer_returned"; break
                if kind == "tool": calls = [ToolCall(f"legacy-{state['attempts']}", payload["name"], payload.get("arguments", {}))]
                else: self.history.append({"role": "system", "content": payload}); store.append_trace("retry", {"reason": payload}); continue
            self.messages.append({"role": "assistant", "content": response.text, "tool_calls": [asdict(c) for c in calls]})
            for call in calls:
                if not call.id or call.id in seen_call_ids: failure_category = "duplicate_tool_call_id"; security_events.append({"type": failure_category, "call_id": call.id}); continue
                seen_call_ids.add(call.id); state["phase"] = "TOOL_PENDING"; cp = self._checkpoint(store, state, registry, last_cp); checkpoints.append(cp); last_cp = cp["checkpoint_id"]
                if interrupt_at == "TOOL_PENDING": raise InterruptedError("injected TOOL_PENDING interruption")
                fingerprint = call_fingerprint(call.name, call.arguments); path_arg = call.arguments.get("path"); target = self.workspace.resolve(path_arg) if path_arg else None
                before_digest = file_digest(target) if target and target.is_file() else None
                store.append_trace("tool_prepare", {"call_id": call.id, "name": call.name, "arguments": call.arguments, "fingerprint": fingerprint, "before_digest": before_digest}); state["phase"] = "TOOL_RUNNING"
                if interrupt_at == "TOOL_RUNNING": raise InterruptedError("injected TOOL_RUNNING interruption")
                if fingerprint in self.completed_fingerprints: output, affected, ok = "reused committed tool result", [], True
                else:
                    try: output, affected = registry.run(call.name, call.arguments); ok = True; self.completed_fingerprints.add(fingerprint); self.memory.after_tool(call.name, call.arguments, output)
                    except Exception as exc: output, affected, ok = f"error: {type(exc).__name__}: {exc}", [], False; failure_category = "tool_error"
                after_digest = file_digest(target) if target and target.is_file() else None
                state["tool_steps"] += 1; state["affected_paths"] = sorted(set(state["affected_paths"] + affected)); state["phase"] = "TOOL_COMMITTED"
                commit = {"call_id": call.id, "name": call.name, "arguments": call.arguments, "output": output, "ok": ok, "affected_paths": affected, "fingerprint": fingerprint, "before_digest": before_digest, "after_digest": after_digest}
                store.append_trace("tool_commit", commit); self.history.append({"role": "tool", "tool": call.name, "arguments": call.arguments, "content": output, "ok": ok}); self.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                for rel in affected:
                    path = self.workspace.resolve(rel)
                    if path.is_file():
                        content = path.read_text(encoding="utf-8", errors="replace"); artifacts[rel] = {"path": rel, "digest": digest(content), "status": "modified", "content": content}
                cp = self._checkpoint(store, state, registry, last_cp); checkpoints.append(cp); last_cp = cp["checkpoint_id"]
                if interrupt_at == "TOOL_COMMITTED": raise InterruptedError("injected TOOL_COMMITTED interruption")
                state["phase"] = "MODEL_PENDING"
        else:
            stop_reason = "step_limit_reached" if state["tool_steps"] >= self.max_steps else "retry_limit_reached"; answer = f"Tifa stopped: {stop_reason}."
        state["phase"] = "FINALIZING"
        if interrupt_at == "FINALIZING": raise InterruptedError("injected FINALIZING interruption")
        if interrupt_at == "VERIFIER_PENDING": raise InterruptedError("injected VERIFIER_PENDING interruption")
        verification = verify_contract(self.workspace.repo_root, verifier)
        state.update({"status": "finished", "stop_reason": stop_reason, "failure_category": failure_category or verification.get("failure_category"), "finished_at": now()})
        report = {"run_id": store.run_id, "session_id": self.session_id, "parent_run_id": self.parent_run_id, "source_checkpoint_id": self.source_checkpoint_id, "answer": answer, "stop_reason": stop_reason, "failure_category": state["failure_category"], "tool_steps": state["tool_steps"], "attempts": state["attempts"], "affected_paths": state["affected_paths"], "verifier": verification}
        store.write("task_state.json", "tifa-task-state.v2", state); store.write("report.json", "tifa-report.v2", report)
        state_event = store.append_trace("state_patch", state)
        for artifact in artifacts.values(): store.append_trace("artifact", artifact)
        report_event = store.append_trace("report", report); store.append_trace("verifier", verification)
        events = [json.loads(line) for line in (store.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        cp_refs = [{"checkpoint_id": c["checkpoint_id"], "event_sequence": c["event_sequence"], "state_digest": c["state_digest"]} for c in checkpoints]
        evidence = {"run_id": store.run_id, "created_at": now(), "parent_replay_id": self.parent_run_id, "task_contract": {"task_id": store.run_id, "prompt": request, "allowed_tools": registry.names, "step_budget": self.max_steps, "expected_artifact": None, "verifier": verifier}, "repo_snapshot": {"snapshot_id": self.workspace.fingerprint(), "commit": None, "dirty": bool(self.workspace.status), "workspace_digest": self.workspace.fingerprint()}, "context_manifest": {"policy": self.context_policy, "memory_enabled": self.memory_enabled, "selected_items": context_manifests, "dropped_items": [], "token_estimate": sum(u.get("input_tokens", u.get("input_estimate", 0)) for u in usage)}, "events": events, "checkpoints": cp_refs, "artifacts": [{k: v for k, v in a.items() if k != "content"} for a in artifacts.values()], "verifier": verification, "metrics": {"expected_state_digest": digest(state_event["payload"]), "expected_report_digest": digest(report_event["payload"]), "provider_usage": usage, "security_events": security_events}, "provenance": {"provider": self.model_client.provider, "model": self.model_client.model, "code_version": "0.3.0"}}
        store.write("evidence_bundle.json", "evidence-bundle.v2", evidence)
        session = {"session_id": self.session_id, "history": self.history, "messages": self.messages, "memory": self.memory.state, "completed_fingerprints": sorted(self.completed_fingerprints), "workspace_fingerprint": WorkspaceContext.build(self.workspace.repo_root).fingerprint(), "runtime_identity": self.runtime_identity(registry), "latest_run_id": store.run_id, "updated_at": now()}
        self.sessions.save(self.session_id, session); self.memory_store.save(self.memory)
        return AgentResult(answer, store.run_id, self.session_id, stop_reason, state["tool_steps"], state["attempts"], str(store.run_dir))


def build_agent(cwd: str | Path = ".", model_client: ModelClient | None = None, **kwargs: Any) -> Tifa:
    return Tifa(WorkspaceContext.build(cwd), model_client or FakeModelClient(), **kwargs)
