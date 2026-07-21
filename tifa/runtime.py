from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path
import re
from typing import Any, Callable
import uuid

from .context_manager import ContextManager
from .cases import CaseStore, tool_schema_digest
from .contracts import CompletionDecision, RepairFeedback, TaskContract, TaskPhase, default_plan
from .execution import ExecutionBackend, ExecutionPolicy, LocalExecutionBackend, ResourceLimits
from .memory import DurableMemoryStore, LayeredMemory
from .models import AgentResult, FakeModelClient, ModelClient, ToolCall
from .operations import RunLock
from .observability import emit_run_span
from .providers import ProviderError
from .replay import digest
from .stores import RunStore, SessionStore, now
from .tools import ToolArgumentError, ToolRegistry, call_fingerprint
from .verifier import file_digest, verify_contract
from .workspace import WorkspaceContext

TOOL_PATTERN = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL | re.IGNORECASE)
FINAL_PATTERN = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL | re.IGNORECASE)
JSON_TOOL_PATTERN = re.compile(r"(?:```(?:json)?\s*)?(\{\s*\"name\"\s*:.*\})(?:\s*```)?", re.DOTALL | re.IGNORECASE)
PHASES = {"MODEL_PENDING", "TOOL_PENDING", "TOOL_RUNNING", "TOOL_COMMITTED", "FINALIZING"}


@dataclass
class RunBudget:
    max_model_calls: int = 16
    max_input_tokens: int = 200_000
    max_output_tokens: int = 50_000
    max_tool_calls: int = 8
    max_duration_seconds: float = 600.0
    max_cost_usd: float | None = None


@dataclass(frozen=True)
class HarnessControls:
    immediate_verifier: bool = True
    structured_recovery: bool = True
    side_effect_governance: bool = True
    context_memory_checkpoint: bool = True


def parse(content: str) -> tuple[str, Any]:
    final = FINAL_PATTERN.search(content)
    if final: return "final", final.group(1)
    tool = TOOL_PATTERN.search(content) or JSON_TOOL_PATTERN.search(content)
    if tool:
        try:
            payload = json.loads(tool.group(1))
            if not isinstance(payload.get("name"), str) or not isinstance(payload.get("arguments", {}), dict): raise ValueError
            return "tool", payload
        except (json.JSONDecodeError, ValueError): pass
    return "retry", "Model output must contain a valid tool call or <final> answer."


class ResumeMismatch(RuntimeError): pass


class Tifa:
    def __init__(self, workspace: WorkspaceContext, model_client: ModelClient, *, max_steps: int = 8, max_attempts: int = 16, approval_policy: str = "on-risk", approver: Callable[[str, dict[str, Any]], bool] | None = None, session_id: str | None = None, memory: LayeredMemory | None = None, history: list[dict[str, Any]] | None = None, messages: list[dict[str, Any]] | None = None, completed_fingerprints: set[str] | None = None, imported_from: str | None = None, parent_run_id: str | None = None, source_checkpoint_id: str | None = None, context_policy: str = "layered-budget-v2", memory_enabled: bool = True, execution_backend: ExecutionBackend | None = None, execution_policy: ExecutionPolicy | None = None, resource_limits: ResourceLimits | None = None, run_budget: RunBudget | None = None, harness_controls: HarnessControls | None = None) -> None:
        self.workspace, self.model_client, self.max_steps, self.max_attempts = workspace, model_client, max_steps, max_attempts
        self.approval_policy, self.approver = approval_policy, approver
        self.session_id, self.memory, self.history = session_id or uuid.uuid4().hex, memory or LayeredMemory(), history or []
        self.messages, self.completed_fingerprints = messages or [], completed_fingerprints or set()
        self.imported_from, self.parent_run_id, self.source_checkpoint_id = imported_from, parent_run_id, source_checkpoint_id
        self.context_policy, self.memory_enabled = context_policy, memory_enabled
        self.harness_controls = harness_controls or HarnessControls()
        if not self.harness_controls.context_memory_checkpoint:
            self.memory_enabled = False
        if not self.harness_controls.structured_recovery:
            self.workspace.normalize_paths = False
        self._registry_depth = 0
        self.execution_backend = execution_backend or LocalExecutionBackend(); self.execution_policy = execution_policy or ExecutionPolicy(); self.resource_limits = resource_limits or ResourceLimits(); self.run_budget = run_budget or RunBudget(max_model_calls=max_attempts, max_tool_calls=max_steps)
        self.sessions = SessionStore(workspace.repo_root); self.memory_store = DurableMemoryStore(workspace.repo_root / ".tifa" / "memory" / "memory.json")

    def _delegate(self, task: str, steps: int) -> str:
        child = Tifa(self.workspace, self.model_client, max_steps=steps, max_attempts=max(steps * 2, 3), approval_policy="never", memory=LayeredMemory(self.memory.state), execution_backend=self.execution_backend, execution_policy=self.execution_policy, resource_limits=self.resource_limits, run_budget=self.run_budget); child._registry_depth = 1
        return child.ask(task).answer

    def _registry(self) -> ToolRegistry:
        contract = getattr(self, "_active_contract", None)
        return ToolRegistry(self.workspace, self.approval_policy, self.approver, self._delegate, getattr(self, "_registry_depth", 0), 1, self.execution_backend, self.resource_limits, self.execution_policy, set(contract.allowed_tools) if contract and contract.allowed_tools else None, contract.writable_paths if contract else None)

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
        context = WorkspaceContext.build(workspace); lock = RunLock(context.repo_root / ".tifa" / "locks" / f"{run_id}.lock").acquire()
        try: return cls._resume_unlocked(workspace, model_client, run_id, checkpoint_id, allow_snapshot_copy=allow_snapshot_copy, runtime_overrides=runtime_overrides, **kwargs)
        finally: lock.release()

    @classmethod
    def _resume_unlocked(cls, workspace: str | Path, model_client: ModelClient, run_id: str, checkpoint_id: str | None = None, *, allow_snapshot_copy: bool = False, runtime_overrides: set[str] | None = None, **kwargs: Any) -> "Tifa":
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

    def ask(self, request: str, *, verifier: dict[str, Any] | None = None, contract: TaskContract | None = None, interrupt_at: str | None = None, case_store: CaseStore | None = None, case_category: str | None = None) -> AgentResult:
        self._active_contract = contract
        if contract:
            request = contract.goal
            verifier = contract.verifier
            if contract.max_steps is not None: self.max_steps = min(self.max_steps, contract.max_steps)
        budget = 6000 if self.context_policy == "minimal" else 20000 if self.context_policy == "expanded" else 12000
        registry, store, context = self._registry(), RunStore(self.workspace.repo_root), ContextManager(self.workspace, total_budget=budget)
        state: dict[str, Any] = {"run_id": store.run_id, "session_id": self.session_id, "parent_run_id": self.parent_run_id, "source_checkpoint_id": self.source_checkpoint_id, "status": "running", "phase": "MODEL_PENDING", "task_phase": TaskPhase.UNDERSTAND.value, "request": request, "tool_steps": 0, "attempts": 0, "repairs": 0, "stop_reason": None, "affected_paths": [], "started_at": now()}
        store.append_log("info", "run_started", {"session_id": self.session_id, "provider": self.model_client.provider, "model": self.model_client.model})
        if not self.messages: self.messages.append({"role": "user", "content": request})
        store.write("task_state.json", "tifa-task-state.v3", state); store.append_trace("run_started", {"request": request, "parent_run_id": self.parent_run_id, "source_checkpoint_id": self.source_checkpoint_id, "contract_id": contract.contract_id if contract else None})
        state["task_phase"] = TaskPhase.PLAN.value; plan = default_plan(contract or TaskContract(request, require_verifier=False)); store.append_trace("execution_plan", asdict(plan)); state["task_phase"] = TaskPhase.EXECUTE.value
        checkpoints: list[dict[str, Any]] = []; last_cp = None
        if self.harness_controls.context_memory_checkpoint:
            cp = self._checkpoint(store, state, registry, last_cp); checkpoints.append(cp); last_cp = cp["checkpoint_id"]
        if interrupt_at == "MODEL_PENDING": raise InterruptedError("injected MODEL_PENDING interruption")
        answer, stop_reason, failure_category = "", "retry_limit_reached", None
        if not self.memory_enabled: self.memory = LayeredMemory()
        candidate_cases = case_store.list() if case_store and case_category else []
        dependency_state = {path: file_digest(self.workspace.resolve(path)) or "" for card in candidate_cases for path in card.dependency_digests if self.workspace.resolve(path).is_file()}
        selected_cases = case_store.search(case_category, repo_snapshot=self.workspace.fingerprint(), dependency_digests=dependency_state, tool_schema_digest=tool_schema_digest(registry.schemas()), context_policy_version=self.context_policy) if case_store and case_category else []
        stale_case_ids = [card.case_id for card in candidate_cases if card.freshness_status == "stale"]
        relevant_cases = [json.dumps({"summary": card.summary, "evidence_refs": card.evidence_refs}, ensure_ascii=False) for card in selected_cases if card.summary]
        context_manifests: list[dict[str, Any]] = []; usage: list[dict[str, Any]] = []; artifacts: dict[str, dict[str, Any]] = {}; security_events: list[dict[str, Any]] = []; execution_events: list[dict[str, Any]] = []; seen_call_ids: set[str] = set(); run_started = time.perf_counter()
        budget_usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "cost_usd": 0.0}; fingerprint_counts: dict[str, int] = {}; failure_counts: dict[str, int] = {}
        harness_metrics = {"verified_after_tool": 0, "tool_batch_truncated": 0, "schema_errors": 0, "schema_recoveries": 0, "path_errors": 0, "path_recoveries": 0}
        pending_recovery: str | None = None
        while state["tool_steps"] < self.max_steps and state["attempts"] < self.max_attempts:
            if budget_usage["model_calls"] >= self.run_budget.max_model_calls or (time.perf_counter() - run_started) > self.run_budget.max_duration_seconds:
                failure_category = "budget_exceeded"; stop_reason = "budget_exceeded"; answer = "Tifa stopped: run budget exceeded."; break
            state["attempts"] += 1
            if self.harness_controls.context_memory_checkpoint:
                built = context.build(request, self.memory, self.history, registry.schemas(), relevant_cases); context_manifests.append({**built.metadata, "candidate_case_ids": [c.case_id for c in candidate_cases], "selected_case_ids": [c.case_id for c in selected_cases], "stale_case_ids": stale_case_ids, "dropped_case_ids": [c.case_id for c in candidate_cases if c not in selected_cases]})
                model_prompt, cache_key = built.prompt, built.cache_key
            else:
                model_prompt, cache_key = request, None; context_manifests.append({"policy": "disabled_ablation", "selected_items": [], "dropped_items": []})
            try: response = self.model_client.complete(model_prompt, registry.schemas(), cache_key, self.messages)
            except ProviderError as exc: failure_category = exc.category; stop_reason = "provider_error"; answer = f"Tifa stopped: {exc.category}."; break
            usage.append(response.usage); budget_usage["model_calls"] += 1; budget_usage["input_tokens"] += int(response.usage.get("input_tokens", response.usage.get("prompt_tokens", response.usage.get("prompt_eval_count", response.usage.get("input_estimate", 0))))); budget_usage["output_tokens"] += int(response.usage.get("output_tokens", response.usage.get("completion_tokens", response.usage.get("eval_count", response.usage.get("output_estimate", 0))))); budget_usage["cost_usd"] += float(response.usage.get("cost_usd", 0.0))
            if budget_usage["input_tokens"] > self.run_budget.max_input_tokens or budget_usage["output_tokens"] > self.run_budget.max_output_tokens or (self.run_budget.max_cost_usd is not None and budget_usage["cost_usd"] > self.run_budget.max_cost_usd): failure_category = "budget_exceeded"; stop_reason = "budget_exceeded"; answer = "Tifa stopped: token or cost budget exceeded."; break
            store.append_trace("model_response", {"attempt": state["attempts"], "text": response.text, "tool_calls": [asdict(c) for c in response.tool_calls], "usage": response.usage, "cache": response.cache, "raw_response_ref": response.raw_response_ref})
            calls = response.tool_calls
            if not calls:
                kind, payload = parse(response.text)
                if kind == "final" or (kind == "retry" and self.model_client.provider != "fake" and response.text.strip()):
                    candidate = response.text.strip() if kind == "retry" else str(payload); state["task_phase"] = TaskPhase.VERIFY.value
                    gate = verify_contract(self.workspace.repo_root, verifier, self.execution_backend, state["affected_paths"])
                    gate_optional = gate.get("status") == "not_configured" and (not contract or not contract.require_verifier)
                    decision = CompletionDecision(gate.get("passed") is True or gate_optional, "verifier passed" if gate.get("passed") is True else "verifier optional" if gate_optional else "completion gate not satisfied", gate["status"])
                    store.append_trace("completion_decision", asdict(decision))
                    if decision.complete: answer, stop_reason, failure_category, state["task_phase"] = candidate, "final_answer_returned", None, TaskPhase.COMPLETE.value; break
                    if contract and state["repairs"] < contract.max_repairs:
                        state["repairs"] += 1; state["task_phase"] = TaskPhase.REPAIR.value
                        feedback = RepairFeedback(state["repairs"], gate.get("failure_category") or "verifier_not_configured", decision.reason, [check for check in gate.get("checks", []) if not check.get("passed")])
                        self.messages.append({"role": "user", "content": "Repair the task using this verifier feedback: " + json.dumps(asdict(feedback), ensure_ascii=False)}); store.append_trace("repair_feedback", asdict(feedback)); state["task_phase"] = TaskPhase.EXECUTE.value; continue
                    answer, stop_reason, failure_category, state["task_phase"] = "Tifa stopped: completion gate failed.", "completion_gate_failed", gate.get("failure_category") or "verifier_not_configured", TaskPhase.FAILED.value; break
                if kind == "tool": calls = [ToolCall(f"legacy-{state['attempts']}", payload["name"], payload.get("arguments", {}))]
                else: self.history.append({"role": "system", "content": payload}); store.append_trace("retry", {"reason": payload}); continue
            remaining_steps = min(self.max_steps - state["tool_steps"], self.run_budget.max_tool_calls - budget_usage["tool_calls"])
            executable_calls, truncated_calls = calls[:max(0, remaining_steps)], calls[max(0, remaining_steps):]
            if truncated_calls:
                harness_metrics["tool_batch_truncated"] += len(truncated_calls)
                store.append_trace("tool_batch_truncated", {"remaining_steps": remaining_steps, "skipped": [{"id": call.id, "name": call.name} for call in truncated_calls]})
            calls = executable_calls
            self.messages.append({"role": "assistant", "content": response.text, "tool_calls": [asdict(c) for c in [*calls, *truncated_calls]]})
            deferred_feedback: list[str] = []
            unexecuted_after_verification: list[ToolCall] = []
            for call_index, call in enumerate(calls):
                affected: list[str]
                if not call.id or call.id in seen_call_ids:
                    failure_category = "duplicate_tool_call_id"; security_events.append({"type": failure_category, "call_id": call.id})
                    self.messages.append({"role": "tool", "tool_call_id": call.id, "content": "error: duplicate tool call id rejected"})
                    continue
                seen_call_ids.add(call.id); state["phase"] = "TOOL_PENDING"
                if self.harness_controls.context_memory_checkpoint:
                    cp = self._checkpoint(store, state, registry, last_cp); checkpoints.append(cp); last_cp = cp["checkpoint_id"]
                if interrupt_at == "TOOL_PENDING": raise InterruptedError("injected TOOL_PENDING interruption")
                fingerprint = call_fingerprint(call.name, call.arguments); path_arg = call.arguments.get("path")
                fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1
                if self.harness_controls.side_effect_governance and fingerprint_counts[fingerprint] > 2:
                    output, affected, ok = "error: repeated identical tool call blocked; choose a different action and complete the task contract", [], False; failure_category = "loop_detected"; store.append_trace("loop_detected", {"fingerprint": fingerprint, "name": call.name}); self.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                    if contract and state["repairs"] < contract.max_repairs:
                        state["repairs"] += 1; fingerprint_counts[fingerprint] = 0; feedback = RepairFeedback(state["repairs"], "loop_detected", f"Do not repeat {call.name} with the same arguments. Use a different tool or arguments, produce the required artifacts, then request verification.", [])
                        deferred_feedback.append(json.dumps(asdict(feedback), ensure_ascii=False)); store.append_trace("repair_feedback", asdict(feedback)); state["task_phase"] = TaskPhase.REPAIR.value
                    continue
                try: target = self.workspace.resolve(path_arg) if path_arg else None
                except ValueError: target = None
                before_digest = file_digest(target) if target and target.is_file() else None
                store.append_trace("tool_prepare", {"call_id": call.id, "name": call.name, "arguments": call.arguments, "fingerprint": fingerprint, "before_digest": before_digest}); state["phase"] = "TOOL_RUNNING"
                if interrupt_at == "TOOL_RUNNING": raise InterruptedError("injected TOOL_RUNNING interruption")
                if self.harness_controls.side_effect_governance and fingerprint in self.completed_fingerprints: output, affected, ok = "reused committed tool result", [], True
                else:
                    try:
                        registry.last_execution = None; output, affected = registry.run(call.name, call.arguments); ok = True; self.completed_fingerprints.add(fingerprint)
                        if self.harness_controls.context_memory_checkpoint: self.memory.after_tool(call.name, call.arguments, output)
                        if pending_recovery == "schema_error": harness_metrics["schema_recoveries"] += 1
                        if pending_recovery == "path_error": harness_metrics["path_recoveries"] += 1
                        pending_recovery = None
                        if registry.last_execution: execution_events.append(asdict(registry.last_execution)); security_events.extend({"type": event, "tool": call.name} for event in registry.last_execution.security_events)
                    except Exception as exc:
                        category = "schema_error" if isinstance(exc, ToolArgumentError) else "path_error" if "path" in str(exc).lower() else "approval_denied" if isinstance(exc, PermissionError) else "invalid_arguments" if isinstance(exc, (ValueError, KeyError, TypeError)) else "tool_timeout" if isinstance(exc, TimeoutError) else "environment_error"
                        output, affected, ok = f"error: {category}: {exc}", [], False; failure_category = category; failure_counts[category] = failure_counts.get(category, 0) + 1
                        if category == "schema_error": harness_metrics["schema_errors"] += 1
                        if category == "path_error": harness_metrics["path_errors"] += 1
                        pending_recovery = category if self.harness_controls.structured_recovery and category in {"schema_error", "path_error", "invalid_arguments"} else None
                        if pending_recovery and contract and state["repairs"] < contract.max_repairs:
                            state["repairs"] += 1
                            feedback = RepairFeedback(state["repairs"], category, str(exc), [{"tool": call.name, "arguments": call.arguments, "remaining_tool_steps": min(self.max_steps - state["tool_steps"] - 1, self.run_budget.max_tool_calls - budget_usage["tool_calls"] - 1)}])
                            deferred_feedback.append("Correct the failed tool call using this exact schema/path/command feedback. Do not repeat unchanged arguments: " + json.dumps(asdict(feedback), ensure_ascii=False))
                            store.append_trace("repair_feedback", asdict(feedback)); state["task_phase"] = TaskPhase.REPAIR.value
                        if failure_counts[category] >= 3: security_events.append({"type": "repeated_tool_failure", "category": category})
                budget_usage["tool_calls"] += 1
                if budget_usage["tool_calls"] > self.run_budget.max_tool_calls: failure_category = "budget_exceeded"; stop_reason = "budget_exceeded"; answer = "Tifa stopped: tool-call budget exceeded."
                after_digest = file_digest(target) if target and target.is_file() else None
                state["tool_steps"] += 1; state["affected_paths"] = sorted(set(state["affected_paths"] + affected)); state["phase"] = "TOOL_COMMITTED"
                commit = {"call_id": call.id, "name": call.name, "arguments": call.arguments, "output": output, "ok": ok, "affected_paths": affected, "fingerprint": fingerprint, "before_digest": before_digest, "after_digest": after_digest}
                store.append_trace("tool_commit", commit); self.history.append({"role": "tool", "tool": call.name, "arguments": call.arguments, "content": output, "ok": ok}); self.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                for rel in affected:
                    path = self.workspace.resolve(rel)
                    if path.is_file():
                        content = path.read_text(encoding="utf-8", errors="replace"); artifacts[rel] = {"path": rel, "digest": digest(content), "status": "modified", "content": content}
                if self.harness_controls.context_memory_checkpoint:
                    cp = self._checkpoint(store, state, registry, last_cp); checkpoints.append(cp); last_cp = cp["checkpoint_id"]
                if interrupt_at == "TOOL_COMMITTED": raise InterruptedError("injected TOOL_COMMITTED interruption")
                state["phase"] = "MODEL_PENDING"
                if self.harness_controls.immediate_verifier and ok and contract and contract.require_verifier:
                    incremental_gate = verify_contract(self.workspace.repo_root, verifier, self.execution_backend, state["affected_paths"])
                    store.append_trace("incremental_verifier", incremental_gate)
                    if incremental_gate.get("passed") is True:
                        harness_metrics["verified_after_tool"] += 1
                        decision = CompletionDecision(True, "verifier passed after committed tool", incremental_gate["status"])
                        store.append_trace("completion_decision", asdict(decision))
                        answer, stop_reason, failure_category, state["task_phase"] = "Task contract verified after committed tool evidence.", "verified_after_tool", None, TaskPhase.COMPLETE.value
                        unexecuted_after_verification = calls[call_index + 1:]
                        break
                if stop_reason == "budget_exceeded": break
            for skipped in truncated_calls:
                self.messages.append({"role": "tool", "tool_call_id": skipped.id, "content": "error: tool call skipped because no execution budget remained"})
            for skipped in unexecuted_after_verification:
                self.messages.append({"role": "tool", "tool_call_id": skipped.id, "content": "skipped: task contract already verified"})
            # OpenAI-compatible protocols require every tool result for an
            # assistant batch to be contiguous. Add repair instructions only
            # after all results have been supplied.
            self.messages.extend({"role": "user", "content": item} for item in deferred_feedback)
            if stop_reason in {"budget_exceeded", "verified_after_tool"}: break
        else:
            stop_reason = "step_limit_reached" if state["tool_steps"] >= self.max_steps else "retry_limit_reached"; failure_category = failure_category or stop_reason; state["task_phase"] = TaskPhase.FAILED.value; answer = f"Tifa stopped: {stop_reason}."
        state["phase"] = "FINALIZING"
        if interrupt_at == "FINALIZING": raise InterruptedError("injected FINALIZING interruption")
        if interrupt_at == "VERIFIER_PENDING": raise InterruptedError("injected VERIFIER_PENDING interruption")
        verification = verify_contract(self.workspace.repo_root, verifier, self.execution_backend, state["affected_paths"])
        if verification.get("passed") is True and stop_reason in {"step_limit_reached", "retry_limit_reached"}:
            stop_reason, failure_category, state["task_phase"], answer = "verified_at_limit", None, TaskPhase.COMPLETE.value, answer or "Task contract verified at execution limit."
        state.update({"status": "finished", "stop_reason": stop_reason, "failure_category": failure_category or verification.get("failure_category"), "finished_at": now()})
        report = {"run_id": store.run_id, "session_id": self.session_id, "parent_run_id": self.parent_run_id, "source_checkpoint_id": self.source_checkpoint_id, "answer": answer, "stop_reason": stop_reason, "failure_category": state["failure_category"], "tool_steps": state["tool_steps"], "attempts": state["attempts"], "affected_paths": state["affected_paths"], "verifier": verification}
        store.write("task_state.json", "tifa-task-state.v3", state); store.write("report.json", "tifa-report.v3", report)
        state_event = store.append_trace("state_patch", state)
        for artifact in artifacts.values(): store.append_trace("artifact", artifact)
        report_event = store.append_trace("report", report); store.append_trace("verifier", verification)
        events = [json.loads(line) for line in (store.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        cp_refs = [{"checkpoint_id": c["checkpoint_id"], "event_sequence": c["event_sequence"], "state_digest": c["state_digest"]} for c in checkpoints]
        task_contract = asdict(contract) if contract else {"contract_id": store.run_id, "goal": request, "allowed_tools": registry.names, "writable_paths": [], "max_steps": self.max_steps, "max_repairs": 0, "require_verifier": False, "verifier": verifier}
        harness_metrics["normalized_paths"] = self.workspace.normalized_path_count
        evidence = {"run_id": store.run_id, "created_at": now(), "parent_replay_id": self.parent_run_id, "task_contract": task_contract, "execution_plan": asdict(plan), "repo_snapshot": {"snapshot_id": self.workspace.fingerprint(), "commit": None, "dirty": bool(self.workspace.status), "workspace_digest": self.workspace.fingerprint(), "tool_schema_digest": tool_schema_digest(registry.schemas())}, "context_manifest": {"policy": self.context_policy, "memory_enabled": self.memory_enabled, "selected_items": context_manifests, "dropped_items": [{"case_id": case_id, "reason": "stale_or_inapplicable"} for case_id in stale_case_ids], "token_estimate": sum(u.get("input_tokens", u.get("input_estimate", 0)) for u in usage)}, "events": events, "checkpoints": cp_refs, "artifacts": [{k: v for k, v in a.items() if k != "content"} for a in artifacts.values()], "verifier": verification, "metrics": {"expected_state_digest": digest(state_event["payload"]), "expected_report_digest": digest(report_event["payload"]), "provider_usage": usage, "budget_usage": budget_usage, "harness": harness_metrics, "execution_events": execution_events, "security_events": security_events}, "provenance": {"provider": self.model_client.provider, "model": self.model_client.model, "temperature": getattr(self.model_client, "temperature", None), "code_version": "0.5.0"}}
        store.write("evidence_bundle.json", "evidence-bundle.v3", evidence)
        store.write("metrics.json", "tifa-run-metrics.v1", {"run_id": store.run_id, "stop_reason": stop_reason, "failure_category": state["failure_category"], "budget_usage": budget_usage, "execution_event_count": len(execution_events), "security_event_count": len(security_events)})
        store.append_log("info", "run_finished", {"stop_reason": stop_reason, "failure_category": state["failure_category"], "tool_steps": state["tool_steps"], "attempts": state["attempts"]})
        emit_run_span(store.run_id, {"stop_reason": stop_reason, "failure_category": state["failure_category"] or "none", "tool_steps": state["tool_steps"], "attempts": state["attempts"]})
        session = {"session_id": self.session_id, "history": self.history, "messages": self.messages, "memory": self.memory.state, "completed_fingerprints": sorted(self.completed_fingerprints), "workspace_fingerprint": WorkspaceContext.build(self.workspace.repo_root).fingerprint(), "runtime_identity": self.runtime_identity(registry), "latest_run_id": store.run_id, "updated_at": now()}
        self.sessions.save(self.session_id, session); self.memory_store.save(self.memory)
        if state["failure_category"] and not self.parent_run_id:
            try: CaseStore(self.workspace.repo_root / ".tifa" / "cases").propose_from_run(self.workspace.repo_root, store.run_id)
            except (FileNotFoundError, ValueError): pass
        return AgentResult(answer, store.run_id, self.session_id, stop_reason, state["tool_steps"], state["attempts"], str(store.run_dir))


def build_agent(cwd: str | Path = ".", model_client: ModelClient | None = None, **kwargs: Any) -> Tifa:
    return Tifa(WorkspaceContext.build(cwd), model_client or FakeModelClient(), **kwargs)
