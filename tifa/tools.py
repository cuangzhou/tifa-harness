from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from .execution import ExecutionBackend, ExecutionPolicy, ExecutionRequest, LocalExecutionBackend, ResourceLimits, command_argv, result_text
from .workspace import WorkspaceContext

READ_ONLY = {"list_files", "read_file", "search", "delegate"}
HIGH_RISK = {"run_shell", "write_file", "patch_file"}


def tool_spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


SPECS = {
    "list_files": tool_spec("list_files", "List files under a workspace directory", {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, ["path"]),
    "read_file": tool_spec("read_file", "Read a UTF-8 text file", {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"]),
    "search": tool_spec("search", "Search text in workspace files", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    "run_shell": tool_spec("run_shell", "Run a command in the workspace", {"command": {"type": "string"}, "timeout": {"type": "integer", "minimum": 1, "maximum": 120}}, ["command"]),
    "write_file": tool_spec("write_file", "Write a UTF-8 file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "patch_file": tool_spec("patch_file", "Replace one unique text occurrence", {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
    "delegate": tool_spec("delegate", "Run a bounded read-only investigation", {"task": {"type": "string"}, "max_steps": {"type": "integer", "minimum": 1, "maximum": 3}}, ["task"]),
}


def call_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps([name, arguments], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ToolRegistry:
    def __init__(self, workspace: WorkspaceContext, approval_policy: str = "on-risk", approver: Callable[[str, dict[str, Any]], bool] | None = None, delegate: Callable[[str, int], str] | None = None, depth: int = 0, max_depth: int = 1, execution_backend: ExecutionBackend | None = None, resource_limits: ResourceLimits | None = None, execution_policy: ExecutionPolicy | None = None) -> None:
        if approval_policy not in {"never", "on-risk", "always"}:
            raise ValueError("invalid approval policy")
        self.workspace, self.approval_policy, self.approver = workspace, approval_policy, approver
        self.delegate_fn, self.depth, self.max_depth = delegate, depth, max_depth
        self.execution_backend = execution_backend or LocalExecutionBackend(); self.resource_limits = resource_limits or ResourceLimits(); self.execution_policy = execution_policy or ExecutionPolicy()
        self.last_execution = None

    @property
    def names(self) -> list[str]:
        names = list(SPECS)
        if self.depth > 0:
            return [name for name in names if name in READ_ONLY and name != "delegate"]
        if self.depth >= self.max_depth or not self.delegate_fn:
            names.remove("delegate")
        return names

    def schemas(self) -> list[dict[str, Any]]:
        return [SPECS[name] for name in self.names]

    def signature(self) -> str:
        return hashlib.sha256(json.dumps(self.schemas(), sort_keys=True).encode()).hexdigest()

    def _approve(self, name: str, arguments: dict[str, Any]) -> None:
        needed = self.approval_policy == "always" or (self.approval_policy == "on-risk" and name in HIGH_RISK)
        if needed and (not self.approver or not self.approver(name, arguments)):
            raise PermissionError(f"approval denied for {name}")

    def run(self, name: str, args: dict[str, Any]) -> tuple[str, list[str]]:
        if name not in self.names:
            raise ValueError(f"unknown or unavailable tool: {name}")
        self._approve(name, args)
        affected: list[str] = []
        if name == "list_files":
            path = self.workspace.resolve(str(args.get("path", ".")), must_exist=True)
            if not path.is_dir(): raise ValueError("list_files target must be a directory")
            items = path.rglob("*") if args.get("recursive") else path.iterdir()
            return "\n".join(str(p.relative_to(self.workspace.repo_root)) for p in list(items)[:500]), affected
        if name == "read_file":
            path = self.workspace.resolve(str(args["path"]), must_exist=True)
            if not path.is_file(): raise ValueError("read_file target must be a file")
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start, end = int(args.get("start", 1)), int(args.get("end", len(lines)))
            if start < 1 or end < start: raise ValueError("invalid line range")
            return "\n".join(lines[start - 1:end])[:30000], affected
        if name == "search":
            pattern = str(args.get("pattern", ""))
            if not pattern: raise ValueError("search pattern cannot be empty")
            root = self.workspace.resolve(str(args.get("path", ".")), must_exist=True)
            regex = re.compile(pattern)
            matches: list[str] = []
            files = [root] if root.is_file() else root.rglob("*")
            for path in files:
                if path.is_file() and ".tifa" not in path.parts:
                    try:
                        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                            if regex.search(line): matches.append(f"{path.relative_to(self.workspace.repo_root)}:{number}:{line[:300]}")
                    except (UnicodeError, OSError): pass
                if len(matches) >= 500: break
            return "\n".join(matches), affected
        if name == "run_shell":
            command, timeout = str(args.get("command", "")), int(args.get("timeout", 30))
            if not command or not 1 <= timeout <= 120: raise ValueError("invalid shell command or timeout")
            limits = ResourceLimits(self.resource_limits.cpus, self.resource_limits.memory_mb, self.resource_limits.pids, timeout)
            self.last_execution = self.execution_backend.execute(ExecutionRequest(command_argv(command, self.execution_policy.allow_shell), self.workspace.repo_root, limits=limits, policy=self.execution_policy))
            return result_text(self.last_execution), affected
        if name == "write_file":
            path = self.workspace.resolve(str(args["path"])); path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.is_dir(): raise ValueError("write_file target is a directory")
            path.write_text(str(args["content"]), encoding="utf-8"); affected.append(str(path.relative_to(self.workspace.repo_root)))
            return f"wrote {affected[0]}", affected
        if name == "patch_file":
            path = self.workspace.resolve(str(args["path"]), must_exist=True)
            old, new = str(args["old_text"]), str(args["new_text"])
            content = path.read_text(encoding="utf-8")
            if not old or content.count(old) != 1: raise ValueError("old_text must occur exactly once")
            path.write_text(content.replace(old, new, 1), encoding="utf-8"); affected.append(str(path.relative_to(self.workspace.repo_root)))
            return f"patched {affected[0]}", affected
        task, steps = str(args.get("task", "")), int(args.get("max_steps", 3))
        if not task or not 1 <= steps <= 3: raise ValueError("invalid delegate task")
        return self.delegate_fn(task, steps), affected  # type: ignore[misc]
