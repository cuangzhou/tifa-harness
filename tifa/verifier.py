from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .execution import ExecutionBackend, ExecutionPolicy, ExecutionRequest, LocalExecutionBackend, ResourceLimits, command_argv


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contract(root: Path, spec: dict[str, Any] | None, backend: ExecutionBackend | None = None) -> dict[str, Any]:
    if not spec: return {"status": "not_configured", "passed": None, "failure_category": None, "checks": []}
    checks: list[dict[str, Any]] = []
    for item in spec.get("files", []):
        path = (root / item["path"]).resolve()
        exists = path.is_file(); ok = exists and (not item.get("digest") or file_digest(path) == item["digest"])
        checks.append({"type": "file", "path": item["path"], "passed": ok})
    for item in spec.get("commands", []):
        limits = ResourceLimits(timeout_seconds=int(item.get("timeout", 30)))
        result = (backend or LocalExecutionBackend()).execute(ExecutionRequest(command_argv(item["command"], bool(item.get("allow_shell", False))), root, limits=limits, policy=ExecutionPolicy()))
        checks.append({"type": "command", "passed": result.exit_code == int(item.get("exit_code", 0)) and not result.timed_out, "exit_code": result.exit_code, "backend": result.backend, "isolation_level": result.isolation_level})
    for item in spec.get("assertions", []):
        actual = item.get("actual"); expected = item.get("equals")
        checks.append({"type": "assertion", "passed": actual == expected})
    passed = all(c["passed"] for c in checks)
    return {"status": "passed" if passed else "failed", "passed": passed, "failure_category": None if passed else "verifier_failed", "checks": checks}
