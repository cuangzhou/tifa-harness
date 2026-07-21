from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any

from .execution import ExecutionBackend, ExecutionPolicy, ExecutionRequest, LocalExecutionBackend, ResourceLimits, command_argv


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_check(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = (root / item["path"]).resolve()
    try: path.relative_to(root.resolve())
    except ValueError: return {"type": "file", "path": item["path"], "passed": False, "reason": "path_escape"}
    exists = path.is_file(); ok = exists; reason = None
    content = path.read_text(encoding="utf-8", errors="replace") if exists else ""
    if item.get("digest"): ok = ok and file_digest(path) == item["digest"]
    if "contains" in item: ok = ok and str(item["contains"]) in content
    if "equals" in item: ok = ok and content == str(item["equals"])
    if item.get("regex"):
        try: ok = ok and re.search(str(item["regex"]), content, re.MULTILINE) is not None
        except re.error: ok, reason = False, "invalid_regex"
    return {"type": "file", "path": item["path"], "passed": ok, **({"reason": reason} if reason else {})}


def _ast_check(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = (root / item["path"]).resolve()
    try: tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as exc: return {"type": "ast", "path": item["path"], "passed": False, "reason": type(exc).__name__}
    kind = item.get("kind", "function"); name = item.get("name")
    types = {"function": (ast.FunctionDef, ast.AsyncFunctionDef), "class": (ast.ClassDef,), "import": (ast.Import, ast.ImportFrom)}
    nodes = types.get(kind, tuple())
    passed = any(isinstance(node, nodes) and (name is None or getattr(node, "name", None) == name or any(alias.name == name for alias in getattr(node, "names", []))) for node in ast.walk(tree))
    return {"type": "ast", "path": item["path"], "kind": kind, "name": name, "passed": passed}


def verify_contract(root: Path, spec: dict[str, Any] | None, backend: ExecutionBackend | None = None, affected_paths: list[str] | None = None) -> dict[str, Any]:
    if not spec: return {"status": "not_configured", "passed": None, "failure_category": None, "checks": [], "summary": "no verifier configured"}
    checks: list[dict[str, Any]] = []
    for stage in spec.get("stages", []):
        stage_result = verify_contract(root, stage, backend, affected_paths); checks.append({"type": "stage", "name": stage.get("name", "unnamed"), "passed": stage_result["passed"], "checks": stage_result["checks"]})
    checks.extend(_file_check(root, item) for item in spec.get("files", []))
    checks.extend(_ast_check(root, item) for item in spec.get("ast", []))
    for item in spec.get("commands", []):
        limits = ResourceLimits(timeout_seconds=int(item.get("timeout", 30)))
        execution = (backend or LocalExecutionBackend()).execute(ExecutionRequest(command_argv(item["command"], bool(item.get("allow_shell", False))), root, limits=limits, policy=ExecutionPolicy()))
        checks.append({"type": "command", "passed": execution.exit_code == int(item.get("exit_code", 0)) and not execution.timed_out, "exit_code": execution.exit_code, "backend": execution.backend, "isolation_level": execution.isolation_level})
    for item in spec.get("assertions", []): checks.append({"type": "assertion", "passed": item.get("actual") == item.get("equals")})
    # Run artifacts use portable repository-relative paths.  Tool execution on
    # Windows may naturally return backslashes, which must not make an otherwise
    # valid change fail a contract authored with POSIX separators.
    changed = [str(path).replace("\\", "/").rstrip("/") for path in (affected_paths or [])]
    if spec.get("allowed_changed_paths") is not None:
        allowed = [str(path).replace("\\", "/").rstrip("/") for path in spec["allowed_changed_paths"]]
        invalid = [path for path in changed if not any(path == prefix or path.startswith(prefix + "/") for prefix in allowed)]
        checks.append({"type": "changed_paths", "passed": not invalid, "invalid": invalid})
    if spec.get("max_changed_files") is not None: checks.append({"type": "patch_size", "passed": len(set(changed)) <= int(spec["max_changed_files"]), "actual": len(set(changed))})
    for key, comparison in (("coverage", "minimum"), ("performance", "maximum")):
        for item in spec.get(key, []):
            actual = float(item["actual"]); threshold = float(item[comparison]); passed = actual >= threshold if comparison == "minimum" else actual <= threshold
            checks.append({"type": key, "metric": item["metric"], "actual": actual, comparison: threshold, "passed": passed})
    passed = bool(checks) and all(check["passed"] for check in checks)
    failed = [check for check in checks if not check["passed"]]
    return {"status": "passed" if passed else "failed", "passed": passed, "failure_category": None if passed else "verifier_failed", "checks": checks, "summary": "all checks passed" if passed else f"{len(failed)} of {len(checks)} checks failed"}
