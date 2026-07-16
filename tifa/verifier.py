from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contract(root: Path, spec: dict[str, Any] | None) -> dict[str, Any]:
    if not spec: return {"status": "not_configured", "passed": None, "failure_category": None, "checks": []}
    checks: list[dict[str, Any]] = []
    for item in spec.get("files", []):
        path = (root / item["path"]).resolve()
        exists = path.is_file(); ok = exists and (not item.get("digest") or file_digest(path) == item["digest"])
        checks.append({"type": "file", "path": item["path"], "passed": ok})
    for item in spec.get("commands", []):
        result = subprocess.run(item["command"], cwd=root, shell=True, capture_output=True, text=True, timeout=int(item.get("timeout", 30)))
        checks.append({"type": "command", "passed": result.returncode == int(item.get("exit_code", 0)), "exit_code": result.returncode})
    for item in spec.get("assertions", []):
        actual = item.get("actual"); expected = item.get("equals")
        checks.append({"type": "assertion", "passed": actual == expected})
    passed = all(c["passed"] for c in checks)
    return {"status": "passed" if passed else "failed", "passed": passed, "failure_category": None if passed else "verifier_failed", "checks": checks}
