from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

from .execution import DockerExecutionBackend, ExecutionRequest


def doctor(workspace: str | Path, model: str = "qwen2.5-coder:3b", ollama_url: str = "http://127.0.0.1:11434") -> dict:
    root = Path(workspace).resolve(); checks: dict[str, dict] = {}
    try:
        version = subprocess.check_output(["docker", "version", "--format", "{{.Server.Version}}"], text=True, timeout=10).strip()
        checks["docker_server"] = {"ok": bool(version), "version": version}
    except Exception as exc: checks["docker_server"] = {"ok": False, "error": type(exc).__name__}
    try:
        runtimes = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .Runtimes}}"], text=True, timeout=10))
        checks["gpu_runtime"] = {"ok": "nvidia" in runtimes, "available": sorted(runtimes)}
    except Exception as exc: checks["gpu_runtime"] = {"ok": False, "error": type(exc).__name__}
    try:
        with urlopen(f"{ollama_url}/api/tags", timeout=3) as response: tags = json.load(response)
        names = [entry.get("name", "") for entry in tags.get("models", [])]
        checks["ollama"] = {"ok": True, "url": ollama_url}; checks["model"] = {"ok": any(name == model or name.startswith(model + ":") for name in names), "requested": model, "available": names}
    except Exception as exc:
        checks["ollama"] = {"ok": False, "url": ollama_url, "error": type(exc).__name__}; checks["model"] = {"ok": False, "requested": model}
    try:
        result = DockerExecutionBackend().execute(ExecutionRequest(["python", "-c", "import os,pathlib; print(os.getuid()); pathlib.Path('doctor.tmp').write_text('ok')"], root))
        (root / "doctor.tmp").unlink(missing_ok=True)
        checks["sandbox"] = {"ok": result.exit_code == 0 and result.stdout.splitlines()[0] != "0", **asdict(result)}
    except Exception as exc: checks["sandbox"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    required = ("docker_server", "gpu_runtime", "ollama", "model", "sandbox")
    return {"status": "healthy" if all(checks[key]["ok"] for key in required) else "degraded", "checks": checks}
