from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Protocol
import uuid


@dataclass
class ResourceLimits:
    cpus: float = 1.0
    memory_mb: int = 512
    pids: int = 64
    timeout_seconds: int = 30


@dataclass
class ExecutionPolicy:
    network: str = "deny"
    allow_shell: bool = False
    writable_workspace: bool = True


@dataclass
class ExecutionRequest:
    argv: list[str]
    workspace: Path
    env: dict[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)


@dataclass
class ExecutionResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool
    oom_killed: bool
    backend: str
    isolation_level: str
    image_digest: str | None = None
    security_events: list[str] = field(default_factory=list)


class ExecutionBackend(Protocol):
    name: str
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


def command_argv(command: str, allow_shell: bool = False) -> list[str]:
    if not command.strip(): raise ValueError("command cannot be empty")
    if allow_shell:
        return (["powershell", "-NoProfile", "-NonInteractive", "-Command", command] if os.name == "nt" else ["/bin/sh", "-lc", command])
    forbidden = ("|", ">", "<", "&&", ";", "`", "$(")
    if any(token in command for token in forbidden): raise ValueError("shell syntax requires allow_shell=true")
    return shlex.split(command, posix=os.name != "nt")


class LocalExecutionBackend:
    name = "local"
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter(); env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "PYTHONPATH", "VIRTUAL_ENV"}}
        env.update(request.env); flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(request.argv, cwd=request.workspace, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        timed_out = False
        try: stdout, stderr = process.communicate(timeout=request.limits.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt": subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            else:
                process.kill()
            stdout, stderr = process.communicate()
        return ExecutionResult(process.returncode, stdout[:30000], stderr[:30000], (time.perf_counter() - started) * 1000, timed_out, False, self.name, "local_degraded", security_events=["network_not_isolated", "resource_limits_not_enforced"])


class DockerExecutionBackend:
    name = "docker"
    def __init__(self, image: str = "tifa-runner:0.4.0") -> None: self.image = image
    def _digest(self) -> str | None:
        try: return subprocess.check_output(["docker", "image", "inspect", self.image, "--format", "{{index .RepoDigests 0}}"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip() or subprocess.check_output(["docker", "image", "inspect", self.image, "--format", "{{.Id}}"], text=True, timeout=5).strip()
        except Exception: return None
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if request.policy.network not in {"deny", "allow"}: raise ValueError("invalid network policy")
        name = f"tifa-{uuid.uuid4().hex[:12]}"; mount = f"{request.workspace.resolve()}:/workspace:{'rw' if request.policy.writable_workspace else 'ro'}"
        command = ["docker", "run", "--rm", "--name", name, "--read-only", "--network", "none" if request.policy.network == "deny" else "bridge", "--cpus", str(request.limits.cpus), "--memory", f"{request.limits.memory_mb}m", "--pids-limit", str(request.limits.pids), "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65532:65532", "--mount", f"type=bind,source={request.workspace.resolve()},target=/workspace", "--workdir", "/workspace", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"]
        for key, value in request.env.items(): command += ["--env", f"{key}={value}"]
        command += [self.image, *request.argv]; started = time.perf_counter(); timed_out = False
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try: stdout, stderr = process.communicate(timeout=request.limits.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True; subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True); stdout, stderr = process.communicate()
        inspect = subprocess.run(["docker", "inspect", name, "--format", "{{json .State}}"], capture_output=True, text=True)
        oom = '"OOMKilled":true' in inspect.stdout
        return ExecutionResult(process.returncode, stdout[:30000], stderr[:30000], (time.perf_counter() - started) * 1000, timed_out, oom, self.name, "container_strong", self._digest(), [] if request.policy.network == "deny" else ["network_enabled"])


def result_text(result: ExecutionResult) -> str:
    return f"exit_code={result.exit_code} timed_out={str(result.timed_out).lower()} backend={result.backend}\n{result.stdout}{result.stderr}"[:30000]
