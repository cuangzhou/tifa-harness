from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

SECRET = re.compile(r"(?i)(api[_-]?key|token|secret|password)([\"']?\s*[:=]\s*[\"']?)([^\s,\"']+)")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        sensitive = {"api_key", "apikey", "access_token", "refresh_token", "authorization", "password", "secret", "client_secret"}
        return {k: ("[REDACTED]" if k.lower() in sensitive or k.lower().endswith("_api_key") else redact(v)) for k, v in value.items()}
    if isinstance(value, list): return [redact(x) for x in value]
    if isinstance(value, str): return SECRET.sub(r"\1\2[REDACTED]", value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(redact(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class SessionStore:
    def __init__(self, workspace: Path) -> None:
        self.root = workspace / ".tifa" / "sessions"

    def save(self, session_id: str, state: dict[str, Any]) -> Path:
        path = self.root / f"{session_id}.json"
        atomic_json(path, {"schema_version": "tifa-session.v1", **state})
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        if session_id == "latest":
            candidates = list(self.root.glob("*.json"))
            if not candidates: raise FileNotFoundError("no session to resume")
            path = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            path = self.root / f"{session_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "tifa-session.v1": raise ValueError("unsupported session schema")
        return payload


class RunStore:
    def __init__(self, workspace: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.run_dir = workspace / ".tifa" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

    def write(self, name: str, schema: str, value: dict[str, Any]) -> Path:
        path = self.run_dir / name
        atomic_json(path, {"schema_version": schema, **value})
        return path

    def append_trace(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        event = {"sequence": self.sequence, "type": event_type, "timestamp": now(), "payload": redact(payload)}
        with (self.run_dir / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
