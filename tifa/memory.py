from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json


def default_memory_state() -> dict[str, Any]:
    return {"task_summary": "", "recent_files": [], "file_summaries": {}, "episodic_notes": []}


def summarize_read_result(content: str, limit: int = 240) -> str:
    compact = " ".join(content.strip().split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


class LayeredMemory:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state = deepcopy(state or default_memory_state())

    def remember_file(self, path: str, content: str | None = None) -> None:
        recent = self.state["recent_files"]
        if path in recent:
            recent.remove(path)
        recent.append(path)
        del recent[:-12]
        if content is not None:
            self.state["file_summaries"][path] = summarize_read_result(content)

    def invalidate_file(self, path: str) -> None:
        self.remember_file(path)
        self.state["file_summaries"].pop(path, None)

    def note(self, text: str) -> None:
        self.state["episodic_notes"].append(text[:500])
        del self.state["episodic_notes"][:-20]

    def after_tool(self, name: str, args: dict[str, Any], output: str) -> None:
        path = args.get("path")
        if name == "read_file" and path:
            self.remember_file(path, output)
            self.note(f"Read {path}: {summarize_read_result(output, 120)}")
        elif name in {"write_file", "patch_file"} and path:
            self.invalidate_file(path)
            self.note(f"Modified {path}; prior summary invalidated.")

    def render(self, limit: int = 4000) -> str:
        return json.dumps(self.state, ensure_ascii=False, indent=2)[:limit]


class DurableMemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, memory: LayeredMemory) -> None:
        self.root.parent.mkdir(parents=True, exist_ok=True)
        temp = self.root.with_suffix(".tmp")
        temp.write_text(json.dumps({"schema_version": "tifa-memory.v1", "memory": memory.state}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.root)

    def load(self) -> LayeredMemory:
        if not self.root.exists():
            return LayeredMemory()
        payload = json.loads(self.root.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "tifa-memory.v1":
            raise ValueError("unsupported memory schema")
        return LayeredMemory(payload["memory"])
