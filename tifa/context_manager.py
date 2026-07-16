from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from .memory import LayeredMemory
from .workspace import WorkspaceContext

DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")
DEFAULT_SECTION_FLOORS = {"prefix": 1200, "memory": 800, "history": 1000, "relevant_memory": 300}


@dataclass
class PromptBuild:
    prompt: str
    cache_key: str
    metadata: dict[str, Any]


class ContextManager:
    def __init__(self, workspace: WorkspaceContext, total_budget: int = DEFAULT_TOTAL_BUDGET) -> None:
        self.workspace, self.total_budget = workspace, total_budget

    def build_prefix(self, tools: list[dict[str, Any]]) -> str:
        names = ", ".join(tool["function"]["name"] for tool in tools)
        return (
            "You are Tifa, a local coding agent. Use only registered tools. "
            "Never claim a change succeeded without tool evidence. Respond with "
            "<tool>{\"name\":...,\"arguments\":{...}}</tool> or <final>...</final>.\n"
            f"Tools: {names}\n\n{self.workspace.text()}"
        )

    @staticmethod
    def _history(history: list[dict[str, Any]]) -> str:
        older, recent = history[:-6], history[-6:]
        seen_reads: set[str] = set()
        collapsed: list[str] = []
        for item in reversed(older):
            path = item.get("arguments", {}).get("path") if item.get("tool") == "read_file" else None
            if path and path in seen_reads:
                continue
            if path:
                seen_reads.add(path)
            collapsed.append(str(item)[:500])
        return "\n".join(reversed(collapsed)) + "\n" + "\n".join(str(x)[:2000] for x in recent)

    def build(self, request: str, memory: LayeredMemory, history: list[dict[str, Any]], tools: list[dict[str, Any]], relevant_memory: list[str] | None = None) -> PromptBuild:
        sections = {
            "prefix": self.build_prefix(tools),
            "memory": memory.render(),
            "history": self._history(history),
            "relevant_memory": "\n".join((relevant_memory or [])[:3]),
        }
        original = {key: len(value) for key, value in sections.items()}
        reductions: list[dict[str, int | str]] = []
        while sum(map(len, sections.values())) + len(request) > self.total_budget:
            changed = False
            for key in DEFAULT_REDUCTION_ORDER:
                floor = DEFAULT_SECTION_FLOORS[key]
                if len(sections[key]) > floor:
                    before = len(sections[key])
                    sections[key] = sections[key][-max(floor, int(before * 0.75)):]
                    reductions.append({"section": key, "before": before, "after": len(sections[key])})
                    changed = True
                    break
            if not changed:
                break
        prompt = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in sections.items() if v) + f"\n\n[CURRENT REQUEST]\n{request}"
        prefix = sections["prefix"]
        return PromptBuild(prompt, hashlib.sha256(prefix.encode()).hexdigest(), {"section_lengths": {k: len(v) for k, v in sections.items()}, "original_section_lengths": original, "budget_reductions": reductions, "total_length": len(prompt)})
