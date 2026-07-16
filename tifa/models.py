from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    raw_response_ref: str | None = None

    @property
    def content(self) -> str:
        return self.text


class ModelClient(Protocol):
    provider: str
    model: str

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse: ...


class FakeModelClient:
    provider = "fake"

    def __init__(self, outputs: list[str | ModelResponse] | None = None, model: str = "fake-model") -> None:
        self.outputs = list(outputs or ["<final>FakeModel completed the request.</final>"])
        self.model = model
        self.prompts: list[str] = []

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse:
        self.prompts.append(prompt)
        output = self.outputs.pop(0) if self.outputs else "<final>No scripted output remains.</final>"
        if isinstance(output, ModelResponse): return output
        return ModelResponse(output, usage={"input_estimate": len(prompt) // 4, "output_estimate": len(output) // 4}, cache={"cache_key": cache_key})


@dataclass
class AgentResult:
    answer: str
    run_id: str
    session_id: str
    stop_reason: str
    tool_steps: int
    attempts: int
    run_dir: str
