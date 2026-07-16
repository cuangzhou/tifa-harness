from __future__ import annotations

import os
import json
from typing import Any
import httpx

from .models import ModelResponse, ToolCall


def _openai_messages(messages: list[dict[str, Any]] | None, prompt: str) -> list[dict[str, Any]]:
    if not messages: return [{"role": "user", "content": prompt}]
    result = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in message["tool_calls"]]
            result.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": calls})
        elif message["role"] == "tool": result.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
        else: result.append({"role": message["role"], "content": message.get("content", "")})
    return result


def _anthropic_messages(messages: list[dict[str, Any]] | None, prompt: str) -> list[dict[str, Any]]:
    if not messages: return [{"role": "user", "content": prompt}]
    result = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            blocks = ([{"type": "text", "text": message["content"]}] if message.get("content") else []) + [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in message["tool_calls"]]
            result.append({"role": "assistant", "content": blocks})
        elif message["role"] == "tool": result.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}]})
        else: result.append({"role": message["role"] if message["role"] in {"user", "assistant"} else "user", "content": message.get("content", "")})
    return result


class ProviderError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


class _HTTPModelClient:
    provider = "http"

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: float = 60) -> None:
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    def _post(self, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        merged = {"content-type": "application/json", **(headers or {})}
        try:
            response = httpx.post(f"{self.base_url}{path}", json=payload, headers=merged, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc: raise ProviderError("timeout", self.provider) from exc
        except httpx.HTTPStatusError as exc:
            category = "rate_limit" if exc.response.status_code == 429 else "auth" if exc.response.status_code in {401, 403} else "transport"
            raise ProviderError(category, self.provider) from exc
        except httpx.HTTPError as exc: raise ProviderError("transport", self.provider) from exc
        except ValueError as exc: raise ProviderError("invalid_response", self.provider) from exc


class OpenAICompatibleModelClient(_HTTPModelClient):
    provider = "openai-compatible"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post("/chat/completions", {"model": self.model, "messages": _openai_messages(messages, prompt), "tools": tools})
        try:
            message = data["choices"][0]["message"]
            calls = [ToolCall(str(c["id"]), c["function"]["name"], json.loads(c["function"].get("arguments") or "{}")) for c in message.get("tool_calls", [])]
            return ModelResponse(message.get("content") or "", calls, data.get("usage", {}), {"cache_key": cache_key})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("invalid_response", "openai-compatible") from exc
        except json.JSONDecodeError as exc: raise ProviderError("invalid_response", "invalid tool arguments") from exc


class AnthropicCompatibleModelClient(_HTTPModelClient):
    provider = "anthropic-compatible"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"} if self.api_key else {"anthropic-version": "2023-06-01"}
        mapped = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in tools]
        data = self._post("/messages", {"model": self.model, "max_tokens": 4096, "messages": _anthropic_messages(messages, prompt), "tools": mapped}, headers)
        try:
            text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
            calls = [ToolCall(str(b["id"]), b["name"], b.get("input", {})) for b in data["content"] if b.get("type") == "tool_use"]
            return ModelResponse(text, calls, data.get("usage", {}), {"cache_key": cache_key})
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid_response", "anthropic-compatible") from exc


class OllamaModelClient(_HTTPModelClient):
    provider = "ollama"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse:
        data = self._post("/api/chat", {"model": self.model, "messages": _openai_messages(messages, prompt), "stream": False, "tools": tools})
        try:
            message = data["message"]
            calls = [ToolCall(str(c.get("id") or f"ollama-{i}"), c["function"]["name"], c["function"].get("arguments", {})) for i, c in enumerate(message.get("tool_calls", []), 1)]
            return ModelResponse(message.get("content", ""), calls, {k: data[k] for k in ("prompt_eval_count", "eval_count") if k in data}, {"cache_key": cache_key})
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid_response", "ollama") from exc


def create_model_client(provider: str, model: str | None = None):
    if provider == "openai":
        return OpenAICompatibleModelClient(model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        return AnthropicCompatibleModelClient(model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"), os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"), os.getenv("ANTHROPIC_API_KEY"))
    if provider == "ollama":
        return OllamaModelClient(model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder"), os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise ValueError(f"unsupported provider: {provider}")
