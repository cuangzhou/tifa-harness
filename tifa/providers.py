from __future__ import annotations

import os
import json
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import hashlib
import random
import threading
import time
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


def _ollama_messages(messages: list[dict[str, Any]] | None, prompt: str) -> list[dict[str, Any]]:
    if not messages: return [{"role": "user", "content": prompt}]
    result = []; call_names: dict[str, str] = {}
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls = []
            for index, call in enumerate(message["tool_calls"]):
                call_names[call["id"]] = call["name"]
                calls.append({"type": "function", "function": {"index": index, "name": call["name"], "arguments": call["arguments"]}})
            result.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": calls})
        elif message["role"] == "tool":
            result.append({"role": "tool", "tool_name": call_names.get(message["tool_call_id"], "unknown"), "content": message["content"]})
        else:
            result.append({"role": message["role"], "content": message.get("content", "")})
    return result


class ProviderError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 0.25
    max_delay: float = 8.0
    jitter: float = 0.2
    max_concurrency: int = 4


def _retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


class _HTTPModelClient:
    provider = "http"

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: float = 60, retry_policy: RetryPolicy | None = None) -> None:
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self._semaphore = threading.BoundedSemaphore(self.retry_policy.max_concurrency)
        self.last_request_meta: dict[str, Any] = {}

    def _post(self, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        merged = {"content-type": "application/json", **(headers or {})}
        canonical = json.dumps({"provider": self.provider, "model": self.model, "path": path, "payload": payload}, sort_keys=True, ensure_ascii=False)
        request_id = hashlib.sha256(canonical.encode()).hexdigest()[:24]
        attempts = 0
        with self._semaphore:
            while True:
                attempts += 1
                try:
                    response = httpx.post(
                        f"{self.base_url}{path}", json=payload,
                        headers={**merged, "x-tifa-request-id": request_id},
                        timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10.0)),
                    )
                    response.raise_for_status()
                    data = response.json()
                    self.last_request_meta = {"request_id": request_id, "attempts": attempts}
                    return data
                except httpx.TimeoutException as exc:
                    category, retryable, retry_after = "timeout", True, None
                    cause: Exception = exc
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    category = "rate_limit" if status == 429 else "auth" if status in {401, 403} else "transport"
                    retryable = status == 429 or 500 <= status < 600
                    retry_after = _retry_after(exc.response.headers)
                    cause = exc
                except httpx.HTTPError as exc:
                    category, retryable, retry_after, cause = "transport", False, None, exc
                except ValueError as exc:
                    raise ProviderError("invalid_response", self.provider) from exc
                if not retryable or attempts > self.retry_policy.max_retries:
                    self.last_request_meta = {"request_id": request_id, "attempts": attempts, "failure_category": category}
                    raise ProviderError(category, self.provider) from cause
                delay = retry_after if retry_after is not None else min(self.retry_policy.max_delay, self.retry_policy.base_delay * (2 ** (attempts - 1)))
                delay *= 1 + random.uniform(-self.retry_policy.jitter, self.retry_policy.jitter)
                time.sleep(max(0.0, delay))


class OpenAICompatibleModelClient(_HTTPModelClient):
    provider = "openai-compatible"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post("/chat/completions", {"model": self.model, "messages": _openai_messages(messages, prompt), "tools": tools}, headers)
        try:
            message = data["choices"][0]["message"]
            calls = [ToolCall(str(c["id"]), c["function"]["name"], json.loads(c["function"].get("arguments") or "{}")) for c in message.get("tool_calls", [])]
            return ModelResponse(message.get("content") or "", calls, data.get("usage", {}), {"cache_key": cache_key, **self.last_request_meta}, self.last_request_meta.get("request_id"))
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
            return ModelResponse(text, calls, data.get("usage", {}), {"cache_key": cache_key, **self.last_request_meta}, self.last_request_meta.get("request_id"))
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid_response", "anthropic-compatible") from exc


class OllamaModelClient(_HTTPModelClient):
    provider = "ollama"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None, messages: list[dict[str, Any]] | None = None) -> ModelResponse:
        data = self._post("/api/chat", {"model": self.model, "messages": _ollama_messages(messages, prompt), "stream": False, "options": {"temperature": 0}, "tools": tools})
        try:
            message = data["message"]
            calls = [ToolCall(str(c.get("id") or f"ollama-{i}"), c["function"]["name"], c["function"].get("arguments", {})) for i, c in enumerate(message.get("tool_calls", []), 1)]
            return ModelResponse(message.get("content", ""), calls, {k: data[k] for k in ("prompt_eval_count", "eval_count") if k in data}, {"cache_key": cache_key, **self.last_request_meta}, self.last_request_meta.get("request_id"))
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid_response", "ollama") from exc


def create_model_client(provider: str, model: str | None = None):
    if provider == "openai":
        return OpenAICompatibleModelClient(model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini", os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1", os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        return AnthropicCompatibleModelClient(model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514", os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1", os.getenv("ANTHROPIC_API_KEY"))
    if provider == "ollama":
        return OllamaModelClient(model or os.getenv("OLLAMA_MODEL") or "qwen2.5-coder:3b", os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
    raise ValueError(f"unsupported provider: {provider}")
