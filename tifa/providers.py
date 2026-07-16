from __future__ import annotations

import os
from typing import Any
import httpx

from .models import ModelResponse


class ProviderError(RuntimeError):
    pass


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
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"{self.provider} request failed: {exc}") from exc


class OpenAICompatibleModelClient(_HTTPModelClient):
    provider = "openai-compatible"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None) -> ModelResponse:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post("/chat/completions", {"model": self.model, "messages": [{"role": "user", "content": prompt}], "tools": tools})
        try:
            return ModelResponse(data["choices"][0]["message"].get("content") or "", data.get("usage", {}), {"cache_key": cache_key})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("invalid OpenAI-compatible response") from exc


class AnthropicCompatibleModelClient(_HTTPModelClient):
    provider = "anthropic-compatible"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None) -> ModelResponse:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"} if self.api_key else {"anthropic-version": "2023-06-01"}
        data = self._post("/messages", {"model": self.model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}], "tools": [t["function"] for t in tools]}, headers)
        try:
            text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
            return ModelResponse(text, data.get("usage", {}), {"cache_key": cache_key})
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid Anthropic-compatible response") from exc


class OllamaModelClient(_HTTPModelClient):
    provider = "ollama"

    def complete(self, prompt: str, tools: list[dict[str, Any]], cache_key: str | None = None) -> ModelResponse:
        data = self._post("/api/chat", {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False, "tools": tools})
        try:
            return ModelResponse(data["message"]["content"], {k: data[k] for k in ("prompt_eval_count", "eval_count") if k in data}, {"cache_key": cache_key})
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid Ollama response") from exc


def create_model_client(provider: str, model: str | None = None):
    if provider == "openai":
        return OpenAICompatibleModelClient(model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        return AnthropicCompatibleModelClient(model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"), os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"), os.getenv("ANTHROPIC_API_KEY"))
    if provider == "ollama":
        return OllamaModelClient(model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder"), os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    raise ValueError(f"unsupported provider: {provider}")
