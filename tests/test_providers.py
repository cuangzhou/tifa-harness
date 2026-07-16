import httpx
import pytest

from tifa.providers import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient, ProviderError


class Response:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): return None
    def json(self): return self.payload


def test_openai_mapping(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response({"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 2}}))
    result = OpenAICompatibleModelClient("m", "http://test", "key").complete("p", [])
    assert result.content == "ok" and result.usage["prompt_tokens"] == 2


def test_anthropic_mapping(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 2}}))
    result = AnthropicCompatibleModelClient("m", "http://test", "key").complete("p", [])
    assert result.content == "ok" and result.usage["input_tokens"] == 2


def test_ollama_mapping(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response({"message": {"content": "ok"}, "prompt_eval_count": 2}))
    result = OllamaModelClient("m", "http://test").complete("p", [])
    assert result.content == "ok" and result.usage["prompt_eval_count"] == 2


def test_openai_structured_tool_call(monkeypatch):
    payload = {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]}}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(payload))
    result = OpenAICompatibleModelClient("m", "http://test").complete("p", [])
    assert result.tool_calls[0].id == "c1" and result.tool_calls[0].arguments == {"path": "a.py"}


def test_anthropic_structured_tool_call(monkeypatch):
    payload = {"content": [{"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.py"}}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(payload))
    result = AnthropicCompatibleModelClient("m", "http://test").complete("p", [])
    assert result.tool_calls[0].name == "read_file"


def test_ollama_structured_tool_call(monkeypatch):
    payload = {"message": {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response(payload))
    result = OllamaModelClient("m", "http://test").complete("p", [])
    assert result.tool_calls[0].arguments["path"] == "a.py"


def test_provider_timeout_is_stable_category(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
    with pytest.raises(ProviderError) as error: OpenAICompatibleModelClient("m", "http://test").complete("p", [])
    assert error.value.category == "timeout"
