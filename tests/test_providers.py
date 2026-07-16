import httpx

from tifa.providers import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient


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
