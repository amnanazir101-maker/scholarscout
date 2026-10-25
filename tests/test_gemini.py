"""GeminiLLM against a mocked HTTP transport, plus provider selection."""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from scholarscout.config import Settings
from scholarscout.llm import ClaudeLLM, GeminiLLM, LLMError, Usage, make_llm
from scholarscout.schemas import ClaimVerdicts, ResearchNote, ResearchPlan
from tests.fakes import FakeLLM
from tests.test_pipeline import QUESTION, make_assistant


def answer(text: str, finish_reason: str = "STOP") -> dict:
    parts = [{"text": text}] if text else []
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": finish_reason}
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 30,
        },
    }


def fake_gemini(body: dict, status: int = 200):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=http))
    return client, requests


def test_structured_request_sends_schema_system_prompt_and_thinking_level():
    plan = '{"sub_questions": [{"question": "What is Mamba?", "search_query": "mamba ssm"}]}'
    client, requests = fake_gemini(answer(plan))
    llm = GeminiLLM("gemini-3.8-flash", effort="medium", client=client)

    result = llm.structured("Plan.", "Question", ResearchPlan, effort="low")
    assert result.sub_questions[0].search_query == "mamba ssm"
    assert requests[0].url.path.endswith("/models/gemini-3.8-flash:generateContent")
    body = json.loads(requests[0].content)
    assert body["systemInstruction"]["parts"][0]["text"] == "Plan."
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["type"] == "OBJECT"
    thinking = config["thinkingConfig"]
    assert (thinking.get("thinkingLevel") or thinking.get("thinking_level")) == "LOW"


def test_nested_schemas_are_parsed():
    verdicts = '{"verdicts": [{"claim_id": 1, "supported": true, "reason": "Stated."}]}'
    client, _ = fake_gemini(answer(verdicts))
    result = GeminiLLM(client=client).structured("Check.", "Claims", ClaimVerdicts)
    assert result.verdicts[0].supported is True


def test_thinking_tokens_count_as_output():
    client, _ = fake_gemini(answer("Hello"))
    llm = GeminiLLM(client=client)
    assert llm.text("s", "p") == "Hello"
    expected = Usage(calls=1, input_tokens=100, output_tokens=50, models={"gemini-3.8-flash": 1})
    assert llm.usage.snapshot() == expected


def test_gemini_2_5_models_get_no_thinking_level():
    client, requests = fake_gemini(answer("Hi"))
    GeminiLLM("gemini-2.5-flash", client=client).text("s", "p")
    assert "thinkingConfig" not in json.loads(requests[0].content).get("generationConfig", {})


def test_blocked_or_empty_answers_raise():
    client, _ = fake_gemini(answer("", finish_reason="SAFETY"))
    with pytest.raises(LLMError, match="empty answer"):
        GeminiLLM(client=client).text("s", "p")


def test_rate_limit_becomes_a_clear_error():
    error = {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    client, _ = fake_gemini(error, status=429)
    with pytest.raises(LLMError, match="free-tier limit"):
        GeminiLLM(client=client, rate_limit_waits=()).text("s", "p")


def busy_gemini(busy_models: set[str], code: int = 503):
    """A fake API where the listed models are busy and every other model answers."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.rsplit("/", 1)[-1].split(":")[0]
        calls.append(model)
        if model in busy_models:
            status = "UNAVAILABLE" if code == 503 else "RESOURCE_EXHAUSTED"
            return httpx.Response(
                code, json={"error": {"code": code, "message": "busy", "status": status}}
            )
        return httpx.Response(200, json=answer(f"answer from {model}"))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=http))
    return client, calls


def test_busy_model_falls_back_for_the_rest_of_the_run():
    client, calls = busy_gemini({"gemini-3.8-flash"})
    llm = GeminiLLM(client=client, fallback_models=["gemini-3.6-flash", "gemini-3.5-flash-lite"])

    assert llm.text("s", "p") == "answer from gemini-3.6-flash"
    assert llm.text("s", "p") == "answer from gemini-3.6-flash"
    assert calls == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.6-flash"]
    assert llm.usage.snapshot().models == {"gemini-3.6-flash": 2}


def test_last_model_waits_out_a_per_minute_limit():
    replies = iter([429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(replies)
        if status == 429:
            error = {"code": 429, "message": "Per-minute quota", "status": "RESOURCE_EXHAUSTED"}
            return httpx.Response(429, json={"error": error})
        return httpx.Response(200, json=answer("OK"))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=http))
    assert GeminiLLM(client=client, rate_limit_waits=(0,)).text("s", "p") == "OK"


def test_error_names_every_model_when_all_are_rate_limited():
    client, calls = busy_gemini({"gemini-3.8-flash", "gemini-3.6-flash"}, code=429)
    llm = GeminiLLM(client=client, fallback_models=["gemini-3.6-flash"], rate_limit_waits=())
    with pytest.raises(LLMError, match="gemini-3.8-flash, gemini-3.6-flash"):
        llm.text("s", "p")
    assert calls == ["gemini-3.8-flash", "gemini-3.6-flash"]


def test_missing_key_points_to_ai_studio(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="aistudio.google.com"):
        GeminiLLM()


def test_full_workflow_runs_through_the_gemini_sdk():
    """Every agent call goes through the real SDK; a scripted fake answers the HTTP requests."""
    scripted = FakeLLM()
    schemas = {cls.__name__: cls for cls in (ResearchPlan, ResearchNote, ClaimVerdicts)}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["systemInstruction"]["parts"][0]["text"]
        prompt = body["contents"][0]["parts"][0]["text"]
        title = body["generationConfig"].get("responseSchema", {}).get("title")
        if title:
            text = scripted.structured(system, prompt, schemas[title]).model_dump_json()
        else:
            text = scripted.text(system, prompt)
        return httpx.Response(200, json=answer(text))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = genai.Client(api_key="test-key", http_options=types.HttpOptions(httpx_client=http))
    llm = GeminiLLM(client=client)
    result = make_assistant(llm).run(QUESTION)

    assert result.review.passed
    assert "## References\n1. " in result.report
    assert llm.usage.snapshot().calls == len(scripted.calls)


def test_make_llm_uses_the_provider_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    gemini = make_llm(Settings())
    assert isinstance(gemini, GeminiLLM) and gemini.model == "gemini-3.8-flash"
    claude = make_llm(Settings(provider="claude"), api_key="test-key")
    assert isinstance(claude, ClaudeLLM) and claude.model == "claude-opus-5-5"
    assert make_llm(Settings(model="gemini-3.5-flash-lite")).model == "gemini-3.5-flash-lite"
    with pytest.raises(LLMError, match="provider"):
        make_llm(Settings(provider="openai"))
