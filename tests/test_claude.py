"""ClaudeLLM against a mocked HTTP transport: checks the exact requests sent to the API."""

import json

import anthropic
import httpx2
import pytest

from scholarscout.llm import ClaudeLLM, LLMError, Usage
from scholarscout.schemas import ResearchPlan


def fake_api(text: str, stop_reason: str = "end_turn"):
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": json.loads(request.content)["model"],
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

    client = anthropic.Anthropic(
        api_key="test-key", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    return client, requests


def test_text_request_on_opus_uses_effort_and_refusal_fallback():
    client, requests = fake_api("Hello")
    llm = ClaudeLLM("claude-opus-5-5", effort="medium", client=client)

    assert llm.text("You are terse.", "Say hello") == "Hello"
    body = json.loads(requests[0].content)
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "You are terse."
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in requests[0].headers["anthropic-beta"]


def test_structured_output_is_parsed_into_the_schema():
    plan_json = '{"sub_questions": [{"question": "What is Mamba?", "search_query": "mamba ssm"}]}'
    client, requests = fake_api(plan_json)
    llm = ClaudeLLM("claude-opus-5-5", client=client)

    plan = llm.structured("Plan.", "Question", ResearchPlan, effort="low")
    assert isinstance(plan, ResearchPlan)
    assert plan.sub_questions[0].search_query == "mamba ssm"
    output_config = json.loads(requests[0].content)["output_config"]
    assert output_config["effort"] == "low"
    assert output_config["format"]["type"] == "json_schema"


def test_models_without_server_fallback_use_the_plain_endpoint():
    client, requests = fake_api("Hi")
    ClaudeLLM("claude-haiku-5-5", client=client).text("s", "p")
    body = json.loads(requests[0].content)
    assert "fallbacks" not in body
    assert "server-side-fallback" not in requests[0].headers.get("anthropic-beta", "")


def test_empty_effort_omits_the_parameter():
    client, requests = fake_api("Hi")
    ClaudeLLM("claude-opus-5-5", effort="", client=client).text("s", "p", effort="low")
    assert "output_config" not in json.loads(requests[0].content)


def test_refusal_raises_a_clear_error():
    client, _ = fake_api("", stop_reason="refusal")
    with pytest.raises(LLMError, match="declined"):
        ClaudeLLM("claude-opus-5-5", client=client).text("s", "p")


def test_usage_and_cost_are_tracked():
    client, _ = fake_api("Hi")
    llm = ClaudeLLM("claude-opus-5-5", client=client)
    llm.text("s", "p")
    llm.text("s", "p")
    usage = llm.usage.snapshot()
    assert usage == Usage(
        calls=2, input_tokens=200, output_tokens=40, models={"claude-opus-5-5": 2}
    )
    assert usage.cost("claude-opus-5-5") == pytest.approx((200 * 4 + 40 * 20) / 1e6)
