"""Tests for the OpenAI-compatible /v1/chat/completions endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from providers.nvidia_nim import NvidiaNimProvider

app = create_app()

mock_provider = MagicMock(spec=NvidiaNimProvider)


async def _mock_stream_response(*args, **kwargs):
    """Return a minimal Anthropic SSE stream."""
    yield (
        "event: message_start\n"
        'data: {"type": "message_start", "message": {"id": "msg_test123", "type": "message", '
        '"role": "assistant", "content": [], "model": "test-model", "stop_reason": null, '
        '"stop_sequence": null, "usage": {"input_tokens": 5, "output_tokens": 1}}}\n\n'
    )
    yield (
        "event: content_block_start\n"
        'data: {"type": "content_block_start", "index": 0, '
        '"content_block": {"type": "text", "text": ""}}\n\n'
    )
    yield (
        "event: content_block_delta\n"
        'data: {"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "text_delta", "text": "Hello"}}\n\n'
    )
    yield (
        "event: content_block_stop\n"
        'data: {"type": "content_block_stop", "index": 0}\n\n'
    )
    yield (
        "event: message_delta\n"
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, '
        '"usage": {"input_tokens": 5, "output_tokens": 10}}\n\n'
    )
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


mock_provider.stream_response = _mock_stream_response


@pytest.fixture(scope="module")
def client():
    with (
        patch("api.dependencies.resolve_provider", return_value=mock_provider),
        patch(
            "providers.registry.ProviderRegistry.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.registry.ProviderRegistry.start_model_list_refresh"),
        TestClient(app) as test_client,
    ):
        yield test_client


def _chat_payload(**kwargs) -> dict:
    base = {
        "model": "nvidia_nim/test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    base.update(kwargs)
    return base


# =============================================================================
# Streaming tests
# =============================================================================


def test_streaming_returns_200_with_text_event_stream(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=True),
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_streaming_returns_openai_sse_format(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=True),
    )
    assert response.status_code == 200
    content = b"".join(response.iter_bytes())
    text = content.decode("utf-8")
    assert "data: " in text
    assert "[DONE]" in text
    assert "chat.completion.chunk" in text


def test_streaming_contains_content(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=True),
    )
    assert response.status_code == 200
    content = b"".join(response.iter_bytes())
    text = content.decode("utf-8")
    assert "Hello" in text


# =============================================================================
# Non-streaming tests
# =============================================================================


def test_non_streaming_returns_200_json(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=False),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) == 1


def test_non_streaming_response_structure(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=False),
    )
    data = response.json()
    assert "id" in data
    assert "model" in data
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]
    choice = data["choices"][0]
    assert "message" in choice
    assert "finish_reason" in choice
    assert choice["message"]["role"] == "assistant"


def test_non_streaming_contains_content(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=False),
    )
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert "Hello" in content


# =============================================================================
# Model routing
# =============================================================================


def test_model_routing_with_provider_prefix(client: TestClient):
    """Provider-prefixed model names route to the correct provider."""
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(model="nvidia_nim/test-model", stream=False),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "test-model"


# =============================================================================
# Tool calls
# =============================================================================


async def _mock_stream_with_tool_call(*args, **kwargs):
    """Return an Anthropic SSE stream with a tool_use block."""
    yield (
        "event: message_start\n"
        'data: {"type": "message_start", "message": {"id": "msg_tool123", "type": "message", '
        '"role": "assistant", "content": [], "model": "test-model", "stop_reason": null, '
        '"stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}}\n\n'
    )
    yield (
        "event: content_block_start\n"
        'data: {"type": "content_block_start", "index": 0, '
        '"content_block": {"type": "tool_use", "id": "toolu_abc123", '
        '"name": "get_weather", "input": {}}}\n\n'
    )
    yield (
        "event: content_block_delta\n"
        'data: {"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "input_json_delta", "partial_json": "{\\"city\\": \\"NYC\\"}"}}\n\n'
    )
    yield (
        "event: content_block_stop\n"
        'data: {"type": "content_block_stop", "index": 0}\n\n'
    )
    yield (
        "event: message_delta\n"
        'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, '
        '"usage": {"input_tokens": 10, "output_tokens": 20}}\n\n'
    )
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def test_tool_calls_converted_to_openai_format(client: TestClient):
    mock_provider.stream_response = _mock_stream_with_tool_call
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(
            stream=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        ),
    )
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"].get("tool_calls", [])
    assert len(tool_calls) >= 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    mock_provider.stream_response = _mock_stream_response


# =============================================================================
# Probes and auth
# =============================================================================


def test_head_probe_returns_204(client: TestClient):
    response = client.head("/v1/chat/completions")
    assert response.status_code == 204
    assert "Allow" in response.headers


def test_options_probe_returns_204(client: TestClient):
    response = client.options("/v1/chat/completions")
    assert response.status_code == 204
    assert "Allow" in response.headers


def test_auth_required():
    """Without auth token, endpoint should return 401."""
    from api.dependencies import get_settings
    from config.settings import Settings

    test_app = create_app()
    settings = Settings()
    settings.anthropic_auth_token = "test-secret"
    test_app.dependency_overrides[get_settings] = lambda: settings

    with (
        patch("api.dependencies.resolve_provider", return_value=mock_provider),
        patch(
            "providers.registry.ProviderRegistry.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.registry.ProviderRegistry.start_model_list_refresh"),
        TestClient(test_app) as test_client,
    ):
        response = test_client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
        )
        assert response.status_code == 401


# =============================================================================
# Error handling
# =============================================================================


def test_empty_messages_returns_error(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": []},
    )
    assert response.status_code == 400


def test_provider_error_returns_status(client: TestClient):
    from providers.exceptions import RateLimitError

    def _raise_rate_limit(*args, **kwargs):
        raise RateLimitError("Too Many Requests")

    mock_provider.stream_response = _raise_rate_limit
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=False),
    )
    assert response.status_code == 429
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_returns_500(client: TestClient):
    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    mock_provider.stream_response = _raise_runtime
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(stream=False),
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response


# =============================================================================
# Legacy /v1/completions tests
# =============================================================================


def _completion_payload(**kwargs) -> dict:
    base = {
        "model": "nvidia_nim/test-model",
        "prompt": "Hello",
    }
    base.update(kwargs)
    return base


def test_completion_streaming_returns_200(client: TestClient):
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=True),
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_completion_streaming_returns_text_completion_object(client: TestClient):
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=True),
    )
    assert response.status_code == 200
    content = b"".join(response.iter_bytes())
    text = content.decode("utf-8")
    assert "data: " in text
    assert "[DONE]" in text
    assert "text_completion" in text
    assert "chat.completion.chunk" not in text


def test_completion_streaming_contains_content(client: TestClient):
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=True),
    )
    assert response.status_code == 200
    content = b"".join(response.iter_bytes())
    text = content.decode("utf-8")
    assert "Hello" in text


def test_completion_non_streaming_returns_200_json(client: TestClient):
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=False),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert "choices" in data
    assert len(data["choices"]) == 1


def test_completion_non_streaming_response_structure(client: TestClient):
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=False),
    )
    data = response.json()
    assert "id" in data
    assert "model" in data
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]
    choice = data["choices"][0]
    assert "text" in choice
    assert "finish_reason" in choice
    assert "logprobs" in choice
    assert choice["logprobs"] is None


def test_completion_non_streaming_contains_content(client: TestClient):
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=False),
    )
    data = response.json()
    text = data["choices"][0]["text"]
    assert "Hello" in text


def test_completion_list_prompt(client: TestClient):
    """list[str] prompt should be joined and sent correctly."""
    response = client.post(
        "/v1/completions",
        json=_completion_payload(prompt=["Hello", "World"], stream=False),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"


def test_completion_head_probe_returns_204(client: TestClient):
    response = client.head("/v1/completions")
    assert response.status_code == 204
    assert "Allow" in response.headers


def test_completion_options_probe_returns_204(client: TestClient):
    response = client.options("/v1/completions")
    assert response.status_code == 204
    assert "Allow" in response.headers


def test_completion_empty_prompt_returns_error(client: TestClient):
    response = client.post(
        "/v1/completions",
        json={"model": "test", "prompt": ""},
    )
    assert response.status_code == 400


def test_completion_provider_error_returns_status(client: TestClient):
    from providers.exceptions import RateLimitError

    def _raise_rate_limit(*args, **kwargs):
        raise RateLimitError("Too Many Requests")

    mock_provider.stream_response = _raise_rate_limit
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=False),
    )
    assert response.status_code == 429
    mock_provider.stream_response = _mock_stream_response


def test_completion_generic_exception_returns_500(client: TestClient):
    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    mock_provider.stream_response = _raise_runtime
    response = client.post(
        "/v1/completions",
        json=_completion_payload(stream=False),
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response
