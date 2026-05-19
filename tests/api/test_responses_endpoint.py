"""Tests for the /v1/responses endpoint."""

import json

import pytest

from api.models import (
    EasyInputMessage,
    FunctionCall,
    FunctionCallOutput,
    ResponsesFunctionTool,
    ResponsesRequest,
)
from api.models.anthropic import (
    ContentBlockToolResult,
    ContentBlockToolUse,
)
from api.responses_service import responses_to_messages_request
from core.anthropic.responses_sse import (
    convert_anthropic_sse_to_responses_stream,
)


class TestResponsesRequestParsing:
    """Verify Pydantic models parse Responses API shapes correctly."""

    def test_parse_simple_string_input(self):
        req = ResponsesRequest(model="test", input="Hello")
        assert req.input == "Hello"
        assert req.model == "test"

    def test_parse_list_input_with_easy_message(self):
        req = ResponsesRequest(
            model="test",
            input=[{"type": "message", "role": "user", "content": "Hello"}],
        )
        assert len(req.input) == 1
        item = req.input[0]
        assert isinstance(item, EasyInputMessage)
        assert item.role == "user"
        assert item.content == "Hello"

    def test_parse_function_call_input_item(self):
        req = ResponsesRequest(
            model="test",
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                },
            ],
        )
        assert len(req.input) == 1
        assert isinstance(req.input[0], FunctionCall)
        assert req.input[0].call_id == "call_1"

    def test_parse_function_call_output_input_item(self):
        req = ResponsesRequest(
            model="test",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Sunny, 72F",
                },
            ],
        )
        assert len(req.input) == 1
        assert isinstance(req.input[0], FunctionCallOutput)
        assert req.input[0].output == "Sunny, 72F"

    def test_parse_tools(self):
        req = ResponsesRequest(
            model="test",
            input="Hello",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        )
        assert len(req.tools) == 1
        assert isinstance(req.tools[0], ResponsesFunctionTool)
        assert req.tools[0].name == "get_weather"

    def test_parse_tool_choice_string(self):
        req = ResponsesRequest(model="test", input="Hello", tool_choice="auto")
        assert req.tool_choice == "auto"

    def test_parse_instructions(self):
        req = ResponsesRequest(
            model="test", input="Hello", instructions="You are helpful"
        )
        assert req.instructions == "You are helpful"

    def test_parse_max_output_tokens(self):
        req = ResponsesRequest(model="test", input="Hello", max_output_tokens=100)
        assert req.max_output_tokens == 100

    def test_parse_stream_default_false(self):
        req = ResponsesRequest(model="test", input="Hello")
        assert req.stream is False

    def test_previous_response_id_accepted(self):
        req = ResponsesRequest(
            model="test", input="Hello", previous_response_id="resp_old"
        )
        assert req.previous_response_id == "resp_old"

    def test_extra_fields_allowed(self):
        req = ResponsesRequest(model="test", input="Hello", some_unknown_field="value")
        assert req.model == "test"


class TestResponsesToAnthropicConversion:
    """Verify Responses API input converts to Anthropic MessagesRequest correctly."""

    def test_string_input_becomes_single_user_message(self):
        req = ResponsesRequest(model="test", input="Hello")
        result = responses_to_messages_request(req, "resolved-model")
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"

    def test_instructions_become_system(self):
        req = ResponsesRequest(model="test", input="Hello", instructions="Be helpful")
        result = responses_to_messages_request(req, "resolved-model")
        assert result.system == "Be helpful"

    def test_user_easy_input_message(self):
        req = ResponsesRequest(
            model="test",
            input=[EasyInputMessage(role="user", content="Hello")],
        )
        result = responses_to_messages_request(req, "resolved-model")
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"

    def test_assistant_easy_input_message(self):
        req = ResponsesRequest(
            model="test",
            input=[
                EasyInputMessage(role="user", content="Hi"),
                EasyInputMessage(role="assistant", content="Hello!"),
            ],
        )
        result = responses_to_messages_request(req, "resolved-model")
        assert len(result.messages) == 2
        assert result.messages[1].role == "assistant"

    def test_function_call_becomes_tool_use(self):
        req = ResponsesRequest(
            model="test",
            input=[
                FunctionCall(
                    type="function_call",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"city": "NYC"}',
                ),
            ],
        )
        result = responses_to_messages_request(req, "resolved-model")
        assistant_msg = result.messages[0]
        assert assistant_msg.role == "assistant"
        tool_block = assistant_msg.content[0]
        assert isinstance(tool_block, ContentBlockToolUse)
        assert tool_block.name == "get_weather"
        assert tool_block.input == {"city": "NYC"}
        assert tool_block.id == "call_1"

    def test_function_call_output_becomes_tool_result(self):
        req = ResponsesRequest(
            model="test",
            input=[
                FunctionCallOutput(
                    type="function_call_output",
                    call_id="call_1",
                    output="Sunny, 72F",
                ),
            ],
        )
        result = responses_to_messages_request(req, "resolved-model")
        user_msg = result.messages[0]
        assert user_msg.role == "user"
        result_block = user_msg.content[0]
        assert isinstance(result_block, ContentBlockToolResult)
        assert result_block.tool_use_id == "call_1"
        assert result_block.content == "Sunny, 72F"

    def test_tools_converted_to_anthropic_format(self):
        req = ResponsesRequest(
            model="test",
            input="Hello",
            tools=[
                ResponsesFunctionTool(
                    type="function",
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
        )
        result = responses_to_messages_request(req, "resolved-model")
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == "get_weather"
        assert result.tools[0].input_schema is not None

    def test_tool_choice_auto(self):
        req = ResponsesRequest(model="test", input="Hello", tool_choice="auto")
        result = responses_to_messages_request(req, "resolved-model")
        assert result.tool_choice == {"type": "auto"}

    def test_tool_choice_none(self):
        req = ResponsesRequest(model="test", input="Hello", tool_choice="none")
        result = responses_to_messages_request(req, "resolved-model")
        assert result.tool_choice == {"type": "none"}

    def test_tool_choice_required(self):
        req = ResponsesRequest(model="test", input="Hello", tool_choice="required")
        result = responses_to_messages_request(req, "resolved-model")
        assert result.tool_choice == {"type": "any"}

    def test_tool_choice_named_function(self):
        req = ResponsesRequest(
            model="test",
            input="Hello",
            tool_choice={"type": "function", "name": "get_weather"},
        )
        result = responses_to_messages_request(req, "resolved-model")
        assert result.tool_choice == {"type": "tool", "name": "get_weather"}

    def test_max_output_tokens_maps_to_max_tokens(self):
        req = ResponsesRequest(model="test", input="Hello", max_output_tokens=100)
        result = responses_to_messages_request(req, "resolved-model")
        assert result.max_tokens == 100

    def test_stream_forced_true(self):
        """Internal MessagesRequest must always have stream=True for provider."""
        req = ResponsesRequest(model="test", input="Hello", stream=False)
        result = responses_to_messages_request(req, "resolved-model")
        assert result.stream is True

    def test_multi_turn_conversation(self):
        """Full multi-turn: user -> assistant function_call -> function_call_output."""
        req = ResponsesRequest(
            model="test",
            input=[
                EasyInputMessage(role="user", content="What is the weather?"),
                FunctionCall(
                    type="function_call",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"city": "NYC"}',
                ),
                FunctionCallOutput(
                    type="function_call_output",
                    call_id="call_1",
                    output="Sunny, 72F",
                ),
            ],
        )
        result = responses_to_messages_request(req, "resolved-model")
        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"
        assert isinstance(result.messages[1].content[0], ContentBlockToolUse)
        assert isinstance(result.messages[2].content[0], ContentBlockToolResult)


class TestResponsesSSEConversion:
    """Verify Anthropic SSE events convert to Responses API streaming events."""

    @pytest.mark.asyncio
    async def test_text_response_stream(self):
        """Full text-only Anthropic SSE stream -> Responses API events."""
        anthropic_chunks = [
            (
                "event: message_start\n"
                'data: {"type": "message_start", "message": {"id": "msg_test123", "type": "message", '
                '"role": "assistant", "content": [], "model": "test-model", "stop_reason": null, '
                '"stop_sequence": null, "usage": {"input_tokens": 5, "output_tokens": 1}}}\n\n'
            ),
            (
                "event: content_block_start\n"
                'data: {"type": "content_block_start", "index": 0, '
                '"content_block": {"type": "text", "text": ""}}\n\n'
            ),
            (
                "event: content_block_delta\n"
                'data: {"type": "content_block_delta", "index": 0, '
                '"delta": {"type": "text_delta", "text": "Hello"}}\n\n'
            ),
            (
                "event: content_block_delta\n"
                'data: {"type": "content_block_delta", "index": 0, '
                '"delta": {"type": "text_delta", "text": " world"}}\n\n'
            ),
            (
                "event: content_block_stop\n"
                'data: {"type": "content_block_stop", "index": 0}\n\n'
            ),
            (
                "event: message_delta\n"
                'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, '
                '"usage": {"input_tokens": 5, "output_tokens": 10}}\n\n'
            ),
            'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]

        async def source():
            for chunk in anthropic_chunks:
                yield chunk

        events = []
        async for chunk in convert_anthropic_sse_to_responses_stream(
            source(), "test-model", request_id="resp_test123"
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload and payload != "[DONE]":
                        events.append(json.loads(payload))

        event_types = [e.get("type") for e in events]
        assert "response.created" in event_types
        assert "response.in_progress" in event_types
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.output_text.done" in event_types
        assert "response.content_part.done" in event_types
        assert "response.output_item.done" in event_types
        assert "response.completed" in event_types

    @pytest.mark.asyncio
    async def test_tool_use_response_stream(self):
        """Anthropic tool_use SSE -> Responses API function_call events."""
        anthropic_chunks = [
            (
                "event: message_start\n"
                'data: {"type": "message_start", "message": {"id": "msg_tool123", "type": "message", '
                '"role": "assistant", "content": [], "model": "test-model", "stop_reason": null, '
                '"stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}}\n\n'
            ),
            (
                "event: content_block_start\n"
                'data: {"type": "content_block_start", "index": 0, '
                '"content_block": {"type": "tool_use", "id": "toolu_abc123", '
                '"name": "get_weather", "input": {}}}\n\n'
            ),
            (
                "event: content_block_delta\n"
                'data: {"type": "content_block_delta", "index": 0, '
                '"delta": {"type": "input_json_delta", "partial_json": "{\\\"city\\\": \\\"NYC\\\"}"}}\n\n'
            ),
            (
                "event: content_block_stop\n"
                'data: {"type": "content_block_stop", "index": 0}\n\n'
            ),
            (
                "event: message_delta\n"
                'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, '
                '"usage": {"input_tokens": 10, "output_tokens": 20}}\n\n'
            ),
            'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]

        async def source():
            for chunk in anthropic_chunks:
                yield chunk

        events = []
        async for chunk in convert_anthropic_sse_to_responses_stream(
            source(), "test-model", request_id="resp_tool123"
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload and payload != "[DONE]":
                        events.append(json.loads(payload))

        event_types = [e.get("type") for e in events]
        assert "response.output_item.added" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types
        assert "response.completed" in event_types

        # Check completed response has function_call in output
        completed = next(e for e in events if e.get("type") == "response.completed")
        resp = completed.get("response", {})
        assert resp.get("status") == "completed"
