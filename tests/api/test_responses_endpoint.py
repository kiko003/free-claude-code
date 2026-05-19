"""Tests for the /v1/responses endpoint."""

from api.models import (
    EasyInputMessage,
    FunctionCall,
    FunctionCallOutput,
    ResponsesFunctionTool,
    ResponsesRequest,
)
from api.models.anthropic import ContentBlockToolResult, ContentBlockToolUse, ContentBlockText, Message, MessagesRequest
from api.responses_service import responses_to_messages_request


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
