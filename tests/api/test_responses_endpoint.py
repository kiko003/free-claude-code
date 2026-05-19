"""Tests for the /v1/responses endpoint."""

from api.models.responses_api import (
    EasyInputMessage,
    FunctionCall,
    FunctionCallOutput,
    ResponsesFunctionTool,
    ResponsesRequest,
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
            input=[{"role": "user", "content": "Hello"}],
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
