"""Pydantic models for OpenAI Responses API /v1/responses requests."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class EasyInputMessage(BaseModel):
    """Simple message input item — {role, content}."""

    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[dict[str, Any]]


class FunctionCall(BaseModel):
    """Prior assistant tool call for multi-turn."""

    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class FunctionCallOutput(BaseModel):
    """Tool result for multi-turn."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


InputItem = EasyInputMessage | FunctionCall | FunctionCallOutput


class ResponsesFunctionTool(BaseModel):
    """Function tool definition for the Responses API."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[InputItem]
    instructions: str | None = None
    tools: list[ResponsesFunctionTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    metadata: dict[str, str] | None = None
    previous_response_id: str | None = None
