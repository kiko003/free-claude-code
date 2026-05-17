"""Pydantic models for OpenAI-compatible /chat/completions requests."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class OpenAIImageUrl(BaseModel):
    url: str
    detail: str | None = "auto"


class OpenAIContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: OpenAIImageUrl | None = None


class OpenAIFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class OpenAIToolCall(BaseModel):
    id: str | None = None
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall | None = None


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[OpenAIContentPart] | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class OpenAIFunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec


class OpenAIToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: dict[str, str]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = True
    tools: list[OpenAITool] | None = None
    tool_choice: str | OpenAIToolChoice | None = None
    n: int | None = 1


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = True
    n: int | None = 1
    suffix: str | None = None
    echo: bool | None = None
    best_of: int | None = None
    logprobs: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
