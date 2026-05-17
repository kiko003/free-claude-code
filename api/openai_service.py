"""OpenAI-compatible /chat/completions service."""

from __future__ import annotations

import json
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from api.models.anthropic import (
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
    MessagesRequest,
    SystemContent,
    Tool,
    ThinkingConfig,
)
from api.models.openai import (
    ChatCompletionRequest,
    OpenAIContentPart,
    OpenAIMessage,
    OpenAITool,
    OpenAIToolChoice,
)
from api.model_router import ModelRouter
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.openai_sse import (
    OPENAI_SSE_RESPONSE_HEADERS,
    build_openai_non_stream_response,
    convert_anthropic_sse_to_openai_stream,
)
from core.trace import trace_event, traced_async_stream
from config.settings import Settings
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

ProviderGetter = Callable[[str], BaseProvider]
TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _convert_openai_message(msg: OpenAIMessage) -> Message:
    """Convert a single OpenAI message to an Anthropic Message."""
    role = msg.role
    content = msg.content

    if role == "system":
        return Message(role="user", content="")

    if role == "tool":
        tool_call_id = msg.tool_call_id or ""
        if isinstance(content, str):
            tool_content: Any = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, OpenAIContentPart):
                    if part.type == "text" and part.text:
                        parts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            tool_content = "\n".join(parts)
        else:
            tool_content = content or ""
        return Message(
            role="user",
            content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id=tool_call_id,
                    content=tool_content,
                )
            ],
        )

    if isinstance(content, str):
        return Message(role=role, content=content)

    if isinstance(content, list):
        blocks: list[Any] = []
        for part in content:
            if isinstance(part, OpenAIContentPart):
                if part.type == "text" and part.text:
                    blocks.append(ContentBlockText(type="text", text=part.text))
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    blocks.append(ContentBlockText(type="text", text=part.get("text", "")))
        if not blocks:
            blocks.append(ContentBlockText(type="text", text=content or ""))
        return Message(role=role, content=blocks)

    return Message(role=role, content=content or "")


def _convert_openai_tool_calls(msg: OpenAIMessage) -> list[Any]:
    """Extract tool_use blocks from an OpenAI assistant message."""
    if not msg.tool_calls:
        return []
    blocks: list[Any] = []
    for tc in msg.tool_calls:
        if tc.function:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            blocks.append(
                ContentBlockToolUse(
                    type="tool_use",
                    id=tc.id or f"tool_{uuid.uuid4().hex[:12]}",
                    name=tc.function.name or "",
                    input=arguments,
                )
            )
    return blocks


def _convert_openai_messages(req: ChatCompletionRequest) -> list[Message]:
    """Convert OpenAI messages to Anthropic Message list."""
    result: list[Message] = []
    pending_assistant_tool_calls: list[Any] = []

    for msg in req.messages:
        if msg.role == "assistant" and msg.tool_calls:
            tool_use_blocks = _convert_openai_tool_calls(msg)
            content_blocks: list[Any] = []
            if msg.content:
                if isinstance(msg.content, str) and msg.content:
                    content_blocks.append(ContentBlockText(type="text", text=msg.content))
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, OpenAIContentPart):
                            if part.type == "text" and part.text:
                                content_blocks.append(ContentBlockText(type="text", text=part.text))
                        elif isinstance(part, dict) and part.get("type") == "text":
                            content_blocks.append(
                                ContentBlockText(type="text", text=part.get("text", ""))
                            )
            content_blocks.extend(tool_use_blocks)
            result.append(Message(role="assistant", content=content_blocks))
            pending_assistant_tool_calls = tool_use_blocks
            continue

        if msg.role == "tool" and pending_assistant_tool_calls:
            converted = _convert_openai_message(msg)
            result.append(converted)
            pending_assistant_tool_calls = []
            continue

        converted = _convert_openai_message(msg)
        if converted.content or converted.role == "assistant":
            result.append(converted)
        pending_assistant_tool_calls = []

    return result


def _convert_openai_tools(req: ChatCompletionRequest) -> list[Tool] | None:
    """Convert OpenAI tools to Anthropic Tool list."""
    if not req.tools:
        return None
    result: list[Tool] = []
    for t in req.tools:
        result.append(
            Tool(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,
            )
        )
    return result


def _convert_openai_tool_choice(req: ChatCompletionRequest) -> dict[str, Any] | None:
    """Convert OpenAI tool_choice to Anthropic tool_choice format."""
    if not req.tool_choice:
        return None
    if isinstance(req.tool_choice, str):
        if req.tool_choice == "auto":
            return {"type": "auto"}
        if req.tool_choice == "none":
            return {"type": "none"}
        if req.tool_choice == "required":
            return {"type": "any"}
        return None
    if isinstance(req.tool_choice, OpenAIToolChoice):
        if req.tool_choice.type == "function" and req.tool_choice.function:
            return {"type": "tool", "name": req.tool_choice.function.get("name", "")}
    return None


def _convert_openai_stop(req: ChatCompletionRequest) -> list[str] | None:
    """Convert OpenAI stop to Anthropic stop_sequences."""
    if not req.stop:
        return None
    if isinstance(req.stop, str):
        return [req.stop]
    return req.stop


def _extract_system_messages(req: ChatCompletionRequest) -> str | list[SystemContent] | None:
    """Extract system messages from OpenAI messages list."""
    system_parts: list[str] = []
    for msg in req.messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, OpenAIContentPart):
                        if part.type == "text" and part.text:
                            system_parts.append(part.text)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
    if not system_parts:
        return None
    if len(system_parts) == 1:
        return system_parts[0]
    return [SystemContent(type="text", text=p) for p in system_parts]


def _filter_non_system_messages(req: ChatCompletionRequest) -> list[OpenAIMessage]:
    """Return messages excluding system role messages."""
    return [m for m in req.messages if m.role != "system"]


def openai_to_messages_request(
    req: ChatCompletionRequest,
    resolved_model: str,
) -> MessagesRequest:
    """Convert an OpenAI ChatCompletionRequest to an internal MessagesRequest."""
    non_system_messages = _filter_non_system_messages(req)
    system = _extract_system_messages(req)
    messages = _convert_openai_messages(
        ChatCompletionRequest(
            model=req.model,
            messages=non_system_messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            stream=req.stream,
            tools=req.tools,
            tool_choice=req.tool_choice,
            n=req.n,
        )
    )

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "system": system,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stop_sequences": _convert_openai_stop(req),
        "stream": True,
        "tools": _convert_openai_tools(req),
        "tool_choice": _convert_openai_tool_choice(req),
    }

    return MessagesRequest(**kwargs)


class OpenAIProxyService:
    """Coordinate OpenAI-format request routing, conversion, and streaming."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def create_chat_completion(
        self, request_data: ChatCompletionRequest
    ) -> StreamingResponse | dict[str, Any]:
        """Handle an OpenAI /chat/completions request."""
        try:
            if not request_data.messages:
                raise InvalidRequestError("messages cannot be empty")

            resolved = self._model_router.resolve(request_data.model)
            internal_request = openai_to_messages_request(
                request_data, resolved.provider_model
            )

            provider = self._provider_getter(resolved.provider_id)
            provider.preflight_stream(
                internal_request,
                thinking_enabled=resolved.thinking_enabled,
            )

            trace_event(
                stage="routing",
                event="api.route.resolved",
                source="api",
                provider_id=resolved.provider_id,
                provider_model=resolved.provider_model,
                provider_model_ref=resolved.provider_model_ref,
                gateway_model=request_data.model,
                thinking_enabled=resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            with logger.contextualize(request_id=request_id):
                trace_event(
                    stage="ingress",
                    event="api.request.received",
                    source="api",
                    kind="chat_completions",
                    message_count=len(request_data.messages),
                )

                input_tokens = self._token_counter(
                    internal_request.messages,
                    internal_request.system,
                    internal_request.tools,
                )

                anthropic_stream = traced_async_stream(
                    provider.stream_response(
                        internal_request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=resolved.thinking_enabled,
                    ),
                    stage="egress",
                    source="api",
                    complete_event="api.response.stream_completed",
                    interrupted_event="api.response.stream_interrupted",
                    chunk_event=None,
                    extra={
                        "request_id": request_id,
                        "provider_id": resolved.provider_id,
                        "gateway_model": request_data.model,
                    },
                )

                openai_stream = convert_anthropic_sse_to_openai_stream(
                    anthropic_stream,
                    resolved.provider_model,
                    request_id=request_id,
                )

                if request_data.stream is False:
                    return self._collect_non_stream(
                        openai_stream, resolved, input_tokens, request_id
                    )

                return StreamingResponse(
                    openai_stream,
                    media_type="text/event-stream",
                    headers=OPENAI_SSE_RESPONSE_HEADERS,
                )

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_CHAT_COMPLETION_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    async def _collect_non_stream(
        self,
        openai_stream: AsyncIterator[str],
        resolved: Any,
        input_tokens: int,
        request_id: str,
    ) -> dict[str, Any]:
        """Collect streaming chunks into a single non-streaming response."""
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        message_id: str = ""

        async for chunk in openai_stream:
            if chunk.startswith("data: "):
                payload = chunk[6:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not message_id:
                    message_id = data.get("id", "")
                choices = data.get("choices", [])
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            if tc.get("type"):
                                tool_calls[idx]["type"] = tc["type"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_calls[idx]["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

        return build_openai_non_stream_response(
            message_id=message_id or request_id,
            model=resolved.provider_model,
            content="".join(content_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            prompt_tokens=input_tokens,
            completion_tokens=len(content_parts),
        )
