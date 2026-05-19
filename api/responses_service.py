"""Responses API service — convert /v1/responses requests to Anthropic format and stream back."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from api.model_router import ModelRouter
from api.models.anthropic import (
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
    MessagesRequest,
    Tool,
)
from api.models.responses_api import (
    EasyInputMessage,
    FunctionCall,
    FunctionCallOutput,
    ResponsesFunctionTool,
    ResponsesRequest,
)
from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.responses_sse import (
    RESPONSES_SSE_RESPONSE_HEADERS,
    convert_anthropic_sse_to_responses_stream,
)
from core.trace import trace_event, traced_async_stream
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

ProviderGetter = Callable[[str], BaseProvider]
TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]


def _convert_input_item_to_messages(
    item: EasyInputMessage | FunctionCall | FunctionCallOutput,
    messages: list[Message],
) -> None:
    """Convert a single Responses API input item and append to the messages list.

    The Responses API input list is linear — each item maps to either a user
    or assistant Anthropic message.  Function calls become assistant tool_use
    blocks; function call outputs become user tool_result blocks.
    """
    if isinstance(item, EasyInputMessage):
        role = item.role if item.role in ("user", "assistant") else "user"
        content = item.content
        if isinstance(content, str):
            messages.append(Message(role=role, content=content))
        elif isinstance(content, list):
            blocks: list[Any] = [
                ContentBlockText(type="text", text=part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "input_text")
            ] or [ContentBlockText(type="text", text="")]
            messages.append(Message(role=role, content=blocks))
    elif isinstance(item, FunctionCall):
        try:
            arguments = json.loads(item.arguments) if item.arguments else {}
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_use = ContentBlockToolUse(
            type="tool_use",
            id=item.call_id,
            name=item.name,
            input=arguments,
        )
        messages.append(Message(role="assistant", content=[tool_use]))
    elif isinstance(item, FunctionCallOutput):
        tool_result = ContentBlockToolResult(
            type="tool_result",
            tool_use_id=item.call_id,
            content=item.output,
        )
        messages.append(Message(role="user", content=[tool_result]))


def _convert_responses_tools(tools: list[ResponsesFunctionTool] | None) -> list[Tool] | None:
    """Convert Responses API function tools to Anthropic Tool list."""
    if not tools:
        return None
    return [
        Tool(
            name=t.name,
            description=t.description,
            input_schema=t.parameters,
        )
        for t in tools
    ]


def _convert_responses_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert Responses API tool_choice to Anthropic tool_choice format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "required":
            return {"type": "any"}
        return None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "name" in tool_choice:
        return {"type": "tool", "name": tool_choice["name"]}
    return None


def responses_to_messages_request(
    req: ResponsesRequest,
    resolved_model: str,
) -> MessagesRequest:
    """Convert a Responses API request to an internal MessagesRequest."""
    if isinstance(req.input, str):
        messages = [Message(role="user", content=req.input)]
    else:
        messages: list[Message] = []
        for item in req.input:
            _convert_input_item_to_messages(item, messages)

    return MessagesRequest(
        model=resolved_model,
        messages=messages,
        system=req.instructions,
        max_tokens=req.max_output_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=True,
        tools=_convert_responses_tools(req.tools),
        tool_choice=_convert_responses_tool_choice(req.tool_choice),
    )


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
        import traceback

        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error("{} request_id={} exc_type={}", context, request_id, type(exc).__name__)
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


class ResponsesProxyService:
    """Coordinate Responses API request routing, conversion, and streaming."""

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

    async def create_response(
        self,
        request_data: ResponsesRequest,
    ) -> StreamingResponse | dict[str, Any]:
        """Handle a /v1/responses request."""
        try:
            if not request_data.input:
                raise InvalidRequestError("input cannot be empty")

            resolved = self._model_router.resolve(request_data.model)
            internal_request = responses_to_messages_request(
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

            request_id = f"resp_{uuid.uuid4().hex[:12]}"
            with logger.contextualize(request_id=request_id):
                trace_event(
                    stage="ingress",
                    event="api.request.received",
                    source="api",
                    kind="responses",
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

                responses_stream = convert_anthropic_sse_to_responses_stream(
                    anthropic_stream,
                    resolved.provider_model,
                    request_id=request_id,
                )

                if request_data.stream is False:
                    return await self._collect_non_stream(
                        responses_stream, resolved, input_tokens, request_id
                    )

                return StreamingResponse(
                    responses_stream,
                    media_type="text/event-stream",
                    headers=RESPONSES_SSE_RESPONSE_HEADERS,
                )

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_RESPONSE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    async def _collect_non_stream(
        self,
        responses_stream: AsyncIterator[str],
        resolved: Any,
        input_tokens: int,
        request_id: str,
    ) -> dict[str, Any]:
        """Collect Responses SSE stream into a single non-streaming response."""
        text_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        response_id: str = request_id
        input_tokens_val: int = input_tokens
        output_tokens_val: int = 0
        status: str = "completed"

        async for chunk in responses_stream:
            if chunk.startswith("data: "):
                payload = chunk[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue

                event_type = data.get("type", "")

                if event_type == "response.created":
                    response_id = data.get("response", {}).get("id", response_id)
                elif event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        text_parts.append(delta)
                elif event_type == "response.function_call_arguments.delta":
                    delta = data.get("delta", "")
                    if function_calls and delta:
                        function_calls[-1]["arguments"] += delta
                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        function_calls.append(
                            {
                                "type": "function_call",
                                "id": item.get("id", f"fc_{uuid.uuid4().hex[:12]}"),
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                                "status": "completed",
                            }
                        )
                elif event_type == "response.completed":
                    resp = data.get("response", {})
                    usage = resp.get("usage", {})
                    output_tokens_val = usage.get("output_tokens", 0)
                elif event_type == "response.failed":
                    status = "failed"

        # Build the response object
        output: list[dict[str, Any]] = []
        if text_parts:
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "".join(text_parts), "annotations": []}
                    ],
                    "status": "completed",
                }
            )
        output.extend(function_calls)

        return {
            "id": response_id,
            "object": "response",
            "model": resolved.provider_model,
            "status": status,
            "output": output,
            "usage": {
                "input_tokens": input_tokens_val,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output_tokens_val,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens_val + output_tokens_val,
            },
        }
