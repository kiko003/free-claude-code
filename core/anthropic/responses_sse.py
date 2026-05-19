# core/anthropic/responses_sse.py
"""Responses API SSE builder and Anthropic-to-Responses SSE converter."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

RESPONSES_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def build_responses_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Build one Responses API SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _parse_anthropic_sse_event(chunk: str) -> tuple[str, dict[str, Any]] | None:
    """Parse an Anthropic SSE chunk into (event_type, data) or None."""
    event_type = "message"
    data: dict[str, Any] | None = None
    for line in chunk.splitlines():
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
            except (json.JSONDecodeError, ValueError):
                return None
    if data is None:
        return None
    return event_type, data


def _make_response_object(
    *,
    response_id: str,
    model: str,
    status: str = "in_progress",
    output: list[dict[str, Any]] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    """Build a Responses API response object (used in created/completed events)."""
    return {
        "id": response_id,
        "object": "response",
        "model": model,
        "status": status,
        "output": output or [],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
    }


def convert_anthropic_sse_to_responses_stream(
    anthropic_stream: AsyncIterator[str],
    model: str,
    *,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Convert an Anthropic SSE stream to a Responses API SSE stream."""
    return _ResponsesStreamConverter(
        anthropic_stream, model, request_id=request_id
    ).run()


class _ResponsesStreamConverter:
    """Stateful converter from Anthropic SSE events to Responses API SSE events."""

    def __init__(
        self,
        anthropic_stream: AsyncIterator[str],
        model: str,
        *,
        request_id: str | None = None,
    ):
        self._source = anthropic_stream
        self._model = model
        self._response_id = request_id or f"resp_{uuid.uuid4().hex[:24]}"
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._text_buffer: list[str] = []
        self._current_text: str = ""
        self._function_calls: list[dict[str, Any]] = []
        self._current_function_call: dict[str, Any] | None = None
        self._stop_reason: str | None = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._started: bool = False
        self._stream_done: bool = False
        self._message_output_emitted: bool = False

    async def run(self) -> AsyncIterator[str]:
        async for chunk in self._source:
            parsed = _parse_anthropic_sse_event(chunk)
            if parsed is None:
                continue
            event_type, data = parsed

            if event_type == "message_start":
                msg = data.get("message", {})
                self._input_tokens = msg.get("usage", {}).get("input_tokens", 0)
                if not self._started:
                    self._started = True
                    yield build_responses_sse_event(
                        "response.created",
                        {
                            "response": _make_response_object(
                                response_id=self._response_id,
                                model=self._model,
                                status="in_progress",
                                input_tokens=self._input_tokens,
                            ),
                            "type": "response.created",
                        },
                    )
                    yield build_responses_sse_event(
                        "response.in_progress",
                        {
                            "response": _make_response_object(
                                response_id=self._response_id,
                                model=self._model,
                                status="in_progress",
                                input_tokens=self._input_tokens,
                            ),
                            "type": "response.in_progress",
                        },
                    )

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                block_type = block.get("type")
                if block_type == "text":
                    if not self._message_output_emitted:
                        self._message_output_emitted = True
                        yield build_responses_sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "type": "message",
                                    "id": self._message_id,
                                    "role": "assistant",
                                    "content": [],
                                    "status": "in_progress",
                                },
                            },
                        )
                    yield build_responses_sse_event(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        },
                    )
                elif block_type == "tool_use":
                    fc_id = f"fc_{uuid.uuid4().hex[:12]}"
                    self._current_function_call = {
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": "",
                        "status": "in_progress",
                    }
                    fc_index = len(self._function_calls)
                    if not self._message_output_emitted:
                        self._message_output_emitted = True
                        yield build_responses_sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": fc_index
                                + (
                                    1
                                    if self._text_buffer or self._current_text
                                    else 0
                                ),
                                "item": self._current_function_call,
                            },
                        )

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        self._current_text += text
                        yield build_responses_sse_event(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": 0,
                                "content_index": 0,
                                "delta": text,
                            },
                        )
                elif delta_type == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    if self._current_function_call is not None:
                        self._current_function_call["arguments"] += partial
                        yield build_responses_sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "output_index": len(self._function_calls) + 1,
                                "item_id": self._current_function_call["id"],
                                "delta": partial,
                            },
                        )

            elif event_type == "content_block_stop":
                block = data.get("content_block", {})
                block_type = block.get("type") if block else None
                if (
                    self._current_function_call is not None
                    and block_type != "text"
                ):
                    self._current_function_call["status"] = "completed"
                    self._function_calls.append(self._current_function_call)
                    fc_index = len(self._function_calls) - 1
                    fc_output_index = fc_index + (
                        1 if self._text_buffer or self._current_text else 0
                    )
                    yield build_responses_sse_event(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "output_index": fc_output_index,
                            "item_id": self._current_function_call["id"],
                            "arguments": self._current_function_call["arguments"],
                        },
                    )
                    yield build_responses_sse_event(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": fc_output_index,
                            "item": self._current_function_call,
                        },
                    )
                    self._current_function_call = None
                else:
                    final_text = self._current_text
                    self._text_buffer.append(final_text)
                    self._current_text = ""
                    yield build_responses_sse_event(
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "output_index": 0,
                            "content_index": 0,
                            "text": final_text,
                        },
                    )
                    yield build_responses_sse_event(
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": final_text,
                                "annotations": [],
                            },
                        },
                    )

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                self._stop_reason = delta.get("stop_reason")
                usage = data.get("usage", {})
                if usage:
                    self._output_tokens = usage.get("output_tokens", 0)

            elif event_type == "message_stop":
                self._stream_done = True
                if self._text_buffer:
                    full_text = "".join(self._text_buffer)
                    yield build_responses_sse_event(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "id": self._message_id,
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": full_text,
                                        "annotations": [],
                                    }
                                ],
                                "status": "completed",
                            },
                        },
                    )
                output: list[dict[str, Any]] = []
                if self._text_buffer:
                    output.append(
                        {
                            "type": "message",
                            "id": self._message_id,
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "".join(self._text_buffer),
                                    "annotations": [],
                                }
                            ],
                            "status": "completed",
                        }
                    )
                output.extend(self._function_calls)
                yield build_responses_sse_event(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": _make_response_object(
                            response_id=self._response_id,
                            model=self._model,
                            status="completed",
                            output=output,
                            input_tokens=self._input_tokens,
                            output_tokens=self._output_tokens,
                        ),
                    },
                )

            elif event_type == "error":
                error_info = data.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                self._stream_done = True
                yield build_responses_sse_event(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": _make_response_object(
                            response_id=self._response_id,
                            model=self._model,
                            status="failed",
                            input_tokens=self._input_tokens,
                            output_tokens=self._output_tokens,
                        ),
                        "error": {
                            "message": error_msg,
                            "type": "server_error",
                            "code": None,
                        },
                    },
                )

        if not self._stream_done:
            output: list[dict[str, Any]] = []
            if self._text_buffer or self._current_text:
                text = "".join(self._text_buffer) + self._current_text
                output.append(
                    {
                        "type": "message",
                        "id": self._message_id,
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                    }
                )
            output.extend(self._function_calls)
            yield build_responses_sse_event(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": _make_response_object(
                        response_id=self._response_id,
                        model=self._model,
                        status="completed",
                        output=output,
                        input_tokens=self._input_tokens,
                        output_tokens=self._output_tokens,
                    ),
                },
            )
