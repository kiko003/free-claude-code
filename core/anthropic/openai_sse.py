"""OpenAI-format SSE builder and Anthropic-to-OpenAI SSE converter."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

OPENAI_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

_STOP_REASON_TO_OPENAI = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


def _map_stop_reason(anthropic_stop_reason: str | None) -> str | None:
    if anthropic_stop_reason is None:
        return None
    return _STOP_REASON_TO_OPENAI.get(anthropic_stop_reason, "stop")


def build_openai_stream_chunk(
    *,
    message_id: str,
    model: str,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> str:
    """Build one OpenAI-format streaming SSE chunk."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    choice: dict[str, Any] = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    }

    data = {
        "id": message_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(data)}\n\n"


def build_openai_stream_done() -> str:
    """Return the OpenAI stream terminator."""
    return "data: [DONE]\n\n"


def build_openai_non_stream_response(
    *,
    message_id: str,
    model: str,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Build a complete OpenAI non-streaming response."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": message_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_text_completion_stream_chunk(
    *,
    completion_id: str,
    model: str,
    text: str | None = None,
    finish_reason: str | None = None,
) -> str:
    """Build one legacy text completion streaming SSE chunk."""
    choice: dict[str, Any] = {
        "text": text if text is not None else "",
        "index": 0,
        "logprobs": None,
        "finish_reason": finish_reason,
    }

    data = {
        "id": completion_id,
        "object": "text_completion",
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(data)}\n\n"


def build_text_completion_non_stream_response(
    *,
    completion_id: str,
    model: str,
    text: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Build a complete legacy text completion non-streaming response."""
    return {
        "id": completion_id,
        "object": "text_completion",
        "model": model,
        "choices": [
            {
                "text": text or "",
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def convert_anthropic_sse_to_text_completion_stream(
    anthropic_stream: AsyncIterator[str],
    model: str,
    *,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Convert an Anthropic SSE stream to a legacy text completion SSE stream."""
    return _TextCompletionStreamConverter(
        anthropic_stream, model, request_id=request_id
    ).run()


class _TextCompletionStreamConverter:
    """Stateful converter from Anthropic SSE events to legacy text completion SSE chunks."""

    def __init__(
        self,
        anthropic_stream: AsyncIterator[str],
        model: str,
        *,
        request_id: str | None = None,
    ):
        self._source = anthropic_stream
        self._model = model
        self._completion_id = request_id or f"cmpl-{uuid.uuid4().hex[:24]}"
        self._text_buffer: list[str] = []
        self._stop_reason: str | None = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._started: bool = False
        self._stream_done: bool = False

    async def run(self) -> AsyncIterator[str]:
        async for chunk in self._source:
            parsed = _parse_anthropic_sse_event(chunk)
            if parsed is None:
                continue
            event_type, data = parsed

            if event_type == "message_start":
                msg = data.get("message", {})
                self._completion_id = msg.get("id", self._completion_id)
                usage = msg.get("usage", {})
                self._input_tokens = usage.get("input_tokens", 0)
                if not self._started:
                    self._started = True
                    yield build_text_completion_stream_chunk(
                        completion_id=self._completion_id,
                        model=self._model,
                    )

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        self._text_buffer.append(text)
                        yield build_text_completion_stream_chunk(
                            completion_id=self._completion_id,
                            model=self._model,
                            text=text,
                        )

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                self._stop_reason = delta.get("stop_reason")
                usage = data.get("usage", {})
                if usage:
                    self._output_tokens = usage.get("output_tokens", 0)

            elif event_type == "message_stop":
                self._stream_done = True
                yield build_text_completion_stream_chunk(
                    completion_id=self._completion_id,
                    model=self._model,
                    finish_reason=_map_stop_reason(self._stop_reason),
                )
                yield build_openai_stream_done()

            elif event_type == "error":
                error_info = data.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                self._stream_done = True
                yield build_text_completion_stream_chunk(
                    completion_id=self._completion_id,
                    model=self._model,
                    text=f"\n[Error: {error_msg}]",
                    finish_reason="stop",
                )
                yield build_openai_stream_done()

        if not self._stream_done:
            yield build_text_completion_stream_chunk(
                completion_id=self._completion_id,
                model=self._model,
                finish_reason=_map_stop_reason(self._stop_reason),
            )
            yield build_openai_stream_done()


def _parse_anthropic_sse_event(chunk: str) -> tuple[str, dict[str, Any]] | None:
    """Parse an Anthropic SSE chunk into (event_type, data) or None if not a data event."""
    event_type = "message"
    data: dict[str, Any] | None = None
    for line in chunk.splitlines():
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError, ValueError:
                return None
    if data is None:
        return None
    return event_type, data


def convert_anthropic_sse_to_openai_stream(
    anthropic_stream: AsyncIterator[str],
    model: str,
    *,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Convert an Anthropic SSE stream to an OpenAI SSE stream.

    This is an async generator wrapper. It yields OpenAI-format SSE strings.
    Usage:
        async for chunk in convert_anthropic_sse_to_openai_stream(provider_stream, model):
            yield chunk
    """
    return _OpenAIStreamConverter(anthropic_stream, model, request_id=request_id).run()


class _OpenAIStreamConverter:
    """Stateful converter from Anthropic SSE events to OpenAI SSE chunks."""

    def __init__(
        self,
        anthropic_stream: AsyncIterator[str],
        model: str,
        *,
        request_id: str | None = None,
    ):
        self._source = anthropic_stream
        self._model = model
        self._message_id = request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self._content_buffer: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._tool_call_buffers: dict[int, str] = {}
        self._stop_reason: str | None = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._started: bool = False
        self._stream_done: bool = False

    async def run(self) -> AsyncIterator[str]:
        async for chunk in self._source:
            parsed = _parse_anthropic_sse_event(chunk)
            if parsed is None:
                continue
            event_type, data = parsed

            if event_type == "message_start":
                msg = data.get("message", {})
                self._message_id = msg.get("id", self._message_id)
                usage = msg.get("usage", {})
                self._input_tokens = usage.get("input_tokens", 0)
                if not self._started:
                    self._started = True
                    yield self._openai_chunk()

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                block_type = block.get("type")
                if block_type == "tool_use":
                    idx = data.get("index", 0)
                    tc = {
                        "index": idx,
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": "",
                        },
                    }
                    if idx < len(self._tool_calls):
                        self._tool_calls[idx] = tc
                    else:
                        self._tool_calls.append(tc)

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        self._content_buffer.append(text)
                        yield self._openai_chunk(content=text)
                elif delta_type == "input_json_delta":
                    idx = data.get("index", 0)
                    partial = delta.get("partial_json", "")
                    if idx < len(self._tool_calls):
                        self._tool_call_buffers[idx] = (
                            self._tool_call_buffers.get(idx, "") + partial
                        )
                        self._tool_calls[idx]["function"]["arguments"] = (
                            self._tool_call_buffers[idx]
                        )
                        yield self._openai_chunk(tool_calls=[self._tool_calls[idx]])

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                self._stop_reason = delta.get("stop_reason")
                usage = data.get("usage", {})
                if usage:
                    self._output_tokens = usage.get("output_tokens", 0)

            elif event_type == "message_stop":
                self._stream_done = True
                yield self._openai_chunk(
                    finish_reason=_map_stop_reason(self._stop_reason),
                )
                yield build_openai_stream_done()

            elif event_type == "error":
                error_info = data.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                self._stream_done = True
                yield self._openai_chunk(
                    content=f"\n[Error: {error_msg}]",
                    finish_reason="stop",
                )
                yield build_openai_stream_done()

        if not self._stream_done:
            yield self._openai_chunk(
                finish_reason=_map_stop_reason(self._stop_reason),
            )
            yield build_openai_stream_done()

    def _openai_chunk(
        self,
        *,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str | None = None,
    ) -> str:
        return build_openai_stream_chunk(
            message_id=self._message_id,
            model=self._model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
