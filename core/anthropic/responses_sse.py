"""Responses API SSE builder and Anthropic-to-Responses SSE converter."""

from collections.abc import AsyncIterator
from typing import Any

RESPONSES_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


async def convert_anthropic_sse_to_responses_stream(
    anthropic_stream: AsyncIterator[str],
    model: str,
    *,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Stub — will be implemented in Task 4."""
    return
    yield  # make this an async generator
