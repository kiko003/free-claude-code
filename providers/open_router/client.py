"""OpenRouter provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger

from core.anthropic import append_request_id, iter_provider_stream_error_sse_events
from core.anthropic.emitted_sse_tracker import EmittedNativeSseTracker
from core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    is_terminal_openrouter_done_event,
    parse_native_sse_event,
    transform_native_sse_block_event,
)
from core.trace import provider_native_messages_body_snapshot, trace_event
from providers.anthropic_messages import (
    AnthropicMessagesTransport,
    StreamChunkMode,
    _maybe_await_aclose,
)
from providers.base import ProviderConfig
from providers.defaults import OPENROUTER_DEFAULT_BASE
from providers.model_listing import (
    ProviderModelInfo,
    extract_openrouter_tool_model_ids,
    extract_openrouter_tool_model_infos,
)

from .key_manager import KeyState, mask_key
from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"


class OpenRouterProvider(AnthropicMessagesTransport):
    """OpenRouter provider using the native Anthropic-compatible messages API."""

    stream_chunk_mode: StreamChunkMode = "event"

    def __init__(self, config: ProviderConfig, *, timezone: str = "UTC"):
        super().__init__(
            config,
            provider_name="OPENROUTER",
            default_base_url=OPENROUTER_DEFAULT_BASE,
        )
        if "," in config.api_key:
            from .key_manager import OpenRouterKeyManager, parse_keys_config

            keys = parse_keys_config(config.api_key)
            self._key_manager: OpenRouterKeyManager | None = OpenRouterKeyManager(
                keys, timezone_name=timezone
            )
        else:
            self._key_manager = None
        self._current_key: KeyState | None = None

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and direct request dispatch."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _active_key(self) -> str:
        """Return the API key to use for the current request."""
        if self._key_manager is not None:
            if self._current_key is not None:
                return self._current_key.api_key
            return self._key_manager._keys[0].api_key
        return self._api_key

    def _request_headers(self) -> dict[str, str]:
        """Return OpenRouter's Anthropic-compatible messages headers."""
        key = self._active_key()
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        """Return OpenRouter's OpenAI-compatible model-list headers."""
        key = self._active_key()
        return {"Authorization": f"Bearer {key}"}

    async def _send_stream_request(
        self, body: dict, *, select_key: bool = True
    ) -> httpx.Response:
        """Create a streaming messages response.

        When *select_key* is True (default), pick a fresh key from the
        manager before sending.  When False, reuse whatever
        ``_current_key`` is already set — used by 5xx retries inside
        ``execute_with_retry`` which should backoff on the *same* key.
        """
        if self._key_manager is not None and select_key:
            from .key_manager import AllKeysExhaustedError

            key_state = self._key_manager.get_available_key()
            if key_state is None:
                raise AllKeysExhaustedError()
            self._current_key = key_state

        request = self._client.build_request(
            "POST",
            "/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _send_model_list_request(self) -> httpx.Response:
        """Query the provider endpoint that advertises available model ids."""
        if self._key_manager is not None:
            key_state = self._key_manager.get_available_key()
            if key_state is not None:
                self._current_key = key_state
        return await self._client.get(
            "/models",
            headers=self._model_list_headers(),
        )

    def _extract_model_ids_from_model_list_payload(
        self, payload: Any
    ) -> frozenset[str]:
        """Only advertise OpenRouter models that can run Claude Code tools."""
        return extract_openrouter_tool_model_ids(
            payload, provider_name=self._provider_name
        )

    def _extract_model_infos_from_model_list_payload(
        self, payload: Any
    ) -> frozenset[ProviderModelInfo]:
        """Advertise OpenRouter tool models with reasoning capability metadata."""
        return extract_openrouter_tool_model_infos(
            payload, provider_name=self._provider_name
        )

    def _new_stream_state(self, request: Any, *, thinking_enabled: bool) -> Any:
        """Create per-stream state for thinking block filtering."""
        return NativeSseBlockPolicyState()

    def _transform_stream_event(
        self,
        event: str,
        state: Any,
        *,
        thinking_enabled: bool,
    ) -> str | None:
        """Drop provider-specific terminal noise and hidden thinking events."""
        if isinstance(state, NativeSseBlockPolicyState):
            event_name, data_text = parse_native_sse_event(event)
            if state.message_stopped or is_terminal_openrouter_done_event(
                event_name, data_text
            ):
                return None
            if event_name == "message_stop":
                state.message_stopped = True

        if isinstance(state, NativeSseBlockPolicyState):
            return transform_native_sse_block_event(
                event, state, thinking_enabled=thinking_enabled
            )
        return event

    def _format_error_message(self, base_message: str, request_id: str | None) -> str:
        """Keep OpenRouter's existing request-id suffix format."""
        return append_request_id(base_message, request_id)

    def _emit_error_events(
        self,
        *,
        request: Any,
        input_tokens: int,
        error_message: str,
        sent_any_event: bool,
    ) -> Iterator[str]:
        """Emit the Anthropic SSE error shape expected by Claude clients."""
        yield from iter_provider_stream_error_sse_events(
            request=request,
            input_tokens=input_tokens,
            error_message=error_message,
            sent_any_event=sent_any_event,
            log_raw_sse_events=self._config.log_raw_sse_events,
        )

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream with key-switching retry on 429/402 for multi-key setups.

        When no key manager is active (single key), delegates to the base
        class implementation unchanged.  With a key manager, a 429 or 402
        response triggers key rotation instead of same-key backoff — the
        blocked key is reported and the next available key is tried.
        5xx errors still use ``execute_with_retry``'s built-in same-key
        backoff-retry.
        """
        if self._key_manager is None:
            async for chunk in super().stream_response(
                request,
                input_tokens,
                request_id=request_id,
                thinking_enabled=thinking_enabled,
            ):
                yield chunk
            return

        from .key_manager import AllKeysExhaustedError

        tag = self._provider_name
        req_tag = f" request_id={request_id}" if request_id else ""
        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)

        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            gateway_model=request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_native_messages_body_snapshot(body),
        )

        max_attempts = len(self._key_manager._keys) * 2

        for _attempt in range(max_attempts):
            key_state = self._key_manager.get_available_key()
            if key_state is None:
                error_message = self._format_error_message(
                    "All OpenRouter keys exhausted", request_id
                )
                for event in self._emit_error_events(
                    request=request,
                    input_tokens=input_tokens,
                    error_message=error_message,
                    sent_any_event=False,
                ):
                    yield event
                return

            self._current_key = key_state
            active_key = key_state.api_key

            response: httpx.Response | None = None
            sent_any_event = False
            state = self._new_stream_state(request, thinking_enabled=thinking_enabled)
            emitted_tracker = EmittedNativeSseTracker()

            async with self._global_rate_limiter.concurrency_slot():
                try:

                    async def _send_with_key() -> httpx.Response:
                        """Send request; return 429/402 without raising so
                        the outer loop can switch keys.  Raise on 5xx so
                        execute_with_retry's backoff-retry applies.  Raise
                        on other 4xx so the error handler emits events.
                        """
                        send_response = await self._send_stream_request(
                            body, select_key=False
                        )
                        if send_response.status_code in (429, 402):
                            return send_response
                        if send_response.status_code != 200:
                            try:
                                await self._raise_for_status(
                                    send_response, req_tag=req_tag
                                )
                            finally:
                                if not send_response.is_closed:
                                    await _maybe_await_aclose(send_response)
                        return send_response

                    response = await self._global_rate_limiter.execute_with_retry(
                        _send_with_key
                    )

                    # 429/402: report and try next key
                    if response.status_code in (429, 402):
                        try:
                            error_body = await response.aread()
                            self._key_manager.report_rate_limit(
                                active_key,
                                error_body.decode("utf-8", errors="replace"),
                            )
                        except Exception:
                            self._key_manager.report_http_error(
                                active_key, response.status_code
                            )
                        logger.warning(
                            "{}_KEY_ROTATE: key={} status={}, switching key",
                            tag,
                            mask_key(active_key),
                            response.status_code,
                        )
                        if not response.is_closed:
                            await _maybe_await_aclose(response)
                        response = None
                        continue

                    # 200: stream chunks
                    chunk_count = 0
                    chunk_bytes = 0

                    async for chunk in self._iter_stream_chunks(
                        response,
                        state=state,
                        thinking_enabled=thinking_enabled,
                    ):
                        chunk_count += 1
                        chunk_bytes += len(chunk.encode("utf-8", errors="replace"))
                        sent_any_event = True
                        emitted_tracker.feed(chunk)
                        yield chunk

                    self._key_manager.report_success(active_key)

                    trace_event(
                        stage="provider",
                        event="provider.response.completed",
                        source="provider",
                        provider=self._provider_name,
                        gateway_model=request.model,
                        sse_chunks_out=chunk_count,
                        sse_bytes_out=chunk_bytes,
                    )
                    return

                except AllKeysExhaustedError:
                    error_message = self._format_error_message(
                        "All OpenRouter keys exhausted", request_id
                    )
                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=self._provider_name,
                        error_message=error_message,
                        exc_type="AllKeysExhaustedError",
                        mid_stream=sent_any_event,
                    )
                    if sent_any_event:
                        for event in emitted_tracker.iter_close_unclosed_blocks():
                            yield event
                        for event in emitted_tracker.iter_midstream_error_tail(
                            error_message,
                            request=request,
                            input_tokens=input_tokens,
                            log_raw_sse_events=self._config.log_raw_sse_events,
                        ):
                            yield event
                    else:
                        for event in self._emit_error_events(
                            request=request,
                            input_tokens=input_tokens,
                            error_message=error_message,
                            sent_any_event=False,
                        ):
                            yield event
                    return

                except httpx.HTTPStatusError as error:
                    error_message = self._get_error_message(error, request_id)

                    if response is not None and not response.is_closed:
                        await _maybe_await_aclose(response)

                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=self._provider_name,
                        error_message=error_message,
                        exc_type=type(error).__name__,
                        mid_stream=sent_any_event,
                    )
                    if sent_any_event:
                        for event in emitted_tracker.iter_close_unclosed_blocks():
                            yield event
                        for event in emitted_tracker.iter_midstream_error_tail(
                            error_message,
                            request=request,
                            input_tokens=input_tokens,
                            log_raw_sse_events=self._config.log_raw_sse_events,
                        ):
                            yield event
                    else:
                        for event in self._emit_error_events(
                            request=request,
                            input_tokens=input_tokens,
                            error_message=error_message,
                            sent_any_event=False,
                        ):
                            yield event
                    return

                except Exception as error:
                    if not isinstance(error, httpx.HTTPStatusError):
                        self._log_stream_transport_error(
                            tag, req_tag, error, request_id=request_id
                        )
                    error_message = self._get_error_message(error, request_id)

                    if response is not None and not response.is_closed:
                        await _maybe_await_aclose(response)

                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=self._provider_name,
                        error_message=error_message,
                        exc_type=type(error).__name__,
                        mid_stream=sent_any_event,
                    )
                    if sent_any_event:
                        for event in emitted_tracker.iter_close_unclosed_blocks():
                            yield event
                        for event in emitted_tracker.iter_midstream_error_tail(
                            error_message,
                            request=request,
                            input_tokens=input_tokens,
                            log_raw_sse_events=self._config.log_raw_sse_events,
                        ):
                            yield event
                    else:
                        for event in self._emit_error_events(
                            request=request,
                            input_tokens=input_tokens,
                            error_message=error_message,
                            sent_any_event=False,
                        ):
                            yield event
                    return

                finally:
                    if response is not None and not response.is_closed:
                        await _maybe_await_aclose(response)

        # All attempts exhausted
        error_message = self._format_error_message(
            "All OpenRouter key rotation attempts exhausted", request_id
        )
        for event in self._emit_error_events(
            request=request,
            input_tokens=input_tokens,
            error_message=error_message,
            sent_any_event=False,
        ):
            yield event
