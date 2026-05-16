"""Tests for OpenRouter provider key rotation: init, single-key compat, key switching."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest

from providers.base import ProviderConfig
from providers.open_router import OpenRouterProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "stepfun/step-3.5-flash:free"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.metadata = None
        self.extra_body = {}
        self.thinking = type("Thinking", (), {"enabled": True})()
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, *, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self.is_closed = False
        self.headers: dict[str, str] = {}

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._text.encode()

    def raise_for_status(self):

        response = httpx.Response(
            self.status_code,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
            text=self._text,
        )
        response.raise_for_status()

    async def aclose(self):
        self.is_closed = True


class TestProviderInit:
    def test_single_key_no_key_manager(self):
        """Single key: no key manager created, backward compatible."""
        config = ProviderConfig(
            api_key="sk-or-v1-single",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config)
        assert provider._key_manager is None
        assert provider._api_key == "sk-or-v1-single"

    def test_multi_key_creates_key_manager(self):
        """Comma-separated keys: key manager is created."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
        assert provider._key_manager is not None
        assert len(provider._key_manager._keys) == 2

    def test_multi_key_no_default_api_key(self):
        """With key manager active, _api_key is not used for requests."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
        # _api_key still set from config for backward compat attribute access
        assert provider._api_key == "sk-or-v1-a:50,sk-or-v1-b:1000"


class TestPerRequestKeyHeaders:
    def test_single_key_uses_config_api_key(self):
        """Single key: headers use config.api_key as before."""
        config = ProviderConfig(
            api_key="sk-or-v1-single",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config)
        headers = provider._request_headers()
        assert headers["Authorization"] == "Bearer sk-or-v1-single"

    def test_multi_key_uses_current_key_from_manager(self):
        """Multi key: headers use key from _current_key when set."""
        from providers.open_router.key_manager import KeyState

        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
            provider._current_key = KeyState(api_key="sk-or-v1-b", daily_limit=1000)
        headers = provider._request_headers()
        assert headers["Authorization"] == "Bearer sk-or-v1-b"

    def test_multi_key_no_current_key_uses_first_key(self):
        """Multi key with no _current_key set: falls back to first key."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
        headers = provider._request_headers()
        assert headers["Authorization"] == "Bearer sk-or-v1-a"

    def test_model_list_headers_single_key(self):
        """Single key: model list headers use config.api_key."""
        config = ProviderConfig(
            api_key="sk-or-v1-single",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config)
        headers = provider._model_list_headers()
        assert headers["Authorization"] == "Bearer sk-or-v1-single"

    def test_model_list_headers_multi_key_uses_current_key(self):
        """Multi key: model list headers use _active_key()."""
        from providers.open_router.key_manager import KeyState

        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
            provider._current_key = KeyState(api_key="sk-or-v1-b", daily_limit=1000)
        headers = provider._model_list_headers()
        assert headers["Authorization"] == "Bearer sk-or-v1-b"


class TestSendStreamRequestWithKeyManager:
    def test_send_stream_selects_key_and_sets_current_key(self):
        """When key manager is active, _send_stream_request selects a key before sending."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")

            # Verify key manager selects a key and sets _current_key
            selected = (
                provider._key_manager.get_available_key()
                if provider._key_manager
                else None
            )
            assert selected is not None
            provider._current_key = selected

            # Headers must now use the selected key
            headers = provider._request_headers()
            assert headers["Authorization"] == f"Bearer {selected.api_key}"

    @pytest.mark.asyncio
    async def test_send_stream_all_keys_exhausted_raises(self):
        """When all keys are exhausted/blocked, raise AllKeysExhaustedError."""
        from providers.open_router.key_manager import AllKeysExhaustedError

        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:50",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
            assert provider._key_manager is not None

            # Exhaust all keys
            for ks in provider._key_manager._keys:
                ks.success_count = 50

            body = provider._build_request_body(MockRequest())
            with pytest.raises(AllKeysExhaustedError):
                await provider._send_stream_request(body)


class TestSendModelListRequestWithKeyManager:
    def test_model_list_uses_first_available_key(self):
        """Model list requests use the first non-blocked key."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
            assert provider._key_manager is not None

            import time as _time

            provider._key_manager._keys[0].blocked_until = _time.monotonic() + 60

            # The second key should be available and used for headers
            selected = provider._key_manager.get_available_key()
            assert selected is not None
            assert selected.api_key == "sk-or-v1-b"

            provider._current_key = selected
            headers = provider._model_list_headers()
            assert headers["Authorization"] == "Bearer sk-or-v1-b"


@asynccontextmanager
async def _noop_concurrency_slot():
    """Replacement for GlobalRateLimiter.concurrency_slot() that does nothing."""
    yield


async def _passthrough_execute_with_retry(fn, *args, **kwargs):
    """Replacement for GlobalRateLimiter.execute_with_retry() — call fn directly."""
    return await fn(*args, **kwargs)


class TestStreamResponseKeySwitch:
    """Tests for the stream_response override with key-switching retry logic."""

    def _make_provider(self, keys_str):
        """Create a multi-key OpenRouterProvider with mocked httpx."""
        config = ProviderConfig(
            api_key=keys_str,
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
        return provider

    @pytest.mark.asyncio
    async def test_single_key_delegates_to_super(self):
        """Single-key provider (no key manager) delegates to base class stream_response."""
        config = ProviderConfig(
            api_key="sk-or-v1-single",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config)

        assert provider._key_manager is None

        mock_chunk = 'data: {"type":"message_stop"}\n\n'

        async def fake_base_stream_response(*args, **kwargs):
            async for item in aiter([mock_chunk]):
                yield item

        with patch.object(
            type(provider).__bases__[0],
            "stream_response",
            fake_base_stream_response,
        ):
            chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert chunks == [mock_chunk]

    @pytest.mark.asyncio
    async def test_multi_key_429_triggers_key_rotation(self):
        """When a 429 is received, the key is reported and the next key is tried."""
        provider = self._make_provider("sk-or-v1-a:50,sk-or-v1-b:50")

        call_count = 0

        async def fake_send_stream_request(body, *, select_key=True):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeResponse(status_code=429, text='{"error":{"code":429}}')
            return FakeResponse(
                status_code=200,
                lines=['data: {"type":"message_stop"}'],
            )

        async def fake_iter_stream_chunks(response, *, state, thinking_enabled):
            yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
            yield 'data: {"type":"message_stop"}\n\n'

        with (
            patch.object(
                provider, "_send_stream_request", side_effect=fake_send_stream_request
            ),
            patch.object(
                provider, "_iter_stream_chunks", side_effect=fake_iter_stream_chunks
            ),
            patch.object(
                provider._global_rate_limiter,
                "concurrency_slot",
                side_effect=_noop_concurrency_slot,
            ),
            patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                side_effect=_passthrough_execute_with_retry,
            ),
        ):
            chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert len(chunks) == 2
        assert call_count == 2
        # First key should have been rate-limited
        assert provider._key_manager._keys[0].blocked_until > 0
        assert provider._key_manager._keys[0].block_reason == "rpm"
        # Second key should have a success
        assert provider._key_manager._keys[1].success_count == 1

    @pytest.mark.asyncio
    async def test_multi_key_402_triggers_key_rotation(self):
        """When a 402 is received, the key is reported and the next key is tried."""
        provider = self._make_provider("sk-or-v1-a:50,sk-or-v1-b:50")

        call_count = 0

        async def fake_send_stream_request(body, *, select_key=True):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeResponse(status_code=402, text='{"error":{"code":402}}')
            return FakeResponse(
                status_code=200, lines=['data: {"type":"message_stop"}']
            )

        async def fake_iter_stream_chunks(response, *, state, thinking_enabled):
            yield 'data: {"type":"message_stop"}\n\n'

        with (
            patch.object(
                provider, "_send_stream_request", side_effect=fake_send_stream_request
            ),
            patch.object(
                provider, "_iter_stream_chunks", side_effect=fake_iter_stream_chunks
            ),
            patch.object(
                provider._global_rate_limiter,
                "concurrency_slot",
                side_effect=_noop_concurrency_slot,
            ),
            patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                side_effect=_passthrough_execute_with_retry,
            ),
        ):
            chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert len(chunks) == 1
        assert call_count == 2
        assert provider._key_manager._keys[0].block_reason == "upstream"
        assert provider._key_manager._keys[1].success_count == 1

    @pytest.mark.asyncio
    async def test_multi_key_200_streams_and_reports_success(self):
        """On a 200 response, chunks are yielded and report_success is called."""
        provider = self._make_provider("sk-or-v1-a:50,sk-or-v1-b:50")

        async def fake_send_stream_request(body, *, select_key=True):
            return FakeResponse(
                status_code=200,
                lines=[
                    'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n',
                    'data: {"type":"message_stop"}\n\n',
                ],
            )

        async def fake_iter_stream_chunks(response, *, state, thinking_enabled):
            yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n'
            yield 'data: {"type":"message_stop"}\n\n'

        with (
            patch.object(
                provider, "_send_stream_request", side_effect=fake_send_stream_request
            ),
            patch.object(
                provider, "_iter_stream_chunks", side_effect=fake_iter_stream_chunks
            ),
            patch.object(
                provider._global_rate_limiter,
                "concurrency_slot",
                side_effect=_noop_concurrency_slot,
            ),
            patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                side_effect=_passthrough_execute_with_retry,
            ),
        ):
            chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert len(chunks) == 2
        assert provider._key_manager._keys[0].success_count == 1

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_at_start_emits_error_events(self):
        """When all keys are already exhausted at the start, error events are emitted."""
        provider = self._make_provider("sk-or-v1-a:50,sk-or-v1-b:50")
        for ks in provider._key_manager._keys:
            ks.success_count = 50

        chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert len(chunks) > 0
        combined = "".join(chunks)
        assert "All OpenRouter keys exhausted" in combined

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_after_rotation_emits_error_events(self):
        """When all keys become exhausted during rotation, error events are emitted."""
        provider = self._make_provider("sk-or-v1-a:50,sk-or-v1-b:50")

        async def fake_send_stream_request(body, *, select_key=True):
            return FakeResponse(status_code=429, text='{"error":{"code":429}}')

        with (
            patch.object(
                provider, "_send_stream_request", side_effect=fake_send_stream_request
            ),
            patch.object(
                provider._global_rate_limiter,
                "concurrency_slot",
                side_effect=_noop_concurrency_slot,
            ),
            patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                side_effect=_passthrough_execute_with_retry,
            ),
        ):
            chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert len(chunks) > 0
        # All keys should be rate-limited
        for ks in provider._key_manager._keys:
            assert ks.blocked_until > 0

    @pytest.mark.asyncio
    async def test_http_status_error_emits_error_events(self):
        """A non-429/402 HTTP error (e.g. 401) emits error events and stops."""
        provider = self._make_provider("sk-or-v1-a:50")

        async def fake_send_stream_request(body, *, select_key=True):
            return FakeResponse(status_code=401, text='{"error":{"code":401}}')

        with (
            patch.object(
                provider, "_send_stream_request", side_effect=fake_send_stream_request
            ),
            patch.object(
                provider._global_rate_limiter,
                "concurrency_slot",
                side_effect=_noop_concurrency_slot,
            ),
            patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                side_effect=_passthrough_execute_with_retry,
            ),
        ):
            chunks = await collect_chunks(provider.stream_response(MockRequest()))

        assert len(chunks) > 0
        combined = "".join(chunks)
        assert "message_start" in combined


async def aiter(items):
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


async def collect_chunks(async_iter):
    """Collect all chunks from an async iterator into a list."""
    return [chunk async for chunk in async_iter]


class TestKeyStatusEndpoint:
    def test_key_status_handler_no_provider(self):
        """If provider is None, returns empty keys list."""
        from api.routes import get_openrouter_key_status

        result = get_openrouter_key_status(provider=None)
        assert result == {"keys": [], "next_reset_at": None}

    def test_key_status_handler_single_key(self):
        """Single-key provider: returns empty keys list (no key manager)."""
        config = ProviderConfig(
            api_key="sk-or-v1-single",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config)

        from api.routes import get_openrouter_key_status

        result = get_openrouter_key_status(provider=provider)
        assert result["keys"] == []

    def test_key_status_handler_multi_key(self):
        """Multi-key provider: returns masked key statuses."""
        config = ProviderConfig(
            api_key="sk-or-v1-abc123xyz:50,sk-or-v1-def456uvw:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")

        from api.routes import get_openrouter_key_status

        result = get_openrouter_key_status(provider=provider)
        assert len(result["keys"]) == 2
        assert result["keys"][0]["daily_limit"] == 50
        assert result["keys"][1]["daily_limit"] == 1000
        assert "..." in result["keys"][0]["api_key"]
        assert "next_reset_at" in result
