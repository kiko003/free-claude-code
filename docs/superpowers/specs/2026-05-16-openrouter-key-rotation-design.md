# OpenRouter Multi-Key Rotation Design

Integrate multi-API-key rotation into the OpenRouter provider, adapted from
[openrouter-proxy-injector](https://github.com/serjs/openrouter-proxy-injector).

## 1. Configuration

**Current**: `OPENROUTER_API_KEY="sk-or-v1-abc123"` — single key string.

**Proposed**: Comma-separated keys with optional per-key daily limits:

```
OPENROUTER_API_KEY="sk-or-v1-abc:50,sk-or-v1-def:1000,sk-or-v1-ghi"
```

- `key:limit` — key with explicit daily request limit (50 for free, 1000 for paid)
- `key` (no limit, multiple keys present) — defaults to 50 (OpenRouter free-tier default)
- Single key: `OPENROUTER_API_KEY="sk-or-v1-abc"` — backward compatible, key manager not created, provider behaves identically to current code (no quota tracking, no rotation)

Parsing is a pure function in `providers/open_router/key_manager.py`. The
`config/settings.py` field stays `str` — comma-separated parsing happens at the
provider level.

New env var: `OPENROUTER_TIMEZONE` (default `"UTC"`) — timezone for daily quota
reset calculation.

## 2. OpenRouterKeyManager

File: `providers/open_router/key_manager.py`

### Per-key data model

```python
@dataclass
class KeyState:
    api_key: str
    daily_limit: int          # 50 or 1000
    success_count: int        # requests completed today
    last_used_at: float       # monotonic timestamp
    blocked_until: float      # monotonic; 0 = not blocked
    block_reason: str         # "rpm" | "upstream" | "daily_limit" | ""
```

### Public interface

| Method | Purpose |
|--------|---------|
| `get_available_key() -> KeyState \| None` | Quota-aware selection: filter blocked/exhausted, enforce 3s RPM minimum, pick highest remaining quota % |
| `report_success(key: str)` | Increment success_count, update last_used_at |
| `report_rate_limit(key: str, error_body: str)` | Parse 429/402 body, apply graduated blocking: ~10s "rate-limited" (OpenRouter RPM), ~30s "upstream provider limit", until-midnight "daily limit exceeded" |
| `report_http_error(key: str, status: int)` | Short block (~10s) on 429 without parseable body |
| `get_key_statuses() -> list[KeyStatusInfo]` | For `/key-status` endpoint — masked key, remaining quota, blocked state |
| `reset_daily_counters_if_needed()` | Reset all counters if UTC midnight has passed (called at start of `get_available_key`) |

### Key selection algorithm

1. Call `reset_daily_counters_if_needed()`
2. Filter out keys where `blocked_until > now` or `success_count >= daily_limit`
3. Filter out keys where `now - last_used_at < 3.0` (20 RPM = 1 req/3s)
4. Sort remaining by `remaining_quota_pct = (daily_limit - success_count) / daily_limit` descending
5. Return top key, or `None` if no keys available

### Dependencies

No new pip packages. Uses `datetime` + `zoneinfo` (stdlib) for timezone-aware
midnight resets instead of the injector's `pendulum` dependency.

## 3. OpenRouter Provider Changes

File: `providers/open_router/client.py`

### __init__ changes

- If `config.api_key` contains no comma (single key), skip key manager creation entirely — set `self._key_manager = None` and use `self._api_key` as before (backward compat, no quota tracking)
- If `config.api_key` contains comma(s), parse into `KeyState` entries via `parse_keys_config()` and create `self._key_manager = OpenRouterKeyManager(keys, timezone=...)`
- When key manager is active, `self._api_key` is unused — active key comes from key manager per-request

### _request_headers / _model_list_headers

Read from `self._current_key` (short-lived attr set just before the HTTP send).
Overrides `_send_stream_request` and `_send_model_list_request` to:

1. Select key via `key_manager.get_available_key()`
2. Set `self._current_key = key_state`
3. Build request with headers from `_request_headers()` (which reads `_current_key`)
4. Send via `self._client.send(request, stream=True)`

If no key available, raise `AllKeysExhaustedError`.

### stream_response override

Override the base class `stream_response()` with a key-switching retry loop:

```
max_attempts = len(keys) * 2
for attempt in range(max_attempts):
    key = key_manager.get_available_key()
    if key is None:
        → yield 429 error events "All OpenRouter keys exhausted"
        → return
    response = send_request(key)
    if 200 → stream chunks, report_success(key), done
    if 429 → report_rate_limit(key, body), continue (next key)
    if 5xx → execute_with_retry (same key, backoff via GlobalRateLimiter)
    if 4xx other → raise (auth/error, not retryable)
```

Mid-stream errors: handled by existing `_emit_error_events` in base class.
No key switching mid-stream.

### Model list requests

Use the first non-blocked key. No rotation needed for read-only metadata queries.

## 4. Key-Status API Endpoint

Route: `GET /v1/openrouter/key-status`

Response shape:

```json
{
  "keys": [
    {
      "api_key": "sk-or-v1-...384c",
      "daily_limit": 50,
      "success_count": 12,
      "remaining": 38,
      "remaining_pct": 76.0,
      "blocked": false,
      "blocked_until": null,
      "block_reason": null,
      "last_used_at": "2026-05-16T14:32:01Z"
    }
  ],
  "next_reset_at": "2026-05-17T00:00:00Z"
}
```

- API keys masked: first 8 + last 4 characters (e.g. `sk-or-v1-...384c`)
- Protected by existing `require_api_key` auth dependency
- Returns empty list if no OpenRouter provider configured
- No changes to existing routes

## 5. Test Strategy

### Unit tests — `tests/test_openrouter_key_manager.py`

| Test case | Verified behavior |
|-----------|-------------------|
| Parse single key | One KeyState, limit=50 |
| Parse multiple keys with limits | Correct KeyStates with declared limits |
| Parse empty/malformed | Empty list or clear error |
| Quota-aware selection | Key with 80% remaining picked over 50% |
| RPM throttle | Key used <3s ago skipped |
| All keys blocked → None | No exception, returns None |
| All keys exhausted → None | success_count >= daily_limit for all |
| report_success | Counter incremented, last_used_at updated |
| report_rate_limit (RPM) | ~10s block |
| report_rate_limit (upstream) | ~30s block |
| report_rate_limit (daily) | Blocked until midnight UTC |
| Daily reset | Counters clear at midnight |
| Key masking | `mask_key("sk-or-v1-abc123xyz")` → `"sk-or-v1-...xyz"` |

### Unit tests — `tests/test_openrouter_provider.py`

| Test case | Verified behavior |
|-----------|-------------------|
| Single key backward compat | No rotation, behaves as before |
| 429 triggers key switch | Second key used after first 429 |
| All keys 429 → 429 response | Client receives 429 "all keys exhausted" |
| 200 reports success | `report_success` called on successful stream |
| 5xx uses backoff | Server errors retry via GlobalRateLimiter, no key switch |

### Smoke tests

Existing `FCC_SMOKE_MODEL_OPEN_ROUTER` passes with single key (backward compat).
Multi-key smoke opt-in via `FCC_SMOKE_OPENROUTER_KEYS_ADDITIONAL` env var.

## 6. File Change Summary

| File | Action | Change |
|------|--------|--------|
| `providers/open_router/key_manager.py` | Create | `OpenRouterKeyManager`, `KeyState`, `parse_keys_config`, `mask_key`, `AllKeysExhaustedError` |
| `providers/open_router/client.py` | Modify | Override `__init__`, `_send_stream_request`, `stream_response`; update header methods for per-request key |
| `providers/open_router/__init__.py` | Modify | Export `OpenRouterKeyManager` |
| `providers/open_router/request.py` | No change | Request body building is key-independent |
| `providers/registry.py` | Modify | Pass `settings` to `_create_open_router` factory |
| `config/settings.py` | Modify | Add `open_router_timezone: str = "UTC"` |
| `.env.example` | Modify | Multi-key format comment, add `OPENROUTER_TIMEZONE` |
| `api/routes.py` | Modify | Add `GET /v1/openrouter/key-status` route |
| `providers/exceptions.py` | Modify | Add `AllKeysExhaustedError` |
| `tests/test_openrouter_key_manager.py` | Create | Key manager unit tests |
| `tests/test_openrouter_provider.py` | Create | Provider key-switch unit tests |

**Unchanged** (zero blast radius): `providers/base.py`, `providers/anthropic_messages.py`,
`providers/rate_limit.py`, `core/rate_limit.py`.
