# OpenRouter Multi-Key Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate multi-API-key rotation into the OpenRouter provider so users can specify comma-separated keys with per-key daily limits, automatic quota-aware selection, RPM throttling, and graduated 429/402 handling.

**Architecture:** A standalone `OpenRouterKeyManager` class inside `providers/open_router/key_manager.py` tracks per-key state (quota, RPM, blocks). The `OpenRouterProvider` optionally creates this manager when it detects a comma in `config.api_key`. When active, the provider overrides `_send_stream_request` for per-request key selection and `stream_response` for a key-switching retry loop. A new `GET /v1/openrouter/key-status` endpoint exposes key statuses. Single-key configs are fully backward compatible — the key manager is never created.

**Tech Stack:** Python 3.14, httpx, FastAPI, pytest, stdlib `datetime`/`zoneinfo` (no new pip deps)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `providers/open_router/key_manager.py` | Create | `KeyState`, `OpenRouterKeyManager`, `parse_keys_config`, `mask_key`, `AllKeysExhaustedError` |
| `providers/open_router/client.py` | Modify | Override `_send_stream_request`, `stream_response`, update header methods for per-request key |
| `providers/open_router/__init__.py` | Modify | Export `OpenRouterKeyManager` |
| `providers/registry.py` | Modify | Pass `settings` to `_create_open_router` factory |
| `config/settings.py` | Modify | Add `open_router_timezone: str = "UTC"` |
| `.env.example` | Modify | Multi-key format comment, add `OPENROUTER_TIMEZONE` |
| `api/routes.py` | Modify | Add `GET /v1/openrouter/key-status` route |
| `tests/providers/test_openrouter_key_manager.py` | Create | Key manager unit tests |
| `tests/providers/test_openrouter_provider_key_rotation.py` | Create | Provider key-switch unit tests |

---

### Task 1: Add `AllKeysExhaustedError` to exceptions

**Files:**
- Modify: `providers/exceptions.py`
- Test: `tests/providers/test_openrouter_key_manager.py` (will test import)

- [ ] **Step 1: Write the failing test**

```python
# tests/providers/test_openrouter_key_manager.py
"""Tests for OpenRouter key manager: parsing, selection, blocking, masking."""

import pytest


def test_all_keys_exhausted_error_importable():
    from providers.open_router.key_manager import AllKeysExhaustedError

    err = AllKeysExhaustedError("all keys exhausted")
    assert str(err) == "all keys exhausted"
    assert err.status_code == 429
    assert err.error_type == "rate_limit_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::test_all_keys_exhausted_error_importable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers.open_router.key_manager'`

- [ ] **Step 3: Write minimal implementation**

Create `providers/open_router/key_manager.py`:

```python
"""OpenRouter multi-key rotation: parsing, state tracking, and selection."""

from __future__ import annotations

from providers.exceptions import RateLimitError


class AllKeysExhaustedError(RateLimitError):
    """All OpenRouter API keys are exhausted (blocked or daily quota reached)."""

    def __init__(self, message: str = "All OpenRouter keys exhausted"):
        super().__init__(message)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::test_all_keys_exhausted_error_importable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/open_router/key_manager.py tests/providers/test_openrouter_key_manager.py
git commit -m "feat(openrouter): add AllKeysExhaustedError for key rotation"
```

---

### Task 2: Implement `parse_keys_config` and `mask_key`

**Files:**
- Modify: `providers/open_router/key_manager.py`
- Test: `tests/providers/test_openrouter_key_manager.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_key_manager.py`:

```python
from dataclasses import fields

from providers.open_router.key_manager import KeyState, mask_key, parse_keys_config


class TestParseKeysConfig:
    def test_single_key_no_limit(self):
        keys = parse_keys_config("sk-or-v1-abc123")
        assert len(keys) == 1
        assert keys[0].api_key == "sk-or-v1-abc123"
        assert keys[0].daily_limit == 50

    def test_single_key_with_limit(self):
        keys = parse_keys_config("sk-or-v1-abc:1000")
        assert len(keys) == 1
        assert keys[0].api_key == "sk-or-v1-abc"
        assert keys[0].daily_limit == 1000

    def test_multiple_keys_mixed_limits(self):
        keys = parse_keys_config("sk-or-v1-abc:50,sk-or-v1-def:1000,sk-or-v1-ghi")
        assert len(keys) == 3
        assert keys[0].daily_limit == 50
        assert keys[1].daily_limit == 1000
        assert keys[2].daily_limit == 50  # default when omitted with multiple keys

    def test_empty_string(self):
        keys = parse_keys_config("")
        assert keys == []

    def test_whitespace_trimming(self):
        keys = parse_keys_config(" sk-or-v1-abc : 50 , sk-or-v1-def : 1000 ")
        assert len(keys) == 2
        assert keys[0].api_key == "sk-or-v1-abc"
        assert keys[0].daily_limit == 50
        assert keys[1].api_key == "sk-or-v1-def"
        assert keys[1].daily_limit == 1000

    def test_invalid_limit_falls_back_to_50(self):
        keys = parse_keys_config("sk-or-v1-abc:invalid")
        assert len(keys) == 1
        assert keys[0].daily_limit == 50

    def test_empty_key_entry_skipped(self):
        keys = parse_keys_config("sk-or-v1-abc:50,,sk-or-v1-def")
        assert len(keys) == 2

    def test_key_state_fields(self):
        key = KeyState(api_key="k", daily_limit=50)
        field_names = {f.name for f in fields(KeyState)}
        assert field_names == {
            "api_key", "daily_limit", "success_count",
            "last_used_at", "blocked_until", "block_reason",
        }
        assert key.success_count == 0
        assert key.last_used_at == 0.0
        assert key.blocked_until == 0.0
        assert key.block_reason == ""


class TestMaskKey:
    def test_long_key(self):
        assert mask_key("sk-or-v1-abc123xyz") == "sk-or-v1...3xyz"

    def test_short_key(self):
        assert mask_key("short") == "short"

    def test_12_char_key(self):
        assert mask_key("sk-or-v1-abc") == "sk-or-v1...-abc"

    def test_empty_key(self):
        assert mask_key("") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::TestParseKeysConfig tests/providers/test_openrouter_key_manager.py::TestMaskKey -v`
Expected: FAIL with `ImportError` (KeyState, parse_keys_config, mask_key not defined)

- [ ] **Step 3: Write minimal implementation**

Add to `providers/open_router/key_manager.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from providers.exceptions import RateLimitError


class AllKeysExhaustedError(RateLimitError):
    """All OpenRouter API keys are exhausted (blocked or daily quota reached)."""

    def __init__(self, message: str = "All OpenRouter keys exhausted"):
        super().__init__(message)


@dataclass
class KeyState:
    api_key: str
    daily_limit: int = 50
    success_count: int = 0
    last_used_at: float = 0.0
    blocked_until: float = 0.0
    block_reason: str = ""


def parse_keys_config(raw: str) -> list[KeyState]:
    """Parse comma-separated keys with optional ``:limit`` suffix.

    Format: ``key1:50,key2:1000,key3``
    - ``key:limit`` — explicit daily request limit
    - ``key`` alone — defaults to 50
    """
    if not raw.strip():
        return []
    keys: list[KeyState] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            key_part, limit_part = entry.rsplit(":", 1)
            key_part = key_part.strip()
            limit_part = limit_part.strip()
        else:
            key_part = entry.strip()
            limit_part = ""
        if not key_part:
            continue
        daily_limit = 50
        if limit_part:
            try:
                daily_limit = int(limit_part)
            except ValueError:
                daily_limit = 50
        keys.append(KeyState(api_key=key_part, daily_limit=daily_limit))
    return keys


def mask_key(key: str) -> str:
    """Mask an API key showing first 8 and last 4 characters.

    Returns the full key unchanged if shorter than 12 characters.
    """
    if len(key) < 12:
        return key
    return f"{key[:8]}...{key[-4:]}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add providers/open_router/key_manager.py tests/providers/test_openrouter_key_manager.py
git commit -m "feat(openrouter): add parse_keys_config and mask_key for key rotation"
```

---

### Task 3: Implement `OpenRouterKeyManager` core — selection, success, daily reset

**Files:**
- Modify: `providers/open_router/key_manager.py`
- Test: `tests/providers/test_openrouter_key_manager.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_key_manager.py`:

```python
import time
from datetime import datetime, timezone
from unittest.mock import patch

from providers.open_router.key_manager import OpenRouterKeyManager


def _make_manager(keys_str: str, *, timezone_name: str = "UTC") -> OpenRouterKeyManager:
    keys = parse_keys_config(keys_str)
    return OpenRouterKeyManager(keys, timezone_name=timezone_name)


class TestKeyManagerSelection:
    def test_prefers_key_with_highest_remaining_quota(self):
        mgr = _make_manager("sk-a:100,sk-b:50")
        mgr._keys[0].success_count = 90  # 10% remaining
        mgr._keys[1].success_count = 25  # 50% remaining
        key = mgr.get_available_key()
        assert key is not None
        assert key.api_key == "sk-b"

    def test_skips_blocked_keys(self):
        mgr = _make_manager("sk-a:50,sk-b:50")
        now = time.monotonic()
        mgr._keys[0].blocked_until = now + 60
        mgr._keys[0].block_reason = "rpm"
        key = mgr.get_available_key()
        assert key is not None
        assert key.api_key == "sk-b"

    def test_skips_exhausted_keys(self):
        mgr = _make_manager("sk-a:50,sk-b:50")
        mgr._keys[0].success_count = 50
        key = mgr.get_available_key()
        assert key is not None
        assert key.api_key == "sk-b"

    def test_rpm_throttle_skips_recently_used_key(self):
        mgr = _make_manager("sk-a:50,sk-b:50")
        mgr._keys[0].last_used_at = time.monotonic() - 1.0  # 1s ago
        key = mgr.get_available_key()
        assert key is not None
        assert key.api_key == "sk-b"

    def test_all_keys_blocked_returns_none(self):
        mgr = _make_manager("sk-a:50,sk-b:50")
        now = time.monotonic()
        mgr._keys[0].blocked_until = now + 60
        mgr._keys[1].blocked_until = now + 60
        assert mgr.get_available_key() is None

    def test_all_keys_exhausted_returns_none(self):
        mgr = _make_manager("sk-a:50,sk-b:50")
        mgr._keys[0].success_count = 50
        mgr._keys[1].success_count = 50
        assert mgr.get_available_key() is None

    def test_all_keys_rpm_throttled_returns_none(self):
        mgr = _make_manager("sk-a:50,sk-b:50")
        now = time.monotonic()
        mgr._keys[0].last_used_at = now - 0.5
        mgr._keys[1].last_used_at = now - 0.5
        assert mgr.get_available_key() is None

    def test_empty_keys_returns_none(self):
        mgr = OpenRouterKeyManager([], timezone_name="UTC")
        assert mgr.get_available_key() is None


class TestReportSuccess:
    def test_increments_counter_and_updates_last_used(self):
        mgr = _make_manager("sk-a:50")
        before = time.monotonic()
        mgr.report_success("sk-a")
        assert mgr._keys[0].success_count == 1
        assert mgr._keys[0].last_used_at >= before

    def test_unknown_key_is_noop(self):
        mgr = _make_manager("sk-a:50")
        mgr.report_success("nonexistent")
        assert mgr._keys[0].success_count == 0


class TestDailyReset:
    def test_resets_counters_when_midnight_passed(self):
        mgr = _make_manager("sk-a:50")
        mgr._keys[0].success_count = 50
        # Set last reset to yesterday
        yesterday = datetime(2026, 5, 15, 0, 0, 0, tzinfo=timezone.utc)
        mgr._last_reset_date = yesterday.date()
        # Patch "now" to be after midnight
        with patch("providers.open_router.key_manager.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 16, 1, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            key = mgr.get_available_key()
        assert key is not None
        assert key.success_count == 0

    def test_no_reset_when_same_day(self):
        mgr = _make_manager("sk-a:50")
        mgr._keys[0].success_count = 25
        today = datetime(2026, 5, 16, 0, 0, 0, tzinfo=timezone.utc)
        mgr._last_reset_date = today.date()
        with patch("providers.open_router.key_manager.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            key = mgr.get_available_key()
        assert key is not None
        assert key.success_count == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::TestKeyManagerSelection tests/providers/test_openrouter_key_manager.py::TestReportSuccess tests/providers/test_openrouter_key_manager.py::TestDailyReset -v`
Expected: FAIL with `ImportError: cannot import name 'OpenRouterKeyManager'`

- [ ] **Step 3: Write minimal implementation**

Add to `providers/open_router/key_manager.py`:

```python
from datetime import datetime, timezone as _tz
from zoneinfo import ZoneInfo


class OpenRouterKeyManager:
    """Quota-aware key rotation for OpenRouter multi-key setups."""

    _RPM_MIN_INTERVAL = 3.0  # 20 RPM = 1 req per 3s

    def __init__(
        self,
        keys: list[KeyState],
        *,
        timezone_name: str = "UTC",
    ):
        self._keys = keys
        self._tz = ZoneInfo(timezone_name)
        now = datetime.now(self._tz)
        self._last_reset_date = now.date()

    def get_available_key(self) -> KeyState | None:
        """Select the best available key by remaining quota %.

        Filters out blocked, exhausted, and RPM-throttled keys.
        Resets daily counters if a new day has started.
        """
        self.reset_daily_counters_if_needed()
        now = time.monotonic()
        candidates: list[tuple[KeyState, float]] = []
        for ks in self._keys:
            if ks.blocked_until > now:
                continue
            if ks.success_count >= ks.daily_limit:
                continue
            if now - ks.last_used_at < self._RPM_MIN_INTERVAL:
                continue
            remaining_pct = (ks.daily_limit - ks.success_count) / ks.daily_limit
            candidates.append((ks, remaining_pct))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def report_success(self, key: str) -> None:
        """Increment success_count and update last_used_at for a key."""
        ks = self._find_key(key)
        if ks is None:
            return
        ks.success_count += 1
        ks.last_used_at = time.monotonic()

    def reset_daily_counters_if_needed(self) -> None:
        """Reset all success counters if the timezone-local date has rolled over."""
        now = datetime.now(self._tz)
        if now.date() > self._last_reset_date:
            for ks in self._keys:
                ks.success_count = 0
                if ks.block_reason == "daily_limit":
                    ks.blocked_until = 0.0
                    ks.block_reason = ""
            self._last_reset_date = now.date()

    def _find_key(self, key: str) -> KeyState | None:
        for ks in self._keys:
            if ks.api_key == key:
                return ks
        return None
```

Also add `import time` at the top of the file (next to the other stdlib imports).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add providers/open_router/key_manager.py tests/providers/test_openrouter_key_manager.py
git commit -m "feat(openrouter): add OpenRouterKeyManager with selection, success, daily reset"
```

---

### Task 4: Implement `report_rate_limit` and `report_http_error`

**Files:**
- Modify: `providers/open_router/key_manager.py`
- Test: `tests/providers/test_openrouter_key_manager.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_key_manager.py`:

```python
import json


class TestReportRateLimit:
    def test_rpm_rate_limit_blocks_10s(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}})
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].blocked_until > time.monotonic()
        assert mgr._keys[0].blocked_until < time.monotonic() + 15
        assert mgr._keys[0].block_reason == "rpm"

    def test_upstream_provider_limit_blocks_30s(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps({
            "error": {
                "code": 429,
                "message": "Provider returned error: model is temporarily rate-limited upstream",
            }
        })
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].blocked_until > time.monotonic() + 25
        assert mgr._keys[0].blocked_until < time.monotonic() + 40
        assert mgr._keys[0].block_reason == "upstream"

    def test_daily_limit_blocks_until_midnight(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps({
            "error": {
                "code": 429,
                "message": "Rate limit exceeded: free-models-per-day. Add 10 credits",
            }
        })
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].block_reason == "daily_limit"
        assert mgr._keys[0].success_count == 50

    def test_credits_exhausted_blocks_until_midnight(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps({
            "error": {
                "code": 429,
                "message": "Credits exhausted for free models",
            }
        })
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].block_reason == "daily_limit"
        assert mgr._keys[0].success_count == 50

    def test_402_payment_required_blocks_5m(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps({"error": {"code": 402, "message": "Payment required"}})
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].blocked_until > time.monotonic() + 290
        assert mgr._keys[0].block_reason == "upstream"

    def test_non_json_body_defaults_to_10s(self):
        mgr = _make_manager("sk-a:50")
        mgr.report_rate_limit("sk-a", "not json")
        assert mgr._keys[0].blocked_until > time.monotonic()
        assert mgr._keys[0].blocked_until < time.monotonic() + 15
        assert mgr._keys[0].block_reason == "rpm"

    def test_unknown_key_is_noop(self):
        mgr = _make_manager("sk-a:50")
        mgr.report_rate_limit("nonexistent", "{}")
        assert mgr._keys[0].blocked_until == 0.0


class TestReportHttpError:
    def test_429_without_body_short_block(self):
        mgr = _make_manager("sk-a:50")
        mgr.report_http_error("sk-a", 429)
        assert mgr._keys[0].blocked_until > time.monotonic()
        assert mgr._keys[0].block_reason == "rpm"

    def test_unknown_key_is_noop(self):
        mgr = _make_manager("sk-a:50")
        mgr.report_http_error("nonexistent", 429)
        assert mgr._keys[0].blocked_until == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::TestReportRateLimit tests/providers/test_openrouter_key_manager.py::TestReportHttpError -v`
Expected: FAIL with `AttributeError` (methods not defined)

- [ ] **Step 3: Write minimal implementation**

Add these methods to `OpenRouterKeyManager` in `providers/open_router/key_manager.py`:

```python
    def report_rate_limit(self, key: str, error_body: str) -> None:
        """Parse a 429/402 response body and apply graduated blocking."""
        ks = self._find_key(key)
        if ks is None:
            return
        now = time.monotonic()
        try:
            data = json.loads(error_body)
            error_data = data.get("error", {})
            if not isinstance(error_data, dict):
                error_data = {}
            error_message = error_data.get("message", "")
            error_code = error_data.get("code")
        except (json.JSONDecodeError, AttributeError):
            error_message = ""
            error_code = None

        if str(error_code) == "402":
            ks.blocked_until = now + 300  # 5 minutes
            ks.block_reason = "upstream"
            return

        if str(error_code) == "429":
            if "free-models-per-day" in error_message or "Credits exhausted" in error_message:
                ks.block_reason = "daily_limit"
                ks.blocked_until = 0.0  # cleared by daily reset
                ks.success_count = ks.daily_limit
                return
            if "Provider returned error" in error_message or "upstream" in error_message.lower():
                ks.blocked_until = now + 30
                ks.block_reason = "upstream"
                return

        # Default: OpenRouter RPM limit or unparseable body
        ks.blocked_until = now + 10
        ks.block_reason = "rpm"

    def report_http_error(self, key: str, status: int) -> None:
        """Short block on 429 without parseable body."""
        ks = self._find_key(key)
        if ks is None:
            return
        if status == 429:
            ks.blocked_until = time.monotonic() + 10
            ks.block_reason = "rpm"
```

Also add `import json` at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add providers/open_router/key_manager.py tests/providers/test_openrouter_key_manager.py
git commit -m "feat(openrouter): add report_rate_limit and report_http_error to key manager"
```

---

### Task 5: Implement `get_key_statuses` for the key-status endpoint

**Files:**
- Modify: `providers/open_router/key_manager.py`
- Test: `tests/providers/test_openrouter_key_manager.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_key_manager.py`:

```python
class TestGetKeyStatuses:
    def test_returns_masked_key_info(self):
        mgr = _make_manager("sk-or-v1-abc123xyz:50")
        mgr._keys[0].success_count = 12
        mgr._keys[0].last_used_at = 1000.0
        statuses = mgr.get_key_statuses()
        assert len(statuses) == 1
        s = statuses[0]
        assert s["api_key"] == "sk-or-v1...3xyz"
        assert s["daily_limit"] == 50
        assert s["success_count"] == 12
        assert s["remaining"] == 38
        assert s["remaining_pct"] == 76.0
        assert s["blocked"] is False

    def test_blocked_key_shows_block_info(self):
        mgr = _make_manager("sk-or-v1-abc123xyz:50")
        mgr._keys[0].blocked_until = time.monotonic() + 30
        mgr._keys[0].block_reason = "upstream"
        statuses = mgr.get_key_statuses()
        assert statuses[0]["blocked"] is True
        assert statuses[0]["block_reason"] == "upstream"

    def test_next_reset_at_is_iso_string(self):
        mgr = _make_manager("sk-a:50")
        statuses = mgr.get_key_statuses()
        assert "next_reset_at" in mgr.get_status_summary()
        # Verify ISO format
        from datetime import datetime
        reset_str = mgr.get_status_summary()["next_reset_at"]
        datetime.fromisoformat(reset_str.replace("Z", "+00:00"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::TestGetKeyStatuses -v`
Expected: FAIL with `AttributeError` (method not defined)

- [ ] **Step 3: Write minimal implementation**

Add these methods to `OpenRouterKeyManager` in `providers/open_router/key_manager.py`:

```python
    def get_key_statuses(self) -> list[dict[str, Any]]:
        """Return status info for all keys (for the key-status endpoint)."""
        now = time.monotonic()
        statuses: list[dict[str, Any]] = []
        for ks in self._keys:
            remaining = max(0, ks.daily_limit - ks.success_count)
            remaining_pct = round((remaining / ks.daily_limit) * 100, 1) if ks.daily_limit else 0.0
            is_blocked = ks.blocked_until > now or ks.success_count >= ks.daily_limit
            blocked_until = ks.blocked_until if ks.blocked_until > now else None
            statuses.append({
                "api_key": mask_key(ks.api_key),
                "daily_limit": ks.daily_limit,
                "success_count": ks.success_count,
                "remaining": remaining,
                "remaining_pct": remaining_pct,
                "blocked": is_blocked,
                "blocked_until": blocked_until,
                "block_reason": ks.block_reason if is_blocked else None,
                "last_used_at": ks.last_used_at if ks.last_used_at > 0 else None,
            })
        return statuses

    def get_status_summary(self) -> dict[str, Any]:
        """Return full key-status response including next reset time."""
        from datetime import timedelta

        now = datetime.now(self._tz)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return {
            "keys": self.get_key_statuses(),
            "next_reset_at": next_midnight.isoformat().replace("+00:00", "Z"),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add providers/open_router/key_manager.py tests/providers/test_openrouter_key_manager.py
git commit -m "feat(openrouter): add get_key_statuses for key-status endpoint"
```

---

### Task 6: Add `open_router_timezone` to Settings and update `.env.example`

**Files:**
- Modify: `config/settings.py`
- Modify: `.env.example`

- [ ] **Step 1: Write the failing test**

Add to `tests/providers/test_openrouter_key_manager.py`:

```python
class TestSettingsTimezone:
    def test_open_router_timezone_default(self):
        from config.settings import Settings

        s = Settings(provider_type="open_router", model="open_router/test")
        assert s.open_router_timezone == "UTC"

    def test_open_router_timezone_from_env(self, monkeypatch):
        from config.settings import Settings

        monkeypatch.setenv("OPENROUTER_TIMEZONE", "America/New_York")
        monkeypatch.setattr(
            Settings, "model_config",
            {**Settings.model_config, "env_file": None},
        )
        s = Settings(provider_type="open_router", model="open_router/test")
        assert s.open_router_timezone == "America/New_York"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::TestSettingsTimezone -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'open_router_timezone'`

- [ ] **Step 3: Write minimal implementation**

In `config/settings.py`, add after the `open_router_api_key` field (line 110):

```python
    open_router_timezone: str = Field(default="UTC", validation_alias="OPENROUTER_TIMEZONE")
```

In `.env.example`, update the OpenRouter section:

```
# OpenRouter Config
# Single key: OPENROUTER_API_KEY="sk-or-v1-..."
# Multiple keys with optional daily limits: OPENROUTER_API_KEY="sk-or-v1-abc:50,sk-or-v1-def:1000,sk-or-v1-ghi"
OPENROUTER_API_KEY="sk-or-v1-api-key-1"
# Timezone for daily quota reset (default: UTC)
OPENROUTER_TIMEZONE="UTC"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py::TestSettingsTimezone -v`
Expected: PASS

- [ ] **Step 5: Run linting and type checks**

Run: `uv run ruff format && uv run ruff check && uv run ty check`
Expected: All pass (no new warnings)

- [ ] **Step 6: Commit**

```bash
git add config/settings.py .env.example tests/providers/test_openrouter_key_manager.py
git commit -m "feat(openrouter): add open_router_timezone setting and update .env.example"
```

---

### Task 7: Update `__init__.py` and registry to pass settings to OpenRouterProvider

**Files:**
- Modify: `providers/open_router/__init__.py`
- Modify: `providers/registry.py`
- Modify: `providers/open_router/client.py`
- Test: `tests/providers/test_openrouter_provider_key_rotation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/test_openrouter_provider_key_rotation.py`:

```python
"""Tests for OpenRouter provider key rotation: init, single-key compat, key switching."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
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
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, *, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self.is_closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._text.encode()

    def raise_for_status(self):
        import httpx
        response = httpx.Response(
            self.status_code,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
            text=self._text,
        )
        response.raise_for_status()

    async def aclose(self):
        self.is_closed = True


@pytest.fixture
def mock_settings():
    return Settings(
        provider_type="open_router",
        model="open_router/test",
        open_router_timezone="UTC",
    )


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
        # _api_key still set from config for backward compat attribute access,
        # but headers will come from key manager
        assert provider._api_key == "sk-or-v1-a:50,sk-or-v1-b:1000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestProviderInit -v`
Expected: FAIL — `OpenRouterProvider.__init__()` doesn't accept `timezone` kwarg yet, and `_key_manager` attr doesn't exist

- [ ] **Step 3: Write minimal implementation**

Update `providers/open_router/__init__.py`:

```python
"""OpenRouter provider - Anthropic-compatible native transport."""

from providers.defaults import OPENROUTER_DEFAULT_BASE

from .client import OpenRouterProvider
from .key_manager import OpenRouterKeyManager

__all__ = ["OPENROUTER_DEFAULT_BASE", "OpenRouterProvider", "OpenRouterKeyManager"]
```

Update `providers/open_router/client.py` — modify `__init__`:

```python
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
```

Update `providers/registry.py` — change the `_create_open_router` function:

```python
def _create_open_router(config: ProviderConfig, settings: Settings) -> BaseProvider:
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config, timezone=settings.open_router_timezone)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestProviderInit -v`
Expected: PASS

- [ ] **Step 5: Run existing OpenRouter tests to verify no regression**

Run: `uv run pytest tests/providers/test_open_router.py -v`
Expected: PASS (single-key backward compat — `timezone` defaults to `"UTC"`)

- [ ] **Step 6: Run linting and type checks**

Run: `uv run ruff format && uv run ruff check && uv run ty check`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add providers/open_router/__init__.py providers/open_router/client.py providers/registry.py tests/providers/test_openrouter_provider_key_rotation.py
git commit -m "feat(openrouter): create key manager on multi-key init, pass settings from registry"
```

---

### Task 8: Override `_send_stream_request` and header methods for per-request key selection

**Files:**
- Modify: `providers/open_router/client.py`
- Test: `tests/providers/test_openrouter_provider_key_rotation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_provider_key_rotation.py`:

```python
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
        """Multi key: headers use key from key_manager._current_key when set."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")
        from providers.open_router.key_manager import KeyState

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestPerRequestKeyHeaders -v`
Expected: FAIL — multi-key headers still use `self._api_key` (the raw comma-separated string)

- [ ] **Step 3: Write minimal implementation**

Update `providers/open_router/client.py` — modify `_request_headers` and `_model_list_headers`:

```python
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

    def _active_key(self) -> str:
        """Return the API key to use for the current request."""
        if self._key_manager is not None:
            current = getattr(self, "_current_key", None)
            if current is not None:
                return current.api_key
            return self._key_manager._keys[0].api_key
        return self._api_key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestPerRequestKeyHeaders -v`
Expected: PASS

- [ ] **Step 5: Run existing OpenRouter tests to verify no regression**

Run: `uv run pytest tests/providers/test_open_router.py -v`
Expected: PASS (single-key still uses config.api_key)

- [ ] **Step 6: Commit**

```bash
git add providers/open_router/client.py tests/providers/test_openrouter_provider_key_rotation.py
git commit -m "feat(openrouter): per-request key selection in header methods"
```

---

### Task 9: Override `_send_stream_request` and `_send_model_list_request` for key manager integration

**Files:**
- Modify: `providers/open_router/client.py`
- Test: `tests/providers/test_openrouter_provider_key_rotation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_provider_key_rotation.py`:

```python
import httpx


class TestSendStreamRequestWithKeyManager:
    @pytest.mark.asyncio
    async def test_send_stream_selects_key_and_sets_current_key(self):
        """When key manager is active, _send_stream_request selects a key and builds request with it."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            provider = OpenRouterProvider(config, timezone="UTC")

        fake_response = FakeResponse(status_code=200, lines=["event: message_start", 'data: {"type":"message_start"}', ""])
        mock_send = AsyncMock(return_value=fake_response)
        provider._client.send = mock_send
        provider._client.build_request = MagicMock(
            return_value=httpx.Request("POST", "https://openrouter.ai/api/v1/messages")
        )

        body = provider._build_request_body(MockRequest())
        response = await provider._send_stream_request(body)

        assert provider._current_key is not None
        assert provider._current_key.api_key in ("sk-or-v1-a", "sk-or-v1-b")

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

        # Exhaust all keys
        for ks in provider._key_manager._keys:
            ks.success_count = 50

        body = provider._build_request_body(MockRequest())
        with pytest.raises(AllKeysExhaustedError):
            await provider._send_stream_request(body)


class TestSendModelListRequestWithKeyManager:
    @pytest.mark.asyncio
    async def test_model_list_uses_first_available_key(self):
        """Model list requests use the first non-blocked key."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:1000",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )
        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")

        import time as _time
        provider._key_manager._keys[0].blocked_until = _time.monotonic() + 60

        fake_response = FakeResponse(status_code=200, lines=[])
        mock_get = AsyncMock(return_value=fake_response)
        provider._client.get = mock_get

        await provider._send_model_list_request()

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-or-v1-b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestSendStreamRequestWithKeyManager tests/providers/test_openrouter_provider_key_rotation.py::TestSendModelListRequestWithKeyManager -v`
Expected: FAIL — `_send_stream_request` doesn't select from key manager yet

- [ ] **Step 3: Write minimal implementation**

Add overrides to `providers/open_router/client.py`:

```python
    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming messages response, selecting key from manager if active."""
        if self._key_manager is not None:
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
```

Add `import httpx` at the top of `providers/open_router/client.py` (it's already imported via the parent class, but the type hint needs it).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py -v`
Expected: PASS

- [ ] **Step 5: Run existing OpenRouter tests**

Run: `uv run pytest tests/providers/test_open_router.py -v`
Expected: PASS

- [ ] **Step 6: Run linting and type checks**

Run: `uv run ruff format && uv run ruff check && uv run ty check`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add providers/open_router/client.py tests/providers/test_openrouter_provider_key_rotation.py
git commit -m "feat(openrouter): override _send_stream_request and _send_model_list_request for key selection"
```

---

### Task 10: Override `stream_response` for key-switching retry loop

**Files:**
- Modify: `providers/open_router/client.py`
- Test: `tests/providers/test_openrouter_provider_key_rotation.py`

This is the core retry logic that distinguishes 429 (key switch) from 5xx (backoff retry) from 4xx (fail fast).

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_openrouter_provider_key_rotation.py`:

```python
import json


class TestStreamResponseKeySwitch:
    @pytest.mark.asyncio
    async def test_429_triggers_key_switch(self):
        """First key gets 429 → report_rate_limit → switch to second key → 200."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:50",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )

        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")

        req = MockRequest()
        ok_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        ok_response = FakeResponse(lines=ok_lines)

        rate_limit_response = FakeResponse(
            status_code=429,
            text=json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}}),
        )

        send_calls = {"n": 0}

        async def send_side_effect(*_a, **_kw):
            send_calls["n"] += 1
            if send_calls["n"] == 1:
                return rate_limit_response
            return ok_response

        @asynccontextmanager
        async def _slot():
            yield

        with (
            patch.object(provider._client, "build_request", return_value=MagicMock()),
            patch.object(
                provider._client, "send", new_callable=AsyncMock, side_effect=send_side_effect,
            ),
            patch("providers.anthropic_messages.GlobalRateLimiter") as mock_gl,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            instance = mock_gl.get_scoped_instance.return_value
            instance.concurrency_slot.side_effect = _slot
            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)
            instance.execute_with_retry = AsyncMock(side_effect=_passthrough)

            events = [e async for e in provider.stream_response(req)]

        assert send_calls["n"] == 2
        event_text = "".join(events)
        assert "message_start" in event_text
        # First key should be blocked
        assert provider._key_manager._keys[0].block_reason == "rpm"

    @pytest.mark.asyncio
    async def test_all_keys_429_yields_429_error(self):
        """All keys return 429 → yield error events to client."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:50",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )

        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")

        req = MockRequest()
        rate_limit_response = FakeResponse(
            status_code=429,
            text=json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}}),
        )

        with (
            patch.object(provider._client, "build_request", return_value=MagicMock()),
            patch.object(
                provider._client, "send",
                new_callable=AsyncMock,
                return_value=rate_limit_response,
            ),
            patch("providers.anthropic_messages.GlobalRateLimiter") as mock_gl,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            instance = mock_gl.get_scoped_instance.return_value
            async def _slot_fn():
                yield
            instance.concurrency_slot.side_effect = _slot_fn
            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)
            instance.execute_with_retry = AsyncMock(side_effect=_passthrough)

            events = [e async for e in provider.stream_response(req)]

        event_text = "".join(events)
        assert "exhausted" in event_text.lower() or "429" in event_text

    @pytest.mark.asyncio
    async def test_5xx_uses_backoff_no_key_switch(self):
        """5xx errors use execute_with_retry on same key, no key switch."""
        config = ProviderConfig(
            api_key="sk-or-v1-a:50,sk-or-v1-b:50",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )

        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config, timezone="UTC")

        req = MockRequest()
        ok_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{}}',
            "",
        ]
        ok_response = FakeResponse(lines=ok_lines)

        send_calls = {"n": 0}

        async def send_side_effect(*_a, **_kw):
            send_calls["n"] += 1
            if send_calls["n"] == 1:
                return FakeResponse(status_code=502, text="bad gateway")
            return ok_response

        @asynccontextmanager
        async def _slot():
            yield

        with (
            patch.object(provider._client, "build_request", return_value=MagicMock()),
            patch.object(
                provider._client, "send", new_callable=AsyncMock, side_effect=send_side_effect,
            ),
            patch("providers.anthropic_messages.GlobalRateLimiter") as mock_gl,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            real_limiter = __import__("providers.rate_limit", fromlist=["GlobalRateLimiter"]).GlobalRateLimiter(
                rate_limit=100, rate_window=60, max_concurrency=5,
            )
            instance = mock_gl.get_scoped_instance.return_value
            instance.concurrency_slot.side_effect = _slot
            instance.wait_if_blocked = real_limiter.wait_if_blocked
            instance.set_blocked = real_limiter.set_blocked
            instance.execute_with_retry = real_limiter.execute_with_retry

            events = [e async for e in provider.stream_response(req)]

        # 5xx should retry on the same key without switching
        assert send_calls["n"] == 2

    @pytest.mark.asyncio
    async def test_single_key_no_key_switch_behavior(self):
        """Single key: base class stream_response unchanged, no key rotation."""
        config = ProviderConfig(
            api_key="sk-or-v1-single",
            base_url="https://openrouter.ai/api/v1",
            rate_limit=10,
            rate_window=60,
        )

        with patch("httpx.AsyncClient"):
            provider = OpenRouterProvider(config)

        assert provider._key_manager is None
        req = MockRequest()
        ok_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        ok_response = FakeResponse(lines=ok_lines)

        @asynccontextmanager
        async def _slot():
            yield

        with (
            patch.object(provider._client, "build_request", return_value=MagicMock()),
            patch.object(
                provider._client, "send", new_callable=AsyncMock, return_value=ok_response,
            ),
            patch("providers.anthropic_messages.GlobalRateLimiter") as mock_gl,
        ):
            instance = mock_gl.get_scoped_instance.return_value
            instance.concurrency_slot.side_effect = _slot
            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)
            instance.execute_with_retry = AsyncMock(side_effect=_passthrough)

            events = [e async for e in provider.stream_response(req)]

        event_text = "".join(events)
        assert "message_start" in event_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestStreamResponseKeySwitch -v`
Expected: FAIL — `stream_response` doesn't have key-switching logic yet

- [ ] **Step 3: Write minimal implementation**

Override `stream_response` in `providers/open_router/client.py`:

```python
    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response with key-switching retry loop when key manager is active."""
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

        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        max_attempts = len(self._key_manager._keys) * 2

        for attempt in range(max_attempts):
            key_state = self._key_manager.get_available_key()
            if key_state is None:
                error_msg = "All OpenRouter keys exhausted"
                for event in self._emit_error_events(
                    request=request,
                    input_tokens=input_tokens,
                    error_message=error_msg,
                    sent_any_event=False,
                ):
                    yield event
                return

            self._current_key = key_state

            async with self._global_rate_limiter.concurrency_slot():
                try:

                    async def _send_with_key() -> httpx.Response:
                        resp = await self._send_stream_request(body)
                        if resp.status_code != 200:
                            try:
                                await self._raise_for_status(resp, req_tag="")
                            finally:
                                if not resp.is_closed:
                                    await _maybe_await_aclose(resp)
                        return resp

                    response = await self._global_rate_limiter.execute_with_retry(
                        _send_with_key
                    )

                    # Stream successful response
                    state = self._new_stream_state(request, thinking_enabled=thinking_enabled)
                    sent_any_event = False
                    try:
                        async for chunk in self._iter_stream_chunks(
                            response, state=state, thinking_enabled=thinking_enabled,
                        ):
                            sent_any_event = True
                            yield chunk
                    finally:
                        if response is not None and not response.is_closed:
                            await _maybe_await_aclose(response)

                    self._key_manager.report_success(key_state.api_key)
                    return

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status == 429 or status == 402:
                        error_body_bytes = await exc.response.aread()
                        error_body = error_body_bytes.decode("utf-8", errors="replace")
                        self._key_manager.report_rate_limit(key_state.api_key, error_body)
                        continue  # try next key
                    # 5xx is already handled by execute_with_retry inside _send_with_key
                    # Other 4xx: emit error and return
                    error_message = self._get_error_message(exc, request_id)
                    for event in self._emit_error_events(
                        request=request,
                        input_tokens=input_tokens,
                        error_message=error_message,
                        sent_any_event=False,
                    ):
                        yield event
                    return

                except Exception as error:
                    if isinstance(error, AllKeysExhaustedError):
                        for event in self._emit_error_events(
                            request=request,
                            input_tokens=input_tokens,
                            error_message="All OpenRouter keys exhausted",
                            sent_any_event=False,
                        ):
                            yield event
                        return
                    error_message = self._get_error_message(error, request_id)
                    for event in self._emit_error_events(
                        request=request,
                        input_tokens=input_tokens,
                        error_message=error_message,
                        sent_any_event=False,
                    ):
                        yield event
                    return

        # Exhausted all attempts
        error_msg = "All OpenRouter keys exhausted after retries"
        for event in self._emit_error_events(
            request=request,
            input_tokens=input_tokens,
            error_message=error_msg,
            sent_any_event=False,
        ):
            yield event
```

Add the missing import at the top of `providers/open_router/client.py`:

```python
from collections.abc import AsyncIterator, Iterator
```

Also, we need to import `_maybe_await_aclose` from the parent module:

```python
from providers.anthropic_messages import AnthropicMessagesTransport, StreamChunkMode, _maybe_await_aclose
```

Wait — `_maybe_await_aclose` is a module-level function in `anthropic_messages.py`. We need to import it. Let's check: the parent `stream_response` already uses it. Since we're overriding `stream_response`, we need to use it directly.

In `providers/open_router/client.py`, update the import line:

```python
from providers.anthropic_messages import AnthropicMessagesTransport, StreamChunkMode, _maybe_await_aclose
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestStreamResponseKeySwitch -v`
Expected: PASS

- [ ] **Step 5: Run all existing OpenRouter tests**

Run: `uv run pytest tests/providers/test_open_router.py -v`
Expected: PASS (single-key backward compat)

- [ ] **Step 6: Run linting and type checks**

Run: `uv run ruff format && uv run ruff check && uv run ty check`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add providers/open_router/client.py tests/providers/test_openrouter_provider_key_rotation.py
git commit -m "feat(openrouter): key-switching retry loop in stream_response"
```

---

### Task 11: Add `GET /v1/openrouter/key-status` API endpoint

**Files:**
- Modify: `api/routes.py`
- Test: `tests/providers/test_openrouter_provider_key_rotation.py` (we'll test via FastAPI TestClient)

- [ ] **Step 1: Write the failing test**

Add to `tests/providers/test_openrouter_provider_key_rotation.py`:

```python
from fastapi.testclient import TestClient


class TestKeyStatusEndpoint:
    def test_key_status_returns_empty_when_no_key_manager(self):
        """No key manager (single key) → empty keys list."""
        from api.dependencies import resolve_provider
        from api.main import create_app

        app = create_app()
        # Patch settings to use single key
        with (
            patch.dict("os.environ", {
                "PROVIDER_TYPE": "open_router",
                "MODEL": "open_router/test-model",
                "OPENROUTER_API_KEY": "sk-or-v1-single",
                "ANTHROPIC_AUTH_TOKEN": "",
            }),
        ):
            client = TestClient(app)
            resp = client.get("/v1/openrouter/key-status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["keys"] == []

    def test_key_status_with_multi_key(self):
        """Multi key → returns key statuses with masked keys."""
        from api.main import create_app

        app = create_app()
        with (
            patch.dict("os.environ", {
                "PROVIDER_TYPE": "open_router",
                "MODEL": "open_router/test-model",
                "OPENROUTER_API_KEY": "sk-or-v1-abc123xyz:50,sk-or-v1-def456uvw:1000",
                "ANTHROPIC_AUTH_TOKEN": "",
            }),
        ):
            client = TestClient(app)
            resp = client.get("/v1/openrouter/key-status")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["keys"]) == 2
            assert data["keys"][0]["daily_limit"] == 50
            assert data["keys"][1]["daily_limit"] == 1000
            assert "..." in data["keys"][0]["api_key"]
            assert "next_reset_at" in data
```

Actually, the endpoint test via TestClient requires full app bootstrap which is complex. Let's simplify to a direct unit test of the route handler instead.

Replace the above test with:

```python
class TestKeyStatusEndpoint:
    def test_key_status_handler_no_openrouter_provider(self):
        """If no open_router provider in registry, returns empty keys list."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestKeyStatusEndpoint -v`
Expected: FAIL with `ImportError: cannot import name 'get_openrouter_key_status'`

- [ ] **Step 3: Write minimal implementation**

Add to `api/routes.py`:

```python
@router.get("/v1/openrouter/key-status")
async def key_status(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """Return OpenRouter key rotation status (masked keys, quotas, blocks)."""
    return get_openrouter_key_status(
        provider=_try_get_openrouter_provider(request, settings)
    )


def get_openrouter_key_status(provider: BaseProvider | None) -> dict:
    """Return key-status payload; empty list if no key manager."""
    if provider is None:
        return {"keys": [], "next_reset_at": None}
    key_manager = getattr(provider, "_key_manager", None)
    if key_manager is None:
        return {"keys": [], "next_reset_at": None}
    return key_manager.get_status_summary()


def _try_get_openrouter_provider(request: Request, settings: Settings) -> BaseProvider | None:
    """Attempt to get the OpenRouter provider; return None on any failure."""
    try:
        return resolve_provider("open_router", app=request.app, settings=settings)
    except Exception:
        return None
```

Also ensure `BaseProvider` is imported in `api/routes.py` (it already imports from `providers.registry`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openrouter_provider_key_rotation.py::TestKeyStatusEndpoint -v`
Expected: PASS

- [ ] **Step 5: Run linting and type checks**

Run: `uv run ruff format && uv run ruff check && uv run ty check`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add api/routes.py tests/providers/test_openrouter_provider_key_rotation.py
git commit -m "feat(openrouter): add GET /v1/openrouter/key-status endpoint"
```

---

### Task 12: Run full test suite and CI gate

**Files:**
- No new files

- [ ] **Step 1: Run full lint chain**

Run: `uv run ruff format && uv run ruff check && uv run ty check`
Expected: All pass

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -x -v`
Expected: All pass

- [ ] **Step 3: Run only the new test files**

Run: `uv run pytest tests/providers/test_openrouter_key_manager.py tests/providers/test_openrouter_provider_key_rotation.py -v`
Expected: All pass

- [ ] **Step 4: Commit any lint/format fixes if needed**

```bash
git add -A
git commit -m "chore: lint and format fixes for openrouter key rotation"
```

(Only if there are changes; skip if clean.)

---

## Self-Review

**1. Spec coverage:**

| Spec Section | Tasks | Status |
|-------------|-------|--------|
| §1 Configuration (comma-separated keys, `:limit`, single-key compat) | Task 2, Task 6, Task 7 | Covered |
| §2 OpenRouterKeyManager (KeyState, get_available_key, report_success, report_rate_limit, report_http_error, get_key_statuses, reset_daily_counters_if_needed) | Tasks 3, 4, 5 | Covered |
| §3 Provider changes (__init__, _send_stream_request, stream_response, headers) | Tasks 7, 8, 9, 10 | Covered |
| §4 Key-Status API endpoint | Task 11 | Covered |
| §5 Test strategy | All tasks (TDD) | Covered |
| §6 File change summary | All tasks mapped | Covered |

**2. Placeholder scan:** No TBD/TODO/placeholders found. All steps contain exact code.

**3. Type consistency:**
- `KeyState` dataclass fields match across all task usages
- `parse_keys_config()` returns `list[KeyState]` — consistent in Task 2, 3, 7
- `OpenRouterKeyManager` constructor takes `(keys: list[KeyState], *, timezone_name: str)` — consistent in Task 3, 7
- `get_available_key()` returns `KeyState | None` — consistent in Task 3, 9, 10
- `AllKeysExhaustedError` extends `RateLimitError` — consistent in Task 1, 9, 10
- `mask_key()` signature `(key: str) -> str` — consistent in Task 2, 5
- Provider `__init__` takes `(config: ProviderConfig, *, timezone: str = "UTC")` — consistent in Task 7, 8, 9
- `_active_key()` returns `str` — consistent in Task 8
- `get_status_summary()` returns `dict[str, Any]` — consistent in Task 5, 11
