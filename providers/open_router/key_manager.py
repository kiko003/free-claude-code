"""OpenRouter multi-key rotation: parsing, state tracking, and selection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from providers.exceptions import RateLimitError


class AllKeysExhaustedError(RateLimitError):
    """All OpenRouter API keys are exhausted (blocked or daily quota reached)."""

    def __init__(self, message: str = "All OpenRouter keys exhausted") -> None:
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


class OpenRouterKeyManager:
    """Quota-aware key rotation for OpenRouter multi-key setups."""

    _RPM_MIN_INTERVAL = 3.0  # 20 RPM = 1 req per 3s

    def __init__(
        self,
        keys: list[KeyState],
        *,
        timezone_name: str = "UTC",
    ) -> None:
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
        except json.JSONDecodeError, AttributeError:
            error_message = ""
            error_code = None

        if str(error_code) == "402":
            ks.blocked_until = now + 300  # 5 minutes
            ks.block_reason = "upstream"
            return

        if str(error_code) == "429":
            if (
                "free-models-per-day" in error_message
                or "Credits exhausted" in error_message
            ):
                ks.block_reason = "daily_limit"
                ks.blocked_until = 0.0  # cleared by daily reset
                ks.success_count = ks.daily_limit
                return
            if (
                "Provider returned error" in error_message
                or "upstream" in error_message.lower()
            ):
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

    def get_key_statuses(self) -> list[dict[str, Any]]:
        """Return status info for all keys (for the key-status endpoint)."""
        now = time.monotonic()
        statuses: list[dict[str, Any]] = []
        for ks in self._keys:
            remaining = max(0, ks.daily_limit - ks.success_count)
            remaining_pct = (
                round((remaining / ks.daily_limit) * 100, 1) if ks.daily_limit else 0.0
            )
            is_blocked = ks.blocked_until > now or ks.success_count >= ks.daily_limit
            blocked_until = ks.blocked_until if ks.blocked_until > now else None
            statuses.append(
                {
                    "api_key": mask_key(ks.api_key),
                    "daily_limit": ks.daily_limit,
                    "success_count": ks.success_count,
                    "remaining": remaining,
                    "remaining_pct": remaining_pct,
                    "blocked": is_blocked,
                    "blocked_until": blocked_until,
                    "block_reason": ks.block_reason if is_blocked else None,
                    "last_used_at": ks.last_used_at if ks.last_used_at > 0 else None,
                }
            )
        return statuses

    def get_status_summary(self) -> dict[str, Any]:
        """Return full key-status response including next reset time."""
        now = datetime.now(self._tz)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return {
            "keys": self.get_key_statuses(),
            "next_reset_at": next_midnight.isoformat().replace("+00:00", "Z"),
        }
