"""Tests for OpenRouter key manager: parsing, selection, blocking, masking."""

import json
import time
from dataclasses import fields
from datetime import UTC, datetime
from unittest.mock import patch

from providers.open_router.key_manager import (
    AllKeysExhaustedError,
    KeyState,
    OpenRouterKeyManager,
    mask_key,
    parse_keys_config,
)


def _make_manager(keys_str: str, *, timezone_name: str = "UTC") -> OpenRouterKeyManager:
    keys = parse_keys_config(keys_str)
    return OpenRouterKeyManager(keys, timezone_name=timezone_name)


def test_all_keys_exhausted_error_importable():
    err = AllKeysExhaustedError("all keys exhausted")
    assert str(err) == "all keys exhausted"
    assert err.status_code == 429
    assert err.error_type == "rate_limit_error"


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
            "api_key",
            "daily_limit",
            "success_count",
            "last_used_at",
            "blocked_until",
            "block_reason",
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


class TestReportRateLimit:
    def test_rpm_rate_limit_blocks_10s(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps(
            {"error": {"code": 429, "message": "Rate limit exceeded"}}
        )
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].blocked_until > time.monotonic()
        assert mgr._keys[0].blocked_until < time.monotonic() + 15
        assert mgr._keys[0].block_reason == "rpm"

    def test_upstream_provider_limit_blocks_30s(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": "Provider returned error: model is temporarily rate-limited upstream",
                }
            }
        )
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].blocked_until > time.monotonic() + 25
        assert mgr._keys[0].blocked_until < time.monotonic() + 40
        assert mgr._keys[0].block_reason == "upstream"

    def test_daily_limit_blocks_until_midnight(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded: free-models-per-day. Add 10 credits",
                }
            }
        )
        mgr.report_rate_limit("sk-a", error_body)
        assert mgr._keys[0].block_reason == "daily_limit"
        assert mgr._keys[0].success_count == 50

    def test_credits_exhausted_blocks_until_midnight(self):
        mgr = _make_manager("sk-a:50")
        error_body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": "Credits exhausted for free models",
                }
            }
        )
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
        summary = mgr.get_status_summary()
        assert "next_reset_at" in summary
        datetime.fromisoformat(summary["next_reset_at"].replace("Z", "+00:00"))


class TestDailyReset:
    def test_resets_counters_when_midnight_passed(self):
        mgr = _make_manager("sk-a:50")
        mgr._keys[0].success_count = 50
        # Set last reset to yesterday
        yesterday = datetime(2026, 5, 15, 0, 0, 0, tzinfo=UTC)
        mgr._last_reset_date = yesterday.date()
        # Patch "now" to be after midnight
        with patch("providers.open_router.key_manager.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 16, 1, 0, 0, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            key = mgr.get_available_key()
            assert key is not None
            assert key.success_count == 0

    def test_no_reset_when_same_day(self):
        mgr = _make_manager("sk-a:50")
        mgr._keys[0].success_count = 25
        today = datetime(2026, 5, 16, 0, 0, 0, tzinfo=UTC)
        mgr._last_reset_date = today.date()
        with patch("providers.open_router.key_manager.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            key = mgr.get_available_key()
            assert key is not None
            assert key.success_count == 25


class TestSettingsTimezone:
    def test_open_router_timezone_default(self):
        from config.settings import Settings

        s = Settings(
            model="open_router/test",
            open_router_api_key="sk-test",
        )
        assert s.open_router_timezone == "UTC"

    def test_open_router_timezone_from_env(self, monkeypatch):
        from config.settings import Settings

        monkeypatch.setenv("OPENROUTER_TIMEZONE", "America/New_York")
        monkeypatch.setattr(
            Settings,
            "model_config",
            {**Settings.model_config, "env_file": None},
        )
        s = Settings(
            model="open_router/test",
            open_router_api_key="sk-test",
        )
        assert s.open_router_timezone == "America/New_York"
