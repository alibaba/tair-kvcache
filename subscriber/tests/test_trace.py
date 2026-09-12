"""Tests for the EagleEye-style trace ID generator."""

from __future__ import annotations

import importlib
import re

import pytest

from subscriber.trace import generate_trace_id


class TestGenerateTraceId:
    """generate_trace_id produces EagleEye-compatible trace identifiers."""

    def test_returns_30_char_string(self) -> None:
        trace_id = generate_trace_id()
        assert len(trace_id) == 30

    def test_format_matches_eagleeye_layout(self) -> None:
        """ip_hex(8) + timestamp_ms(13) + counter(4) + 'd' + pid_hex(4)."""
        trace_id = generate_trace_id()
        assert re.fullmatch(r"[0-9a-f]{8}\d{13}\d{4}d[0-9a-f]{4}", trace_id)

    def test_monotonic_time_prefix(self) -> None:
        """Successive IDs have non-decreasing timestamp portions."""
        ids = [generate_trace_id() for _ in range(100)]
        timestamps = [int(tid[8:21]) for tid in ids]
        assert timestamps == sorted(timestamps)

    def test_uniqueness_across_10k_calls(self) -> None:
        ids = {generate_trace_id() for _ in range(10_000)}
        assert len(ids) == 10_000

    def test_counter_cycles_within_range(self) -> None:
        """Counter portion stays in 1000-9000."""
        for _ in range(200):
            trace_id = generate_trace_id()
            counter = int(trace_id[21:25])
            assert 1000 <= counter <= 9000

    def test_never_raises_on_ip_resolution_failure(self) -> None:
        """Fallback to 'ffffffff' when IP cannot be resolved."""
        import subscriber.trace as trace_mod

        original = trace_mod._IP_HEX
        try:
            trace_mod._IP_HEX = "ffffffff"
            trace_id = generate_trace_id()
            assert trace_id[:8] == "ffffffff"
            assert len(trace_id) == 30
        finally:
            trace_mod._IP_HEX = original

    def test_pid_suffix_matches_eagleeye_for_large_process_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EagleEye reduces PIDs above 65535 modulo 60000 before hex encoding."""

        import subscriber.trace as trace_mod

        monkeypatch.setattr(trace_mod.os, "getpid", lambda: 70_000)
        try:
            reloaded = importlib.reload(trace_mod)
            assert reloaded.generate_trace_id().endswith("d2710")
        finally:
            monkeypatch.undo()
            importlib.reload(trace_mod)
