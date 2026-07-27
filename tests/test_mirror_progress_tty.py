# SPDX-License-Identifier: Apache-2.0
"""The mirror pull's byte-progress renderer must adapt to its output sink.

Two contracts share one ``_ProgressTracker``:

* **Piped stdout** (the desktop app's captured subprocess, a redirected log,
  pytest capture) keeps the machine-readable ``[bytes] D/T`` heartbeat the
  desktop's ``DownloadProgress`` parser reads. Breaking this format silently
  freezes the desktop progress bar — the regression this file guards.
* **A human TTY** (someone running bare ``rapid-mlx chat`` in a terminal) gets
  ONE in-place bar redrawn with ``\\r`` instead of the hundreds of raw
  ``[bytes]`` lines a multi-GB cold pull used to scroll (0.11 first-run
  dogfood papercut). Per-file completion lines route through ``write_line`` so
  they erase the bar cleanly rather than shredding it.

These are direct unit tests of the tracker: fast and hermetic, no network.
``_mirror._PROGRESS_HEARTBEAT_SECONDS`` is dropped to 0 so every ``add`` emits.
"""

from __future__ import annotations

import io
import sys

import pytest

from vllm_mlx import _mirror
from vllm_mlx._mirror import _ProgressTracker


class _FakeTTY(io.StringIO):
    """A captured stdout that claims to be an interactive terminal."""

    def isatty(self) -> bool:
        return True


class _FakePipe(io.StringIO):
    """A captured stdout that is NOT a terminal (desktop pipe / log)."""

    def isatty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _every_add_emits(monkeypatch):
    # No throttle — every add() renders, so a single add is observable.
    monkeypatch.setattr(_mirror, "_PROGRESS_HEARTBEAT_SECONDS", 0.0)
    # NO_COLOR forces the non-TTY branch regardless of isatty() — clear it so
    # the _FakeTTY tests actually exercise the human bar.
    monkeypatch.delenv("NO_COLOR", raising=False)


def test_non_tty_keeps_machine_bytes_heartbeat(monkeypatch):
    """Piped stdout emits ``[bytes] D/T`` and never the human bar."""
    fake = _FakePipe()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(400)
    t.add(600)  # done == total
    t.flush()
    out = fake.getvalue()
    assert "[bytes] 400/1000" in out
    assert "[bytes] 1000/1000" in out
    # No terminal control sequences or bar glyphs leak onto a pipe.
    assert "\r" not in out
    assert "↓" not in out
    assert "\x1b" not in out


def test_tty_draws_single_inplace_bar(monkeypatch):
    """A human terminal gets one carriage-return bar, no ``[bytes]`` lines."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(250)
    t.add(250)  # 50%
    mid = fake.getvalue()
    assert "[bytes]" not in mid  # machine format never shown to a human
    assert "\r" in mid and "↓" in mid  # in-place bar
    assert "50%" in mid
    assert not mid.endswith("\n")  # mid-stream redraw carries no newline
    t.add(500)  # 100%
    t.flush()
    final = fake.getvalue()
    assert "100%" in final
    assert final.endswith("\n")  # flush finalizes the bar's row


def test_tty_bar_clamps_display_at_100_percent(monkeypatch):
    """An oversized/corrupt stream can't render >100% (same clamp as add)."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(1500)  # Content-Length lied / proxy injected bytes
    out = fake.getvalue()
    assert "100%" in out
    assert "150%" not in out


def test_tty_write_line_erases_bar_then_prints(monkeypatch):
    """A completion line erases the live bar and lands on its own row."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(300)  # bar now live
    t.write_line("  [1/3] config.json")
    out = fake.getvalue()
    # erase-line sequence, the completion text, then a newline
    assert "\r\x1b[2K  [1/3] config.json\n" in out


def test_non_tty_write_line_is_plain_print(monkeypatch):
    """On a pipe, write_line == the old ``_print_dim`` (byte-identical)."""
    fake = _FakePipe()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.write_line("  [1/3] config.json")
    out = fake.getvalue()
    assert out == "  [1/3] config.json\n"
    assert "\r" not in out and "\x1b" not in out


def test_tty_stays_silent_when_total_unknown(monkeypatch):
    """total==0 (HF didn't expose sizes) → no bar, no divide-by-zero."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=0)
    t.add(500)
    t.flush()
    assert fake.getvalue() == ""


def test_no_color_tty_keeps_bar_without_ansi(monkeypatch):
    """NO_COLOR must NOT re-trigger the machine ``[bytes]`` flood on a
    terminal — that's the exact regression this feature removes, and the
    NO_COLOR user is precisely who wants clean output. NO_COLOR only drops
    ANSI: the bar still redraws in place via ``\\r`` + space-pad, emitting
    zero escape sequences.
    """
    monkeypatch.setenv("NO_COLOR", "1")  # re-set (autouse fixture cleared it)
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(250)
    t.add(250)  # 50%
    t.write_line("  [1/3] config.json")  # erases the bar without ANSI
    t.add(500)  # 100%
    t.flush()
    out = fake.getvalue()
    assert "[bytes]" not in out  # no machine flood for a NO_COLOR terminal
    assert "\x1b" not in out  # zero ANSI escapes
    assert "\r" in out  # still an in-place bar
    assert "↓" in out and "100%" in out
    assert "[1/3] config.json" in out
    assert out.endswith("\n")  # flush finalized the row
