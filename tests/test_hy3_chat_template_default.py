# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the Hy3 ``reasoning_effort`` default override.

The Tencent Hunyuan 3 (Hy3) chat_template.jinja defaults ``reasoning_effort``
to ``no_think`` which empirically returns "France" instead of "Paris" for
factual-recall questions (upstream PR #1211 comment 4927711484, observed
2026-07-09 during the vendor drop's boot-time spike on
``mlx-community/Hy3-preview-4bit``). Rapid-mlx overrides this default to
``low`` at the ``apply_chat_template`` boundary so out-of-the-box requests
produce correct answers without the client having to learn the template
kwarg.

This test uses a mock template applicator (rather than a live tokenizer)
so it stays hermetic — no HY3 weights required.
"""

from __future__ import annotations

import pytest

from vllm_mlx.utils.chat_template import _looks_like_hy3, apply_chat_template


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hy3-preview-4bit", True),
        ("mlx-community/Hy3-preview-4bit", True),
        ("Hy3-Preview-4bit", True),  # case-insensitive
        ("Hunyuan-3-Preview", True),  # dash separator
        ("hunyuan3", True),
        ("hy-v3-experimental", True),
        # Negatives — must NOT match.
        ("qwen3.5-4b-4bit", False),
        ("qwen3.6-35b-a3b-mtp-4bit", False),
        ("mlx-community/Qwen3.6-27B-4bit", False),
        ("gemma4-27b-8bit", False),
        ("", False),
        # Codex round-3 NIT (PR #1070 finding #4): unanchored substring
        # matches leaked family detection into unrelated names. Lock the
        # boundary behaviour so an incidental ``hunyuanx3`` /
        # ``mymodelhy3embedded`` substring doesn't get Hy3-only kwarg
        # injection.
        ("not-hunyuanx3-test", False),
        ("mymodelhy3embedded", False),
        ("mlx-community/hunyuan5-preview", False),
    ],
)
def test_looks_like_hy3_predicate(name: str, expected: bool) -> None:
    assert _looks_like_hy3(name) is expected


class _CapturingTokenizer:
    """Mock template applicator that captures the kwargs it was called with.

    Mirrors the ``PreTrainedTokenizerBase.apply_chat_template`` shape well
    enough for our purposes — has an ``apply_chat_template`` method that
    returns a stub prompt and records the kwargs so the test can assert
    on the presence / value of ``reasoning_effort``.
    """

    def __init__(self):
        self.captured_kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.captured_kwargs = kwargs
        return "<stub prompt>"


def test_hy3_default_reasoning_effort_low_is_injected():
    """The load-bearing fix. When ``model_name`` looks like Hy3 and the
    caller does NOT pass ``enable_thinking=False``, the default template
    kwarg ``reasoning_effort='low'`` MUST be injected."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Capital of France?"}],
        model_name="mlx-community/Hy3-preview-4bit",
    )
    assert tok.captured_kwargs.get("reasoning_effort") == "low"


def test_hy3_default_not_injected_when_enable_thinking_false():
    """Client explicitly disabled thinking — the ``no_think`` default is
    what they want. The parser MUST respect that intent and NOT override
    ``reasoning_effort``."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Capital of France?"}],
        enable_thinking=False,
        model_name="mlx-community/Hy3-preview-4bit",
    )
    assert "reasoning_effort" not in tok.captured_kwargs


def test_non_hy3_model_never_sees_reasoning_effort_kwarg():
    """Every other family's chat template rejects unknown kwargs with
    ``TypeError``. The Hy3-only injection MUST NOT fire for other models
    or the fallback retry chain would kick in unnecessarily."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Hello"}],
        model_name="qwen3.5-4b-4bit",
    )
    assert "reasoning_effort" not in tok.captured_kwargs


def test_hy3_default_survives_enable_thinking_true():
    """Explicit ``enable_thinking=True`` is orthogonal — thinking is on
    AND the default effort should be ``low`` (not ``no_think``)."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
        model_name="hy3-preview-4bit",
    )
    assert tok.captured_kwargs.get("reasoning_effort") == "low"
    assert tok.captured_kwargs.get("enable_thinking") is True


def test_hy3_default_dropped_only_after_second_type_error():
    """Two-stage retry (codex round-1 NIT fix). The first retry drops
    ``enable_thinking`` and KEEPS ``reasoning_effort`` — a Hy3
    checkpoint that supports the effort override but rejects
    ``enable_thinking`` MUST still see the ``low`` value on retry.
    Only the SECOND TypeError drops ``reasoning_effort``."""

    class EnableThinkingFlakyTokenizer:
        """Rejects ``enable_thinking`` only; accepts ``reasoning_effort``."""

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "enable_thinking" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'enable_thinking'"
                )
            return "<enable_thinking-dropped ok>"

    tok = EnableThinkingFlakyTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        model_name="hy3-preview-4bit",
    )
    # First call had both; retry dropped enable_thinking but preserved
    # reasoning_effort=low. Codex round-3 NIT (PR #1070 finding #5):
    # assert the exact expected value (``True`` — the default when the
    # caller doesn't pass ``enable_thinking`` and the model_name isn't
    # a coder alias) rather than the looser ``is not None`` which
    # would silently accept an invalid value.
    assert len(tok.calls) == 2
    assert tok.calls[0].get("enable_thinking") is True
    assert tok.calls[0].get("reasoning_effort") == "low"
    assert "enable_thinking" not in tok.calls[1]
    assert tok.calls[1].get("reasoning_effort") == "low"
    assert result == "<enable_thinking-dropped ok>"


def test_hy3_default_dropped_when_reasoning_effort_alone_is_rejected():
    """Realistic degradation path — checkpoint accepts ``enable_thinking``
    but not ``reasoning_effort``. First call fails because of
    ``reasoning_effort``; the two-stage retry drops it and Step 2's
    tools-restore path (which re-adds ``enable_thinking``) succeeds."""

    class ReasoningEffortFlakyTokenizer:
        """Rejects ``reasoning_effort`` only; accepts ``enable_thinking``."""

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'reasoning_effort'"
                )
            return "<reasoning_effort-dropped ok>"

    tok = ReasoningEffortFlakyTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        model_name="hy3-preview-4bit",
    )
    # First call had reasoning_effort=low and enable_thinking; failed.
    # Retry-1 dropped enable_thinking but kept reasoning_effort; failed.
    # Second-TypeError handler dropped reasoning_effort; Step 2
    # tools-restore re-added enable_thinking; the final call succeeded.
    assert tok.calls[0].get("reasoning_effort") == "low"
    assert "reasoning_effort" not in tok.calls[-1]
    assert result == "<reasoning_effort-dropped ok>"
