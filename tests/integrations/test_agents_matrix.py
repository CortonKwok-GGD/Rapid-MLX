# SPDX-License-Identifier: Apache-2.0
"""Tier-1 agents × 4 families integration matrix (0.10.2 PR-2).

Eight Tier-1 agents from ``0.10-TODO.md`` §0.10.2, each exercised
against the four Tier-1 family strong picks:

* codex-cli (/v1/responses)
* claude-code (/v1/messages via Anthropic SDK)
* opencode (/v1/chat/completions)
* qwen-code (/v1/chat/completions)
* openhands (/v1/chat/completions — text-action wire smoke only)
* hermes-agent (/v1/chat/completions — tool-call wire; deep in test_hermes.py)
* aider (/v1/chat/completions — wire smoke; edit-and-write in test_aider.sh)
* kilo-code (/v1/chat/completions)

Each cell drives the agent's real wire against a booted rapid-mlx
server (auto-boot by conftest; see ``rapid_mlx_server`` fixture). No
mocks, no synthetic fixtures — the assertion catches the actual bytes
the server put on the socket.

Cell assertions (shared shape):

1. 2xx response
2. Non-empty visible content
3. No ``<think>...</think>`` leak (Qwen / DeepSeek)
4. No ``<|channel|>analysis`` leak (gpt-oss Harmony)
5. Tool-call cells: shape valid, function name matches, arguments
   parse as JSON, arguments carry the semantic slot (``city``,
   ``operation``, etc.).

Timeout: ``DEFAULT_TIMEOUT_S`` (180 s) per cell — 35B/120B decodes are
slow; the wire is what we're testing, not decode speed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.integrations.conftest import (
    DEFAULT_TIMEOUT_S,
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    assert_tool_call_shape,
    strict_skip_or_fail,
)

# --------------------------------------------------------------------------- #
# Shared per-cell tool-call payloads
# --------------------------------------------------------------------------- #


_WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_WEATHER_PROMPT = "What's the weather in Tokyo? Call the get_weather tool."


def _openai_client_and_errors(base_url: str):
    """Lazy openai import — the pkg is optional in the base venv."""
    try:
        from openai import (
            APIStatusError,
            BadRequestError,
            NotFoundError,
            OpenAI,
        )
    except ImportError:
        pytest.skip("openai package not installed — agent matrix skipped")
    client = OpenAI(
        base_url=base_url,
        api_key="not-needed",
        timeout=DEFAULT_TIMEOUT_S,
    )
    return client, (BadRequestError, NotFoundError, APIStatusError)


def _extract_text_from_message(msg) -> str:
    """Coerce a chat.completions message content into str."""
    content = getattr(msg, "content", None) or ""
    if isinstance(content, str):
        return content
    # Some SDK builds return content parts.
    parts = []
    for part in content:
        text = getattr(part, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _run_openai_tool_smoke(
    rapid_mlx_server: dict[str, Any],
    family_alias: FamilyAlias,
    *,
    agent_label: str,
) -> None:
    """Run one tool-call cell against ``/v1/chat/completions``.

    Shared by every Tier-1 agent that speaks the OpenAI wire (opencode,
    qwen-code, hermes-agent, kilo-code). Asserts the wire response is
    either a valid tool_call or (fallback) non-empty visible content
    that doesn't leak reasoning-channel markers. In strict mode a missing
    tool_call fails so the tool-call plumbing regression the matrix
    exists to catch can't hide.
    """
    client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
    model_id = rapid_mlx_server["model_id"]

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": _WEATHER_PROMPT}],
            tools=_WEATHER_TOOL,
            temperature=0.0,
            max_tokens=384,
        )
    except wire_errors as exc:
        strict_skip_or_fail(
            f"{agent_label}/{family_alias.family}: server rejected tool request "
            f"on {model_id!r}: {exc}"
        )
        return

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    content = _extract_text_from_message(msg)

    # Regardless of route, visible content must not leak channel markers.
    assert_no_think_tag_leak(content)
    assert_no_analysis_channel_leak(content)

    if not tool_calls:
        # A strong-pick model that answers inline still failed to route
        # through tool_calls — a real regression signal for the Tier-1 gate.
        # Non-strict local runs skip so a dev without ``--enable-auto-tool-choice``
        # on a hand-booted server doesn't get spurious reds.
        strict_skip_or_fail(
            f"{agent_label}/{family_alias.family}: {model_id!r} answered inline "
            f"instead of calling get_weather (content={content[:120]!r}). "
            f"Strict CI treats this as a tool-call wire regression."
        )
        return

    tc = tool_calls[0]
    tc_dict = {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    assert_tool_call_shape(tc_dict)
    assert tc.function.name == "get_weather", tc.function.name
    args = json.loads(tc.function.arguments)
    assert "city" in args, args
    assert "tokyo" in args["city"].lower(), args


# --------------------------------------------------------------------------- #
# Cells — one per Tier-1 agent
# --------------------------------------------------------------------------- #


class TestCodexCLI:
    """Codex CLI /v1/responses — stateless shim (see codex.yaml)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        import httpx

        base_url = rapid_mlx_server["base_url"]
        model_id = rapid_mlx_server["model_id"]

        payload = {
            "model": model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with just SHIPPED."}
                    ],
                }
            ],
            "stream": False,
            "max_output_tokens": 64,
        }
        try:
            r = httpx.post(
                f"{base_url}/responses",
                json=payload,
                timeout=DEFAULT_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: transport error hitting "
                f"/v1/responses: {exc!r}"
            )
            return
        if r.status_code in (404, 405):
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: /v1/responses returned "
                f"{r.status_code} — route not wired on this server."
            )
            return
        if r.status_code >= 400:
            strict_skip_or_fail(
                f"codex-cli/{family_alias.family}: server returned {r.status_code} "
                f"({r.text[:200]!r})"
            )
            return
        data = r.json()
        outputs = data.get("output") or []
        assert outputs, f"empty output envelope: {data}"
        text = ""
        for output_msg in outputs:
            for block in output_msg.get("content", []) or []:
                if block.get("type") in ("output_text", "text"):
                    text = block.get("text", "") or ""
                    if text:
                        break
            if text:
                break
        assert_content_nonempty(text, ctx=f"codex-cli/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestClaudeCode:
    """Claude Code /v1/messages — Anthropic SDK route (see test_anthropic_sdk.py)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from anthropic import (
                Anthropic,
                APIStatusError,
                BadRequestError,
                NotFoundError,
            )
        except ImportError:
            pytest.skip("anthropic SDK not installed — cell deferred")

        base_no_v1 = rapid_mlx_server["base_url"].rstrip("/").removesuffix("/v1")
        client = Anthropic(
            base_url=base_no_v1,
            api_key="not-needed",
            timeout=DEFAULT_TIMEOUT_S,
        )

        try:
            resp = client.messages.create(
                model=rapid_mlx_server["model_id"],
                max_tokens=128,
                messages=[{"role": "user", "content": "Reply with just SHIPPED."}],
            )
        except NotFoundError:
            strict_skip_or_fail(
                f"claude-code/{family_alias.family}: /v1/messages returned 404 "
                f"on {rapid_mlx_server['base_url']} — Anthropic route not wired."
            )
            return
        except (BadRequestError, APIStatusError) as exc:
            strict_skip_or_fail(
                f"claude-code/{family_alias.family}: server rejected request: {exc}"
            )
            return

        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        assert_content_nonempty(text, ctx=f"claude-code/{family_alias.family}")
        assert_no_think_tag_leak(text)
        assert_no_analysis_channel_leak(text)


class TestOpenCode:
    """OpenCode /v1/chat/completions with tool call."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="opencode")


class TestQwenCode:
    """Qwen Code /v1/chat/completions with tool call."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="qwen-code")


class TestOpenHands:
    """OpenHands wire smoke — text-action format, not OpenAI function calls."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
        try:
            resp = client.chat.completions.create(
                model=rapid_mlx_server["model_id"],
                messages=[{"role": "user", "content": "Reply with just OK."}],
                temperature=0.0,
                max_tokens=64,
            )
        except wire_errors as exc:
            strict_skip_or_fail(
                f"openhands/{family_alias.family}: server rejected request: {exc}"
            )
            return
        content = _extract_text_from_message(resp.choices[0].message)
        assert_content_nonempty(content, ctx=f"openhands/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)


class TestHermesAgent:
    """Hermes Agent — tool-call over OpenAI wire (deep flow in test_hermes.py)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(
            rapid_mlx_server, family_alias, agent_label="hermes-agent"
        )


class TestAider:
    """Aider wire smoke — deep flow in test_aider.sh CLI harness."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
        try:
            resp = client.chat.completions.create(
                model=rapid_mlx_server["model_id"],
                messages=[
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "Say hi."},
                ],
                temperature=0.0,
                max_tokens=64,
            )
        except wire_errors as exc:
            strict_skip_or_fail(
                f"aider/{family_alias.family}: server rejected request: {exc}"
            )
            return
        content = _extract_text_from_message(resp.choices[0].message)
        assert_content_nonempty(content, ctx=f"aider/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)


class TestKiloCode:
    """Kilo Code /v1/chat/completions with tool call (Cline fork)."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        _run_openai_tool_smoke(rapid_mlx_server, family_alias, agent_label="kilo-code")


class TestStreamingDeltas:
    """Cross-agent streaming assertion — SSE parses cleanly per family.

    Shared cell — every family should stream via ``/v1/chat/completions``
    without dropping tokens or leaking channel markers into deltas.
    Split off from per-agent cells because streaming validation is
    orthogonal to tool-call routing.

    Codex #1033 round-4 BLOCKING #2: leak assertions run on the
    per-delta text AND on the final assembled content. A server that
    leaks ``<think>`` / ``<|channel|>analysis`` mid-stream but strips
    them from the assembled final object would previously pass — a
    real regression (mid-stream leaks are visible to a client that
    renders as tokens arrive). The concatenated ``streamed_text`` gates
    that.
    """

    def test_stream_deltas(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
        # Import here so a missing openai SDK skips at the imports helper
        # rather than crashing at the module level below.
        try:
            from openai import LengthFinishReasonError
        except ImportError:  # very old openai
            LengthFinishReasonError = ()  # type: ignore[misc]
        streamed_text = ""
        # gpt-oss reasoning models emit long analysis-channel content
        # BEFORE the final content. 256 tokens is enough for a "count
        # to three" prompt to reach the assistant-channel output on
        # reasoning models; 64 was tuned for the small aliases and
        # cut the stream mid-reasoning on the strong-pick gpt-oss 120B.
        max_stream_tokens = 256
        final = None
        try:
            events: list[str] = []
            with client.chat.completions.stream(
                model=rapid_mlx_server["model_id"],
                messages=[{"role": "user", "content": "Count to three."}],
                temperature=0.0,
                max_tokens=max_stream_tokens,
            ) as stream:
                for event in stream:
                    events.append(getattr(event, "type", "unknown"))
                    # Accumulate content deltas so per-token leaks
                    # surface even if the final object was scrubbed.
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        streamed_text += delta
                        assert_no_think_tag_leak(delta)
                        assert_no_analysis_channel_leak(delta)
                try:
                    final = stream.get_final_completion()
                except LengthFinishReasonError:
                    # gpt-oss ran the reasoning past max_tokens without
                    # reaching the assistant-channel final — the per-
                    # delta assertions above already covered the leak
                    # gate, so this is not a regression. Fall through
                    # with final=None; the assembled streamed_text
                    # assertion below is the only remaining gate.
                    final = None
        except wire_errors as exc:
            strict_skip_or_fail(
                f"stream/{family_alias.family}: server rejected stream: {exc}"
            )
            return
        except AttributeError:
            # older openai versions don't have .stream context manager
            pytest.skip("openai SDK too old for .stream context manager")
            return
        assert events, "no stream events collected"
        # Concatenated-delta gate: catches leaks that were split across
        # deltas so no single delta contained the marker literally.
        assert_no_think_tag_leak(streamed_text)
        assert_no_analysis_channel_leak(streamed_text)
        # Assembled-final gate — only when the SDK gave us one. On
        # gpt-oss reasoning-heavy models that hit max_tokens mid-
        # reasoning, the per-delta gates above are the only assertion
        # (and they've already run).
        if final is not None:
            text = _extract_text_from_message(final.choices[0].message)
            assert_content_nonempty(text, ctx=f"stream/{family_alias.family}")
            assert_no_think_tag_leak(text)
            assert_no_analysis_channel_leak(text)
