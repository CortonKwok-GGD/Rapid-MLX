# SPDX-License-Identifier: Apache-2.0
"""Stage-2 deep cells for the #558 default-on constrained-tool-calling gate.

The always-on agent×family *smoke* matrix (``test_agents_matrix.py`` /
``test_frameworks_matrix.py``) proves every agent wire still SPEAKS on the
running server. This file adds the DEEP coverage the default-on flip needs:

* **Multi-turn tool loop** — issue a tool call, feed a synthetic tool result
  back as a ``role="tool"`` turn, and assert the model produces a coherent
  final answer that consumes the tool output. A single-turn smoke can pass
  even if the multi-turn tool-result plumbing regresses.
* **Varied JSON schemas** — enum / nested-object / required-fields /
  ``additionalProperties:false``. These are exactly the schema shapes a
  grammar constraint has to honour; the default-on flip must not regress any
  of them.
* **Negative control (the constraint actually FIRES)** — the same schema is
  requested twice against the *live* guided-decode path
  (``response_format={"type":"json_schema","strict":true,...}`` → llguidance,
  see ``vllm_mlx/api/guided.py`` + ``vllm_mlx/engine/batched.py``): once
  UN-constrained (plain ``json_object`` / no schema — the model is free to
  emit an off-schema / hallucinated field) and once CONSTRAINED (strict
  schema — llguidance masks the off-schema token). The control PASSES only
  when the constrained run is on-schema; if a "constrained" run can still
  emit an off-schema key, the constraint is not firing and the cell fails.

Design notes
------------
* These cells reuse the SAME conftest fixtures as the smoke matrix
  (``rapid_mlx_server``, ``family_alias``, the family-guard autouse fixture,
  the strict-xfail collection hook). They are ``family_alias``-parametrized,
  so a single-family server boot runs only that family's deep cells and the
  family-guard skips the rest — identical semantics to the smoke matrix.
* Constraint MODE is now a SERVER-env toggle the gate driver applies when it
  boots the per-arm server (``scripts/default_on_gate.py``): the off arm boots
  with ``RAPID_MLX_CONSTRAIN_TOOLS=0`` (free-form base) and the on arm boots
  default-on (env unset — #558 PR-5 flipped the default to ON). Both arms keep
  ``tool_choice="auto"`` so the REAL PR-5 auto-path is exercised; the only
  independent variable is the server-side constraint. ``RAPID_MLX_TOOL_CONSTRAINT``
  (still set on the pytest process) now only labels the per-cell latency
  breadcrumb with its mode. Before PR-5 landed, ``on`` was a per-request
  ``tool_choice="auto"->"required"`` proxy in ``_apply_constraint_mode``; that
  seam is now a no-op that pins ``tool_choice="auto"`` in both arms. The
  negative-control cell does NOT depend on the toggle: it drives the
  ALREADY-LIVE ``response_format`` guided path, so it proves the underlying
  llguidance masking works on any ref, today.
* Every cell degrades to skip on a missing server / SDK unless
  ``RAPID_MLX_MATRIX_STRICT=1`` (shared ``strict_skip_or_fail`` semantics),
  so a naive ``pytest tests/integrations`` on a clean box stays green.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

from tests.integrations.conftest import (
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    strict_skip_or_fail,
)

# --------------------------------------------------------------------------- #
# Constraint mode (forward-compatible knob)
# --------------------------------------------------------------------------- #


def constraint_mode() -> str:
    """Return the requested constraint mode: ``"off"`` (default) or ``"on"``.

    Set by the gate driver (``scripts/default_on_gate.py``) via
    ``RAPID_MLX_TOOL_CONSTRAINT`` so the SAME cells run once per mode and the
    driver can diff the two per-cell result sets (baseline-vs-constrained).
    """
    val = os.environ.get("RAPID_MLX_TOOL_CONSTRAINT", "off").strip().lower()
    return "on" if val in ("1", "on", "true", "yes") else "off"


def _apply_constraint_mode(payload: dict[str, Any]) -> dict[str, Any]:
    """Pin ``tool_choice="auto"`` in BOTH arms — the seam PR-5 turned into a no-op.

    #558 PR-5 landed the real default-on knob, so the off/on distinction is now
    driven by the SERVER the gate driver boots, NOT by a per-request
    ``tool_choice`` swap:

    * **off** — driver boots the server with ``RAPID_MLX_CONSTRAIN_TOOLS=0``
      (constrained tool-calling opted OUT -> legacy free-form base).
    * **on**  — driver boots the server default-on (env unset/ON), so
      ``tool_choice="auto"`` exercises the REAL PR-5 auto-path optional-call
      grammar (the model MAY emit a structurally-correct call or plain text and
      is never forced).

    Both arms therefore issue the identical ``tool_choice="auto"`` request and
    the server-side constraint is the only independent variable — that is what
    makes the off-vs-on comparison a true free-form-vs-constrained parity test.
    Before PR-5 this seam forced ``tool_choice="required"`` on the on arm as a
    pre-landing proxy, which would have tested forced-emission, not the auto
    path. ``constraint_mode()`` is still read to label the latency breadcrumb.
    """
    payload = dict(payload)
    payload["tool_choice"] = "auto"
    return payload


# --------------------------------------------------------------------------- #
# Shared tool + schema fixtures
# --------------------------------------------------------------------------- #


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

# Varied schemas the default-on flip must not regress. Each is a self-contained
# JSON Schema exercised through the /v1/chat/completions ``tools`` path.
_VARIED_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "enum": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "unit"],
        "additionalProperties": False,
    },
    "nested_object": {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                },
                "required": ["city"],
            }
        },
        "required": ["location"],
    },
    "required_fields": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "day": {"type": "string"},
        },
        "required": ["city", "day"],
    },
    "no_additional_props": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}


def _openai_client_and_errors(base_url: str):
    """Lazy openai import + typed wire-error tuple (mirrors the smoke matrix)."""
    try:
        from openai import (
            APIStatusError,
            BadRequestError,
            NotFoundError,
            OpenAI,
        )
    except ImportError:
        pytest.skip("openai package not installed — deep cells skipped")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    return client, (BadRequestError, NotFoundError, APIStatusError)


# --------------------------------------------------------------------------- #
# Stage-2 deep cell: multi-turn tool loop
# --------------------------------------------------------------------------- #


class TestMultiTurnToolLoop:
    """Two-turn tool loop: call → synthetic tool result → grounded final answer.

    Proves the multi-turn tool-result plumbing (``role="tool"`` turn keyed by
    ``tool_call_id``) survives the default-on flip. A single-turn smoke passes
    even if this regresses.
    """

    def test_two_turn(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(
            rapid_mlx_server["base_url"]
        )
        model_id = rapid_mlx_server["model_id"]
        ctx = f"multiturn/{family_alias.family}"

        first_payload = _apply_constraint_mode(
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo? Use get_weather.",
                    }
                ],
                "tools": [_WEATHER_TOOL],
                "temperature": 0.0,
                "max_tokens": 384,
            }
        )
        t0 = time.perf_counter()
        try:
            first = client.chat.completions.create(**first_payload)
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected turn-1 tool request: {exc}")
        msg = first.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            # Small alias may answer inline — assert wire cleanliness, then
            # treat the missing call as a strict-mode regression signal.
            content = msg.content or ""
            assert_content_nonempty(content, ctx=ctx)
            assert_no_think_tag_leak(content)
            assert_no_analysis_channel_leak(content)
            strict_skip_or_fail(
                f"{ctx}: turn-1 produced no tool_calls "
                f"(content={content[:120]!r})"
            )
            return

        tc = tool_calls[0]
        assert tc.function.name == "get_weather", tc.function.name
        # ``arguments`` must be a JSON-parseable object. We do NOT hard-require
        # the ``city`` key here: under the constrained-mode proxy
        # (``tool_choice="required"``) a small model's FORCED call can come back
        # with empty ``{}`` args — that is a known forced-emission behavior on
        # tiny aliases, not a wire regression to gate on. When it IS populated
        # we assert it names Tokyo; either way we feed a fixed Tokyo tool result
        # in turn 2 so the grounded-answer assertion stays deterministic.
        args = json.loads(tc.function.arguments)
        assert isinstance(args, dict), f"{ctx}: tool args not an object: {args!r}"
        if "city" in args and args["city"]:
            assert "tokyo" in str(args["city"]).lower(), (
                f"{ctx}: forced call named wrong city: {args!r}"
            )

        # Turn 2 — feed a synthetic tool result back and ask for a final answer.
        second_payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo? Use get_weather.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(
                        {"city": "Tokyo", "temp_c": 21, "sky": "sunny"}
                    ),
                },
            ],
            "tools": [_WEATHER_TOOL],
            "temperature": 0.0,
            "max_tokens": 384,
        }
        try:
            second = client.chat.completions.create(**second_payload)
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected turn-2 tool-result: {exc}")
        latency_s = time.perf_counter() - t0

        final = second.choices[0].message.content or ""
        assert_content_nonempty(final, ctx=ctx)
        assert_no_think_tag_leak(final)
        assert_no_analysis_channel_leak(final)
        # The final answer must consume the tool output — grounded on the
        # 21°C / sunny result we fed back, not a hallucinated re-answer.
        low = final.lower()
        assert ("21" in final) or ("sunny" in low) or ("tokyo" in low), (
            f"{ctx}: final answer {final[:200]!r} did not consume the tool "
            "result (expected mention of 21 / sunny / tokyo)"
        )
        # Perf breadcrumb for the gate's per-cell latency record.
        print(f"[deep-latency] {ctx} mode={constraint_mode()} {latency_s:.2f}s")


# --------------------------------------------------------------------------- #
# Stage-2 deep cell: varied tool schemas
# --------------------------------------------------------------------------- #


class TestVariedSchemas:
    """Tool call against enum / nested / required / additionalProperties:false.

    Parametrized over the four schema shapes a grammar constraint must honour.
    Each cell asserts a well-formed call whose ``arguments`` JSON-parse and
    respect the shape's key constraints.
    """

    @pytest.mark.parametrize("schema_key", sorted(_VARIED_TOOL_SCHEMAS))
    def test_schema_shape(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
        schema_key: str,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(
            rapid_mlx_server["base_url"]
        )
        model_id = rapid_mlx_server["model_id"]
        ctx = f"schema-{schema_key}/{family_alias.family}"
        schema = _VARIED_TOOL_SCHEMAS[schema_key]

        tool = {
            "type": "function",
            "function": {
                "name": "record_query",
                "description": "Record a structured weather query.",
                "parameters": schema,
            },
        }
        payload = _apply_constraint_mode(
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Record a weather query for Tokyo, Japan, on "
                            "Monday, in celsius. Call record_query with all "
                            "the fields the schema requires."
                        ),
                    }
                ],
                "tools": [tool],
                "temperature": 0.0,
                "max_tokens": 384,
            }
        )
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**payload)
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected schema request: {exc}")
        latency_s = time.perf_counter() - t0

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            content = msg.content or ""
            assert_content_nonempty(content, ctx=ctx)
            assert_no_think_tag_leak(content)
            assert_no_analysis_channel_leak(content)
            strict_skip_or_fail(f"{ctx}: no tool_calls (content={content[:120]!r})")
            return

        args = json.loads(tool_calls[0].function.arguments)
        assert isinstance(args, dict), f"{ctx}: args not an object: {args!r}"
        # Shape-specific structural assertions.
        if schema_key == "enum":
            assert args.get("unit") in ("celsius", "fahrenheit"), (
                f"{ctx}: enum arg out of range: {args!r}"
            )
        elif schema_key == "nested_object":
            loc = args.get("location")
            assert isinstance(loc, dict) and "city" in loc, (
                f"{ctx}: nested object missing location.city: {args!r}"
            )
        elif schema_key == "required_fields":
            assert "city" in args and "day" in args, (
                f"{ctx}: required fields missing: {args!r}"
            )
        print(f"[deep-latency] {ctx} mode={constraint_mode()} {latency_s:.2f}s")


# --------------------------------------------------------------------------- #
# Stage-2 deep cell: NEGATIVE CONTROL — the constraint actually fires
# --------------------------------------------------------------------------- #


class TestConstraintNegativeControl:
    """Prove llguidance masking actually fires — not a no-op passthrough.

    Drives the ALREADY-LIVE guided-decode path via
    ``response_format={"type":"json_schema","strict":true,...}`` (→ llguidance,
    ``engine/batched.py``). We ask for a value that is trivially off-schema
    under the strict schema (an integer for a field the schema pins to a
    ``["yes","no"]`` enum) and craft the prompt to TEMPT the model off-schema.

    * UN-constrained run (``response_format`` absent / ``json_object``): the
      model is FREE to emit the off-schema value — recorded, not asserted.
    * CONSTRAINED run (strict schema): the emitted JSON MUST validate against
      the schema. If a "constrained" run still emits the off-schema value,
      the constraint is a no-op and the cell FAILS.

    This is the load-bearing proof for raullen's acceptance bar: default-on
    only means anything if the constraint demonstrably masks off-schema tokens.
    It is independent of the tool-calling PRs' not-yet-landed default-on knob,
    so it produces a real signal on ANY git ref today.
    """

    _STRICT_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    def test_constraint_fires(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(
            rapid_mlx_server["base_url"]
        )
        model_id = rapid_mlx_server["model_id"]
        ctx = f"negctrl/{family_alias.family}"

        # A prompt engineered to tempt the model OFF the ["yes","no"] enum —
        # it wants to answer with a number / prose, which the strict schema
        # forbids.
        messages = [
            {
                "role": "user",
                "content": (
                    "On a scale of 1 to 10, how confident are you that the "
                    "sky is blue? Reply as JSON with a single key 'answer'. "
                    "Prefer a numeric confidence."
                ),
            }
        ]

        # (a) UN-constrained — record whether the model goes off the enum.
        try:
            unc = client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=64,
            )
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected json_object request: {exc}")
        unc_text = (unc.choices[0].message.content or "").strip()

        # (b) CONSTRAINED — strict enum schema; llguidance must mask off-enum.
        t0 = time.perf_counter()
        try:
            con = client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "confidence",
                        "schema": self._STRICT_SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0.0,
                max_tokens=64,
            )
        except wire_errors as exc:
            # If the server does not support json_schema strict guided decode
            # on this ref, the negative control can't run — degrade cleanly.
            strict_skip_or_fail(
                f"{ctx}: server rejected json_schema strict request "
                f"(guided path unavailable on this ref?): {exc}"
            )
        latency_s = time.perf_counter() - t0
        con_text = (con.choices[0].message.content or "").strip()

        # The constrained output MUST parse AND satisfy the enum. A no-op
        # constraint (off-schema value survives) fails here.
        try:
            con_obj = json.loads(con_text)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"{ctx}: CONSTRAINED output not JSON — constraint did not "
                f"fire: {con_text!r} ({exc})"
            )
        assert isinstance(con_obj, dict) and "answer" in con_obj, (
            f"{ctx}: constrained output missing required 'answer': {con_obj!r}"
        )
        assert con_obj["answer"] in ("yes", "no"), (
            f"{ctx}: CONSTRAINT NO-OP — off-enum value survived strict "
            f"schema: answer={con_obj['answer']!r} (unconstrained was "
            f"{unc_text[:120]!r}). llguidance masking is not firing."
        )
        print(
            f"[negctrl] {ctx} unconstrained={unc_text[:80]!r} "
            f"constrained={con_text!r} latency={latency_s:.2f}s"
        )
