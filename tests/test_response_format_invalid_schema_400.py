# SPDX-License-Identifier: Apache-2.0
"""0.10.16 dogfood P1-③ — an invalid ``response_format`` schema must NOT
silently degrade to unconstrained generation with an HTTP 200.

Repro (pre-fix): a request with
``response_format={"type":"json_schema","json_schema":{"name":"x",
"schema":{"type":"notatype"}}}`` returned **HTTP 200 with non-conforming
free-form text**. The server logged the llguidance compile error
(``Invalid type: notatype``) and then ``WARNING: Guided generation failed,
falling back to regular generation`` — but the caller received ZERO signal
that the constraint had been dropped. That is worse than a 400: the client
believes its output is schema-constrained when it is not.

Root cause: ``GuidedGenerator._decode_constrained`` returned ``None`` on a
grammar-compile error, indistinguishable from the benign
guided-unavailable / truncated-parse ``None``. ``BatchedEngine.
generate_with_schema`` swallowed that ``None`` into a silent
``self.chat(...)`` fallback (HTTP 200, unconstrained).

Fix: the guided layer now raises ``GuidedSchemaCompileError`` for a
caller-schema compile failure. The batched engine propagates it instead
of falling back, and the chat route maps it to a clean HTTP 400
(non-streaming) or a terminal ``invalid_request_error`` SSE envelope
(streaming) — in BOTH strict and non-strict mode, because an invalid
schema is malformed input regardless of the strictness flag.

These tests pin the route-level contract with a mock engine (no model /
llguidance needed). A ``GuidedSchemaCompileError``-raising engine stands in
for a real invalid-schema compile failure; a generic ``RuntimeError`` stands
in for a transient guided-decode failure whose best-effort fallback must be
PRESERVED. One additional test exercises the guided layer directly when the
``[guided]`` extra is installed.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api.guided import GuidedSchemaCompileError, is_guided_available
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.chat import router as chat_router

_VALID_PAYLOAD = '{"answer": 42}'
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_INVALID_SCHEMA = {"type": "notatype"}
_COMPILE_ERR = GuidedSchemaCompileError("Invalid type: notatype")


class _Engine:
    """Mock engine.

    ``guided_raises`` selects what ``generate_with_schema`` does:
      * ``None``                    → return the fixed valid payload,
      * a ``GuidedSchemaCompileError`` → simulate an invalid caller schema,
      * any other exception         → simulate a transient guided failure.
    """

    preserve_native_tool_format = False
    is_mllm = False
    tokenizer = None

    def __init__(
        self,
        *,
        supports_guided: bool = True,
        guided_text: str = _VALID_PAYLOAD,
        chat_text: str = "FALLBACK unconstrained text",
        guided_raises: Exception | None = None,
    ):
        self.supports_guided_generation = supports_guided
        self._guided_text = guided_text
        self._chat_text = chat_text
        self._guided_raises = guided_raises
        self.guided_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append({"json_schema": json_schema, "kwargs": kwargs})
        if self._guided_raises is not None:
            raise self._guided_raises
        return GenerationOutput(
            text=self._guided_text,
            new_text=self._guided_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._chat_text,
            new_text=self._chat_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        yield GenerationOutput(
            text=self._chat_text,
            new_text=self._chat_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_client(engine: _Engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app)


def _response_format(schema: dict, *, strict: bool | None = None) -> dict:
    js: dict = {"name": "x", "schema": schema}
    if strict is not None:
        js["strict"] = strict
    return {"type": "json_schema", "json_schema": js}


def _parse_sse_events(text: str) -> tuple[list[dict], bool]:
    events: list[dict] = []
    saw_done = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events, saw_done


def _assert_invalid_schema_400(resp) -> None:
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert err["param"] == "response_format.json_schema.schema"
    assert "failed to compile" in err["message"]
    assert "notatype" in err["message"]


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def test_sync_invalid_schema_returns_400_not_silent_200():
    """The core P1-③ repro: invalid schema → 400, NOT a silent 200 with
    unconstrained text. The unconstrained ``chat`` fallback must NOT run."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA),
        },
    )
    _assert_invalid_schema_400(resp)
    assert engine.guided_calls, "guided path must have been attempted"
    assert engine.chat_calls == [], (
        "invalid schema must NOT silently fall back to unconstrained chat"
    )


def test_sync_invalid_schema_strict_also_returns_400_not_502():
    """Under ``strict=true`` a grammar-compile failure surfacing at
    generation time is a 400 (malformed request), NOT the strict-mode 502
    ``strict_schema_violation`` — the schema itself is the fault, not a
    server-side soundness breach.

    The request carries a valid-SHAPED object schema (so it clears the
    earlier strict ``check_schema_validity`` pre-flight), but the engine
    reports a ``GuidedSchemaCompileError`` at compile time — the case of an
    llguidance-unsupported construct the shape check accepts. This must map
    to my 400, not fall through to the strict 502 arm.
    """
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA, strict=True),
        },
    )
    _assert_invalid_schema_400(resp)
    assert engine.chat_calls == []


def test_sync_valid_schema_still_returns_200():
    """Regression guard: a VALID schema must still return 200 with the
    constrained content — the fix must not turn valid schemas into 400s."""
    engine = _Engine(guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == _VALID_PAYLOAD
    assert len(engine.guided_calls) == 1
    assert engine.chat_calls == []


def test_sync_plain_chat_without_response_format_still_works():
    """A plain request with no ``response_format`` must be unaffected — it
    routes through ``chat`` and returns 200, never touching guided."""
    engine = _Engine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.guided_calls == []
    assert len(engine.chat_calls) == 1


def test_sync_generic_guided_failure_still_falls_back_non_strict():
    """A GENERIC (non-compile) guided failure under ``strict=false`` must
    PRESERVE today's best-effort fallback to unconstrained ``chat`` (200).
    Only a ``GuidedSchemaCompileError`` (invalid schema) is a hard 400 —
    this pins that the fix narrowly special-cases compile errors and does
    not turn every transient guided hiccup into a 400."""
    engine = _Engine(guided_raises=RuntimeError("transient llguidance blip"))
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA, strict=False),
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(engine.guided_calls) == 1
    assert len(engine.chat_calls) == 1, (
        "a transient (non-compile) guided failure must keep its best-effort "
        "unconstrained fallback"
    )


# ---------------------------------------------------------------------------
# Streaming (SSE — the 200 status is already committed, so a compile error
# surfaces as a terminal error envelope, NEVER a silent unconstrained fallback)
# ---------------------------------------------------------------------------


def _assert_stream_compile_error(resp, engine: _Engine) -> None:
    assert resp.status_code == 200, resp.text  # SSE always opens 200
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done, "stream must terminate with [DONE]"
    err_events = [e for e in events if e.get("error")]
    assert len(err_events) == 1, f"expected exactly one error envelope; got {events}"
    err = err_events[0]["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert err["param"] == "response_format.json_schema.schema"
    assert "failed to compile" in err["message"]
    assert engine.stream_calls == [], (
        "invalid schema must NOT silently fall back to unconstrained streaming"
    )


def test_stream_invalid_schema_emits_error_envelope_no_fallback():
    """Streaming, ``strict=false``: an invalid schema must emit an
    ``invalid_request_error`` SSE envelope + DONE, NOT silently fall back to
    the unconstrained ``stream_chat`` helper."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA, strict=False),
        },
    )
    _assert_stream_compile_error(resp, engine)


def test_stream_invalid_schema_strict_emits_error_envelope():
    """Streaming, ``strict=true``: a compile failure surfacing at generation
    time emits the terminal ``invalid_request_error`` envelope (NOT a silent
    unconstrained-streaming fallback, NOT the strict 502 arm).

    Uses a valid-SHAPED schema (clears the strict ``check_schema_validity``
    pre-flight, which would otherwise 400 a malformed-shape schema cleanly
    before the SSE stream opens) with the engine reporting a compile error —
    the llguidance-unsupported-construct case.
    """
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA, strict=True),
        },
    )
    _assert_stream_compile_error(resp, engine)


# ---------------------------------------------------------------------------
# Guided layer (only meaningful when the [guided] extra is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_raises_on_schema_compile_failure(monkeypatch):
    """``GuidedGenerator.generate_json`` must RAISE ``GuidedSchemaCompileError``
    (not swallow to ``None``) when the schema fails to compile, so the engine
    can distinguish an invalid schema from a benign guided-unavailable None.

    We monkeypatch ``grammar_from_json_schema`` to raise so the test needs no
    real model/tokenizer — it pins the ``generate_json`` compile-guard, the
    contract the whole 400 path depends on.
    """
    from vllm_mlx.api import guided as guided_mod

    def _boom(*_a, **_k):
        raise ValueError("Invalid type: notatype")

    monkeypatch.setattr(
        guided_mod.LLMatcher, "grammar_from_json_schema", staticmethod(_boom)
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    with pytest.raises(GuidedSchemaCompileError):
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
