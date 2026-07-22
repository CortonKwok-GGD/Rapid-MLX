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

# Import from the DEPENDENCY-FREE errors module — and assert it is the SAME
# class the guided module re-exports, so the whole chain agrees on identity.
from vllm_mlx.api.errors import (
    CHAT_RESPONSE_FORMAT_PARAM,
    RESPONSES_TEXT_FORMAT_PARAM,
    GuidedSchemaCompileError,
    guided_schema_compile_error_detail,
    stamp_compile_error_param,
)
from vllm_mlx.api.guided import GuidedSchemaCompileError as _GuidedErrFromGuided
from vllm_mlx.api.guided import is_guided_available
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.chat import router as chat_router

assert _GuidedErrFromGuided is GuidedSchemaCompileError  # re-export identity

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
# Dependency-free errors module (Round-3 NIT #4): the envelope builder lives in
# ``vllm_mlx.api.errors`` with NO mlx/llguidance import, so the exception
# handlers and routes can build the 400 body without triggering engine init.
# ---------------------------------------------------------------------------


def test_errors_module_is_dependency_free():
    """``vllm_mlx.api.errors`` must not drag in the heavy engine stack — that
    is the whole point of splitting it out of ``api.guided`` (which imports
    mlx/llguidance). If either becomes importable-through-errors, importing the
    module for the envelope builder would boot the engine on every handler.

    Runs in a FRESH subprocess: an in-process ``importlib.reload`` would mint a
    NEW ``GuidedSchemaCompileError`` class object and poison the ``isinstance``
    identity the rest of this suite (and the live handlers) depend on. A clean
    interpreter both avoids that and proves the import graph from a cold start,
    which is what a real handler process sees.
    """
    import subprocess
    import sys

    code = (
        "import sys; import vllm_mlx.api.errors as e;"
        "assert hasattr(e, 'GuidedSchemaCompileError');"
        "assert hasattr(e, 'guided_schema_compile_error_detail');"
        "assert 'mlx.core' not in sys.modules, sorted(m for m in sys.modules "
        "if m.startswith('mlx'));"
        "assert 'llguidance' not in sys.modules;"
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK", proc.stdout


def test_guided_compile_error_detail_envelope_and_param_override():
    """The envelope builder produces the canonical OpenAI-style body and honors
    the ``param`` override the ``/v1/responses`` route uses
    (``text.format.schema`` vs the chat default)."""
    exc = GuidedSchemaCompileError("Invalid type: notatype")

    default = guided_schema_compile_error_detail(exc)["error"]
    assert default["type"] == "invalid_request_error"
    assert default["code"] == "invalid_response_format_schema"
    assert default["param"] == "response_format.json_schema.schema"
    assert "failed to compile" in default["message"]
    assert "notatype" in default["message"]

    overridden = guided_schema_compile_error_detail(exc, param="text.format.schema")[
        "error"
    ]
    assert overridden["param"] == "text.format.schema"


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
# Guided layer. llguidance is a CORE dependency (promoted out of the [guided]
# extra in 0.10.15), so ``is_guided_available()`` is True in CI and these run.
# They depend on NO developer-local HF tokenizer cache — every llguidance
# interaction is stubbed deterministically, so a regression (e.g. reverting
# _decode_constrained's raise to ``return None``) fails the suite in clean CI
# rather than silently skipping.
# ---------------------------------------------------------------------------


class _StubMatcher:
    """Minimal stand-in for ``llguidance.mlx.LLMatcher``. ``get_error()``
    returns the configured string, driving ``_decode_constrained``'s
    grammar-rejection branch without any real grammar/tokenizer/model."""

    def __init__(self, error: str):
        self._error = error

    def get_error(self) -> str:
        return self._error


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_decode_constrained_raises_on_matcher_get_error(monkeypatch):
    """Round-4 #2 — DETERMINISTIC coverage of the load-bearing
    ``matcher.get_error()`` raise, with NO HF-cache dependency.

    ``_decode_constrained`` builds an ``LLMatcher`` and, when ``get_error()``
    is non-empty, MUST raise ``GuidedSchemaCompileError`` (not ``return None``,
    which is indistinguishable from a benign guided-unavailable None and let the
    original silent unconstrained 200 through). We stub ``LLMatcher`` so its
    ``get_error()`` returns an error string and stub ``_get_lltokenizer`` so the
    method reaches the matcher check; the raise happens before any real
    tokenizer/model use, so ``model``/``tokenizer`` are bare objects.

    Reverting the raise to ``return None`` turns this test red in clean CI —
    which the previous HF-cache-gated test could not do.
    """
    from vllm_mlx.api import guided as guided_mod

    monkeypatch.setattr(
        guided_mod, "LLMatcher", lambda _lltok, _grammar: _StubMatcher("Invalid type")
    )
    gen = guided_mod.GuidedGenerator(model=object(), tokenizer=object())
    monkeypatch.setattr(gen, "_get_lltokenizer", lambda: object())
    with pytest.raises(GuidedSchemaCompileError) as excinfo:
        gen._decode_constrained(
            grammar="<grammar>", prompt="hi", max_tokens=8, temperature=0.0
        )
    assert "Invalid type" in str(excinfo.value)


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_upfront_invalid_schema_raises_before_llguidance(monkeypatch):
    """Round-4 #1 (the core hole) — a structurally-invalid schema is rejected
    UP FRONT, before llguidance is invoked at all, so a tolerant/unavailable
    compiler cannot let it slip to a silent unconstrained 200.

    We monkeypatch ``grammar_from_json_schema`` to FAIL THE TEST if called: the
    up-front ``_schema_invalid_reason`` check must raise
    ``GuidedSchemaCompileError`` before reaching it. This is the deterministic
    proof that the fix does not depend on llguidance raising."""
    from vllm_mlx.api import guided as guided_mod

    def _must_not_be_called(*_a, **_k):
        raise AssertionError(
            "llguidance was invoked on a validator-invalid schema — up-front "
            "validation must reject it first"
        )

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(_must_not_be_called),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    with pytest.raises(GuidedSchemaCompileError) as excinfo:
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)
    # The up-front raise carries the validator reason (mentions the bad type)
    # and NO surface param yet (the route stamps it).
    assert "notatype" in str(excinfo.value)
    assert excinfo.value.param is None


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_tolerant_compiler_on_invalid_schema_still_400(monkeypatch):
    """Round-4 #1 — the tolerant-compiler variant the fix specifically closes:
    even if llguidance would SUCCEED (return a grammar) on a validator-invalid
    schema, the up-front check must still raise → 400. We monkeypatch
    ``grammar_from_json_schema`` to return a dummy grammar and
    ``_decode_constrained`` to a plausible completion; the up-front reject means
    neither is ever reached, so the invalid schema can never silently produce a
    200."""
    from vllm_mlx.api import guided as guided_mod

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(lambda *_a, **_k: "<tolerant-grammar>"),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    monkeypatch.setattr(gen, "_decode_constrained", lambda **_k: '{"whatever": 1}')
    with pytest.raises(GuidedSchemaCompileError):
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_valid_schema_get_error_secondary_degrades_to_none(monkeypatch):
    """Round-4 #1 secondary net — a schema the validator ACCEPTS that
    llguidance nonetheless rejects at ``get_error()`` (an unsupported-but-valid
    construct) is OPERATIONAL: ``generate_json`` degrades to ``None`` (→
    best-effort fallback / strict 502), NOT a misleading 400.

    Deterministic: ``grammar_from_json_schema`` succeeds and
    ``_decode_constrained`` raises ``GuidedSchemaCompileError`` (the get_error()
    signal); with a VALID ``_SCHEMA`` the discriminator says None → the arm
    swallows to None."""
    from vllm_mlx.api import guided as guided_mod

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(lambda *_a, **_k: "<grammar>"),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)

    def _decode_raises(**_k):
        raise GuidedSchemaCompileError("llguidance-internal limit on a valid schema")

    monkeypatch.setattr(gen, "_decode_constrained", _decode_raises)
    assert gen.generate_json(prompt="hi", json_schema=_SCHEMA, max_tokens=8) is None


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_valid_schema_operational_error_degrades_to_none(monkeypatch):
    """Secondary net, operational arm: an INTERNAL failure (e.g. a
    ``RuntimeError`` / eager ``ValueError``) on a schema the validator ACCEPTS
    must NOT be misclassified as invalid client input — it degrades to ``None``,
    never a 400 that would leak a server-internal diagnostic and tell the caller
    to fix a perfectly valid schema."""
    from vllm_mlx.api import guided as guided_mod

    def _internal_boom(*_a, **_k):
        raise RuntimeError("internal resource failure / OOM")

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(_internal_boom),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    assert gen.generate_json(prompt="hi", json_schema=_SCHEMA, max_tokens=8) is None


# ---------------------------------------------------------------------------
# Engine path (Issue 3): the REAL BatchedEngine must PROPAGATE a compile error
# from the guided layer, NOT swallow it into a silent chat() fallback. The
# mock is injected at the GuidedGenerator layer (not at generate_with_schema),
# so these tests exercise BatchedEngine._run_guided_generation /
# generate_with_schema — deleting the engine-layer propagation would turn them
# red while the route-level mock tests above stayed green.
# ---------------------------------------------------------------------------


def _bare_engine(monkeypatch, gen_exc: Exception):
    """A minimally-wired REAL ``BatchedEngine`` whose ``GuidedGenerator``
    raises ``gen_exc`` from ``generate_json``. Constructed via ``__new__`` so
    no model load happens; only the attributes the guided path reads are set.
    """
    from vllm_mlx.engine import batched as batched_mod

    class _FakeGuidedGenerator:
        def __init__(self, model, tokenizer):
            pass

        def generate_json(self, **_kw):
            raise gen_exc

    monkeypatch.setattr(batched_mod, "GuidedGenerator", _FakeGuidedGenerator)
    eng = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    eng._model = object()
    eng._tokenizer = object()
    eng._is_mllm = False
    return eng, batched_mod


def test_run_guided_generation_propagates_compile_error(monkeypatch):
    """The sync guided worker re-raises ``GuidedSchemaCompileError`` instead
    of returning ``None`` — the load-bearing line at batched.py that stops the
    silent fallback."""
    eng, _ = _bare_engine(
        monkeypatch, GuidedSchemaCompileError("Invalid type: notatype")
    )
    with pytest.raises(GuidedSchemaCompileError):
        eng._run_guided_generation(
            prompt="p", json_schema=_INVALID_SCHEMA, max_tokens=8, temperature=0.0
        )


def test_run_guided_generation_generic_failure_returns_none(monkeypatch):
    """A transient (non-compile) guided failure stays a graceful ``None`` so
    the best-effort fallback is preserved — only compile errors propagate."""
    eng, _ = _bare_engine(monkeypatch, RuntimeError("transient blip"))
    assert (
        eng._run_guided_generation(
            prompt="p", json_schema=_SCHEMA, max_tokens=8, temperature=0.0
        )
        is None
    )


async def test_generate_with_schema_propagates_compile_error_no_chat_fallback(
    monkeypatch,
):
    """End-to-end through the REAL async ``generate_with_schema`` wrapper: a
    compile error must propagate to the caller (which the route maps to 400)
    and MUST NOT silently fall back to unconstrained ``chat()``. Deleting the
    engine-layer propagation would make this fail — ``chat()`` would run and
    no exception would surface (the exact production silent-degrade)."""
    from concurrent.futures import ThreadPoolExecutor

    eng, batched_mod = _bare_engine(
        monkeypatch, GuidedSchemaCompileError("Invalid type: notatype")
    )
    monkeypatch.setattr(
        batched_mod, "shared_apply_chat_template", lambda *a, **k: "PROMPT"
    )
    eng._loaded = True
    eng._model_name = "test-model"
    chat_calls: list = []

    async def _spy_chat(**kwargs):
        chat_calls.append(kwargs)
        return GenerationOutput(
            text="FALLBACK", new_text="FALLBACK", finish_reason="stop"
        )

    eng.chat = _spy_chat
    executor = ThreadPoolExecutor(max_workers=1)
    eng._model_load_executor = executor
    try:
        with pytest.raises(GuidedSchemaCompileError):
            await eng.generate_with_schema(
                messages=[{"role": "user", "content": "hi"}],
                json_schema=_INVALID_SCHEMA,
                # raise_on_failure=False is the non-streaming default that,
                # pre-fix, silently fell back to chat() on a None result.
                raise_on_failure=False,
            )
        assert chat_calls == [], "compile error must NOT trigger a chat() fallback"
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Non-chat endpoint coverage (Issue 1): /v1/responses must ALSO surface a
# compile error as HTTP 400 (not the strict 502, not a silent 200 / 500).
# ---------------------------------------------------------------------------


@pytest.fixture
def _rate_limiter_state():
    """Save/restore the global rate-limiter so disabling it for the responses
    route does not leak into other tests."""
    from vllm_mlx.middleware.auth import rate_limiter

    saved_enabled = rate_limiter.enabled
    saved_rpm = rate_limiter.requests_per_minute
    saved_requests = dict(rate_limiter._requests)
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()
    yield rate_limiter
    rate_limiter.enabled = saved_enabled
    rate_limiter.requests_per_minute = saved_rpm
    rate_limiter._requests.clear()
    rate_limiter._requests.update(saved_requests)


def _make_responses_client(engine: _Engine) -> TestClient:
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    return TestClient(app)


def test_responses_invalid_schema_returns_400(_rate_limiter_state):
    """/v1/responses strict + a schema that fails to compile at guided time →
    HTTP 400 (param ``text.format.schema``), NOT the strict 502 and NOT a
    silent 200. Uses a valid-SHAPED schema so it clears the strict
    ``check_schema_validity`` pre-flight and reaches ``generate_with_schema``,
    where the (mock) engine reports the compile failure."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x",
                    "schema": _SCHEMA,
                    "strict": True,
                }
            },
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert err["param"] == "text.format.schema"
    assert "failed to compile" in err["message"]
    assert engine.chat_calls == [], "compile error must NOT fall back to chat()"


def test_responses_strict_stream_rejected_before_generation(_rate_limiter_state):
    """Round-3 #3 pin: ``/v1/responses`` NEVER has the chat streaming's silent
    gap because a strict schema with ``stream=true`` is rejected UP-FRONT with
    a 400 (``strict_stream_unsupported``) — constrained decoding on this surface
    is buffered-only, so ``_stream_responses`` never calls
    ``generate_with_schema`` and no compile error can surface mid-SSE. This is
    the structural reason the compile-error 400 for ``/v1/responses`` lives only
    on the non-stream path (``test_responses_invalid_schema_returns_400``).

    Guarding this here means a future change that lets strict schemas stream on
    ``/v1/responses`` (re-opening the silent-degrade hole) turns this test red.
    ``guided_raises`` is set so that IF generation were (wrongly) reached, the
    engine would blow up — but it must not be reached at all.
    """
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "stream": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x",
                    "schema": _SCHEMA,
                    "strict": True,
                }
            },
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "strict_stream_unsupported"
    assert engine.guided_calls == [], (
        "strict+stream must be rejected before any generation is attempted"
    )
    assert engine.chat_calls == []


# ---------------------------------------------------------------------------
# Round-3 #5: the TaskGroup / thread-boundary safety net must recover a
# GuidedSchemaCompileError even when a 3.11+ ``TaskGroup`` re-raises it wrapped
# in a (possibly nested) ``ExceptionGroup``, which a bare ``isinstance`` check
# in the dedicated handler would miss — it would fall through to a sanitized
# 500 and re-hide the caller's invalid schema behind an opaque server error.
# ---------------------------------------------------------------------------


def _exc_group(*excs: BaseException) -> BaseException:
    """Build a real ``ExceptionGroup`` on 3.11+, else skip the group tests
    (3.10 has no builtin ``ExceptionGroup``; the safety net degrades to a plain
    ``isinstance`` there, which is correct because TaskGroups can't wrap on
    3.10)."""
    try:
        return ExceptionGroup("boom", list(excs))  # noqa: F821 (3.11+ builtin)
    except NameError:  # pragma: no cover - 3.10 path
        pytest.skip("ExceptionGroup requires Python 3.11+")


def _group_raising_app(exc: BaseException) -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    async def _boom():
        raise exc

    # raise_server_exceptions=False: the generic ``Exception`` handler
    # (ServerErrorMiddleware) always re-raises after sending its response, so
    # to OBSERVE the 400 it produced we must stop TestClient from propagating
    # that re-raise into the test body.
    return TestClient(app, raise_server_exceptions=False)


def test_exception_group_wrapped_compile_error_maps_to_400():
    """A ``GuidedSchemaCompileError`` wrapped in an ``ExceptionGroup`` (the
    3.11+ TaskGroup shape) reaching the generic handler is unwrapped via
    ``_extract_from_group`` and mapped to the clean 400, NOT a sanitized 500.

    An UNSTAMPED error (``param is None``) renders the chat default locator."""
    group = _exc_group(GuidedSchemaCompileError("Invalid type: notatype"))
    client = _group_raising_app(group)
    resp = client.get("/boom")
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert "notatype" in err["message"]
    assert err["param"] == CHAT_RESPONSE_FORMAT_PARAM


def test_nested_exception_group_wrapped_compile_error_maps_to_400():
    """The unwrap is RECURSIVE: a compile error nested two ``ExceptionGroup``
    levels deep (group-of-group, e.g. nested TaskGroups) is still recovered to
    the 400 rather than leaking as a 500."""
    inner = _exc_group(GuidedSchemaCompileError("Invalid type: notatype"))
    outer = _exc_group(inner)
    client = _group_raising_app(outer)
    resp = client.get("/boom")
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["code"] == "invalid_response_format_schema"


def test_exception_group_without_compile_error_stays_500():
    """A safety guard: an ``ExceptionGroup`` that does NOT contain a
    ``GuidedSchemaCompileError`` must NOT be coerced into a 400 — it stays a
    sanitized 500. This pins that ``_extract_from_group`` returns ``None`` (no
    false-positive 400s) for unrelated grouped failures."""
    group = _exc_group(RuntimeError("some unrelated internal failure"))
    client = _group_raising_app(group)
    resp = client.get("/boom")
    assert resp.status_code == 500, resp.text
    # The sanitized 500 body carries no ``code`` — the point is only that the
    # unrelated group was NOT coerced into the compile-error 400 envelope.
    assert resp.json().get("error", {}).get("code") != "invalid_response_format_schema"


# ---------------------------------------------------------------------------
# Round-4 #3: the surface-correct ``param`` is carried ON the exception and the
# handler reads ``exc.param``, so /v1/responses renders ``text.format.schema``
# (not the chat default) even on the ExceptionGroup escape path that bypasses
# the route's local ``except`` and lands on the generic handler.
# ---------------------------------------------------------------------------


def test_guided_compile_error_carries_param():
    """The exception carries ``param``; the default (unset) is ``None`` so the
    envelope builder falls back to the chat locator, and an explicit value is
    preserved for the handler to read."""
    assert GuidedSchemaCompileError("boom").param is None
    tagged = GuidedSchemaCompileError("boom", param=RESPONSES_TEXT_FORMAT_PARAM)
    assert tagged.param == RESPONSES_TEXT_FORMAT_PARAM
    # Builder with no explicit param reads exc.param.
    assert (
        guided_schema_compile_error_detail(tagged)["error"]["param"]
        == RESPONSES_TEXT_FORMAT_PARAM
    )
    # Explicit param argument still wins over the stamped one.
    assert (
        guided_schema_compile_error_detail(tagged, param="override.path")["error"][
            "param"
        ]
        == "override.path"
    )


async def test_stamp_compile_error_param_tags_bare_and_grouped():
    """``stamp_compile_error_param`` (what the responses route wraps its engine
    coroutine with) tags the surface param on an escaping compile error —
    whether it propagates bare or already ``ExceptionGroup``-wrapped — and never
    swallows. It also leaves an already-tagged inner error untouched."""

    async def _raise_bare():
        raise GuidedSchemaCompileError("boom")

    with pytest.raises(GuidedSchemaCompileError) as ei:
        await stamp_compile_error_param(_raise_bare(), RESPONSES_TEXT_FORMAT_PARAM)
    assert ei.value.param == RESPONSES_TEXT_FORMAT_PARAM

    async def _raise_grouped():
        raise _exc_group(GuidedSchemaCompileError("boom"))

    with pytest.raises(BaseException) as ei2:
        await stamp_compile_error_param(_raise_grouped(), RESPONSES_TEXT_FORMAT_PARAM)
    from vllm_mlx.api.errors import _first_compile_error

    inner = _first_compile_error(ei2.value)
    assert inner is not None and inner.param == RESPONSES_TEXT_FORMAT_PARAM

    async def _raise_already_tagged():
        raise GuidedSchemaCompileError("boom", param="already.set")

    with pytest.raises(GuidedSchemaCompileError) as ei3:
        await stamp_compile_error_param(
            _raise_already_tagged(), RESPONSES_TEXT_FORMAT_PARAM
        )
    assert ei3.value.param == "already.set"

    # A non-compile error passes straight through, unmodified, never swallowed.
    async def _raise_other():
        raise RuntimeError("unrelated")

    with pytest.raises(RuntimeError):
        await stamp_compile_error_param(_raise_other(), RESPONSES_TEXT_FORMAT_PARAM)


def test_responses_surface_exception_group_renders_text_format_param():
    """End-to-end composition of the /v1/responses escape path: the route wraps
    its engine coroutine with ``stamp_compile_error_param(..., text.format.schema)``;
    an upstream TaskGroup then wraps the (now-stamped) error in an
    ``ExceptionGroup`` that bypasses the route's bare ``except`` and reaches the
    generic handler. The handler must render ``text.format.schema`` — NOT the
    chat default — because the param rode along on the exception."""
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    async def _boom():
        async def _engine_coro():
            # The engine raises surface-agnostic (param=None).
            raise GuidedSchemaCompileError("Invalid type: notatype")

        try:
            # Exactly what routes/responses.py wraps generate_with_schema with.
            await stamp_compile_error_param(_engine_coro(), RESPONSES_TEXT_FORMAT_PARAM)
        except GuidedSchemaCompileError as e:
            # Simulate an upstream 3.11+ TaskGroup wrapping the stamped error,
            # so it escapes the local bare-except and hits the generic handler.
            raise _exc_group(e) from None

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["code"] == "invalid_response_format_schema"
    assert err["param"] == RESPONSES_TEXT_FORMAT_PARAM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
