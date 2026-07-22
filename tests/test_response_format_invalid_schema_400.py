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
import os
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import from the DEPENDENCY-FREE errors module — and assert it is the SAME
# class the guided module re-exports, so the whole chain agrees on identity.
from vllm_mlx.api.errors import (
    GuidedSchemaCompileError,
    guided_schema_compile_error_detail,
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

_QWEN_TOK_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots"
)


def _load_real_fast_tokenizer():
    """Load the cached Qwen FAST tokenizer (no model weights, no download) and
    wrap it the way ``GuidedGenerator`` expects (``._tokenizer`` = the HF fast
    tokenizer). Returns ``None`` when unavailable so tests skip gracefully.
    """
    if not is_guided_available() or not os.path.isdir(_QWEN_TOK_CACHE):
        return None
    try:
        from transformers import AutoTokenizer

        snaps = [
            d
            for d in os.listdir(_QWEN_TOK_CACHE)
            if os.path.isdir(os.path.join(_QWEN_TOK_CACHE, d))
        ]
        if not snaps:
            return None
        hf = AutoTokenizer.from_pretrained(os.path.join(_QWEN_TOK_CACHE, snaps[0]))
        if not getattr(hf, "is_fast", False):
            return None
        return types.SimpleNamespace(
            _tokenizer=hf, bos_token_id=getattr(hf, "bos_token_id", None)
        )
    except Exception:
        return None


_REAL_TOK = _load_real_fast_tokenizer()


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
# Guided layer (only meaningful when the [guided] extra is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_raises_on_schema_compile_failure(monkeypatch):
    """``GuidedGenerator.generate_json`` must RAISE ``GuidedSchemaCompileError``
    (not swallow to ``None``) when the schema fails to compile, so the engine
    can distinguish an invalid schema from a benign guided-unavailable None.

    We monkeypatch ``grammar_from_json_schema`` to raise ``ValueError`` (the
    type llguidance raises for an unparseable schema) so the test needs no
    real model/tokenizer — it pins the ``generate_json`` compile-guard, the
    contract the whole 400 path depends on.
    """
    from vllm_mlx.api import guided as guided_mod

    def _boom(*_a, **_k):
        raise ValueError("expected ident at line 1 column 2")

    monkeypatch.setattr(
        guided_mod.LLMatcher, "grammar_from_json_schema", staticmethod(_boom)
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    with pytest.raises(GuidedSchemaCompileError):
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_internal_error_on_valid_schema_degrades_to_none(monkeypatch):
    """Discriminator, operational arm: an INTERNAL / operational failure (e.g.
    a ``RuntimeError`` resource failure) on the compile call for a schema a
    standard validator ACCEPTS must NOT be misclassified as invalid client
    input. The discriminator (``_schema_invalid_reason`` = ``None`` for a valid
    schema) sends it to the runtime-failure path — ``generate_json`` returns
    ``None`` (best-effort fallback / strict 502), NEVER a 400 that would tell
    the caller to fix a perfectly valid schema and leak a server-internal
    diagnostic into the body.
    """
    from vllm_mlx.api import guided as guided_mod

    def _internal_boom(*_a, **_k):
        raise RuntimeError("internal resource failure / OOM")

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(_internal_boom),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    # _SCHEMA is valid → schema_invalid_reason is None → the RuntimeError is an
    # operational failure caught by the broad `except Exception -> None` arm.
    assert gen.generate_json(prompt="hi", json_schema=_SCHEMA, max_tokens=8) is None


@pytest.mark.skipif(
    not is_guided_available(), reason="requires the [guided] (llguidance) extra"
)
def test_generate_json_valueerror_on_invalid_schema_raises_400(monkeypatch):
    """Discriminator, client-fault arm: an EAGER ``ValueError`` from
    ``grammar_from_json_schema`` on a schema a standard validator REJECTS is a
    caller fault → ``GuidedSchemaCompileError`` (→ 400). This pins the eager
    ``except ValueError`` branch that mirrors the lazy ``get_error()`` one."""
    from vllm_mlx.api import guided as guided_mod

    def _unparseable(*_a, **_k):
        raise ValueError("expected ident at line 1 column 2")

    monkeypatch.setattr(
        guided_mod.LLMatcher, "grammar_from_json_schema", staticmethod(_unparseable)
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    # _INVALID_SCHEMA is invalid → schema_invalid_reason is non-None → 400.
    with pytest.raises(GuidedSchemaCompileError):
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)


@pytest.mark.skipif(
    _REAL_TOK is None,
    reason="requires the [guided] extra AND the cached Qwen fast tokenizer",
)
def test_generate_json_real_llguidance_get_error_raises_400():
    """Round-3 #2 — drive the REAL, load-bearing ``matcher.get_error()`` path
    with NO mock at the grammar-compile layer.

    ``{"type": "notatype"}`` is a schema llguidance accepts EAGERLY at
    ``grammar_from_json_schema`` but rejects LAZILY at ``LLMatcher`` construction
    (``get_error()`` → "Invalid type: notatype"). The route-level mocks stand in
    a ``GuidedSchemaCompileError`` at ``generate_with_schema``; this test proves
    the real llguidance stack actually PRODUCES that exception, so the whole
    400 contract rests on observed behaviour, not an assumed one.

    Uses the cached Qwen FAST tokenizer (no weights, no download); ``model`` is
    an ``object()`` because the raise happens at matcher construction, before
    any forward pass. The jsonschema discriminator confirms the schema invalid,
    so ``generate_json`` re-raises rather than degrading to ``None``.
    """
    from vllm_mlx.api.guided import GuidedGenerator

    gen = GuidedGenerator(model=object(), tokenizer=_REAL_TOK)
    with pytest.raises(GuidedSchemaCompileError) as excinfo:
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)
    assert "notatype" in str(excinfo.value)


@pytest.mark.skipif(
    _REAL_TOK is None,
    reason="requires the [guided] extra AND the cached Qwen fast tokenizer",
)
def test_generate_json_real_llguidance_get_error_on_valid_schema_degrades(monkeypatch):
    """Round-3 discriminator, real-stack operational arm: if the REAL
    ``matcher.get_error()`` fires (via a monkeypatched ``_decode_constrained``
    that re-raises ``GuidedSchemaCompileError``) on a schema a standard
    validator ACCEPTS, ``generate_json`` must degrade to ``None`` (operational),
    NOT raise a 400. This guards the exact false-positive the discriminator was
    added to prevent: an llguidance-unsupported-but-valid construct must not be
    reported to the caller as a malformed schema."""
    from vllm_mlx.api.guided import GuidedGenerator

    gen = GuidedGenerator(model=object(), tokenizer=_REAL_TOK)

    def _fake_decode(**_kw):
        # Simulate llguidance rejecting a structurally-valid schema at
        # matcher construction (get_error) — the operational case.
        raise GuidedSchemaCompileError("some llguidance-internal limit")

    monkeypatch.setattr(gen, "_decode_constrained", _fake_decode)
    # _SCHEMA is valid → discriminator None → operational → None, not a raise.
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
    ``_extract_from_group`` and mapped to the clean 400, NOT a sanitized 500."""
    group = _exc_group(GuidedSchemaCompileError("Invalid type: notatype"))
    client = _group_raising_app(group)
    resp = client.get("/boom")
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert "notatype" in err["message"]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
