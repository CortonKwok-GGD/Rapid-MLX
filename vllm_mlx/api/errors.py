# SPDX-License-Identifier: Apache-2.0
"""Dependency-free error types + envelope builders for structured output.

This module deliberately imports NOTHING heavy (no ``mlx`` / ``llguidance`` /
``jsonschema``) so that lightweight consumers — the FastAPI exception handler
registered at app startup, route modules — can import ``GuidedSchemaCompileError``
and its 400-envelope builder WITHOUT triggering native MLX / llguidance module
initialization on apps that never touch guided decoding.

``vllm_mlx.api.guided`` re-exports both names for backward compatibility.
"""

from __future__ import annotations

from typing import Any

# Per-surface locator for the offending field in the 400 body. The chat/
# completions API nests the schema under ``response_format.json_schema.schema``;
# the responses API under ``text.format.schema``. The chat param is the DEFAULT
# because it is both the most common surface and the historical value.
CHAT_RESPONSE_FORMAT_PARAM = "response_format.json_schema.schema"
RESPONSES_TEXT_FORMAT_PARAM = "text.format.schema"


class GuidedSchemaCompileError(Exception):
    """A caller-supplied structured-output schema/grammar failed to compile.

    Distinct from a *generic* guided-generation failure (a transient runtime
    error, a missing ``[guided]`` extra, a slow/absent fast tokenizer, or a
    parse truncated by ``max_tokens``). Those degrade to ``None`` and, for
    best-effort/suggestion-only requests, MAY fall back to unconstrained
    generation.

    A compile error means the CLIENT's own
    ``response_format.json_schema.schema`` is invalid — a deterministic,
    request-level fault (e.g. ``{"type": "notatype"}``) confirmed by an
    independent JSON-Schema validator, NOT an operational llguidance failure
    on a structurally-valid schema (which stays on the runtime-failure /
    5xx path). The route layer translates it into an HTTP 400 (non-streaming)
    or a terminal SSE error envelope (streaming) instead of silently degrading
    to unconstrained output, so a caller who asked for a hard schema guarantee
    is never left believing their output is constrained when it is not.

    ``param`` carries the surface-correct field locator (see the module
    constants). It is raised surface-agnostic by the engine (``param=None``)
    and STAMPED by the route that owns the surface (via
    :func:`stamp_compile_error_param`) BEFORE the exception can be wrapped in a
    TaskGroup ``ExceptionGroup`` and land on the generic exception handler.
    Carrying it on the exception — rather than only passing it at the route's
    local ``except`` mapping — is what lets the generic-handler safety net
    render the right field name for ``/v1/responses`` on the escape path,
    instead of defaulting to the chat locator.
    """

    def __init__(self, message: str = "", param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


# ``BaseExceptionGroup`` is a builtin only on Python 3.11+. On 3.10 there are
# no ExceptionGroups to unwrap, so bind an empty tuple that makes every
# ``isinstance`` check against it False.
try:
    _BASE_EXCEPTION_GROUP: Any = BaseExceptionGroup
except NameError:  # pragma: no cover - Python 3.10
    _BASE_EXCEPTION_GROUP = ()


def _first_compile_error(exc: BaseException) -> GuidedSchemaCompileError | None:
    """Return the first ``GuidedSchemaCompileError`` reachable from ``exc``,
    recursing into (nested) ``BaseExceptionGroup`` members; else ``None``."""
    if isinstance(exc, GuidedSchemaCompileError):
        return exc
    if _BASE_EXCEPTION_GROUP and isinstance(exc, _BASE_EXCEPTION_GROUP):
        for sub in exc.exceptions:
            found = _first_compile_error(sub)
            if found is not None:
                return found
    return None


async def stamp_compile_error_param(coro: Any, param: str) -> Any:
    """Await ``coro``, stamping ``param`` onto any escaping compile error.

    The route knows its surface; the engine that RAISES the compile error does
    not. This awaits the engine coroutine as close to the raise site as the
    route can reach and tags the surface ``param`` on a ``GuidedSchemaCompileError``
    the moment it propagates — BEFORE any upstream ``TaskGroup`` could wrap it in
    an ``ExceptionGroup`` that would slip past the route's bare-``except`` and
    hit the generic handler with the default (chat) locator. Handles a compile
    error that is already ``ExceptionGroup``-wrapped (recursively), and only
    stamps when unset so an inner surface that already tagged it wins. Always
    re-raises — never swallows (so cancellation/timeout propagate unchanged).
    """
    try:
        return await coro
    except BaseException as exc:
        guided = _first_compile_error(exc)
        if guided is not None and getattr(guided, "param", None) is None:
            guided.param = param
        raise


def guided_schema_compile_error_detail(
    exc: BaseException,
    param: str | None = None,
) -> dict[str, Any]:
    """Build the canonical OpenAI-shaped 400 envelope for a compile error.

    Shared by EVERY endpoint that constrains output to a caller schema (and
    the centralized exception handler) so the 400 body is byte-identical
    across ``/v1/chat/completions`` and ``/v1/responses``. ``param`` locates
    the offending field per surface: ``response_format.json_schema.schema``
    on the chat/completions API, ``text.format.schema`` on the responses API.

    Resolution order for the field locator:
      1. an explicit ``param`` argument (a route's local mapping passes this),
      2. else ``exc.param`` stamped on the exception by the owning route
         (this is what makes the generic-handler safety net render the right
         surface for ``/v1/responses`` on the ExceptionGroup escape path),
      3. else the chat default.

    The message embeds only the schema-level diagnostic carried by
    ``GuidedSchemaCompileError`` (which describes the CALLER's own malformed
    schema, confirmed by an independent validator) — never a server-internal
    exception, because the raise sites narrow to confirmed schema-invalid
    signals before constructing it.
    """
    resolved = param
    if resolved is None:
        resolved = getattr(exc, "param", None)
    if resolved is None:
        resolved = CHAT_RESPONSE_FORMAT_PARAM
    return {
        "error": {
            "message": f"{resolved} failed to compile: {exc}",
            "type": "invalid_request_error",
            "code": "invalid_response_format_schema",
            "param": resolved,
        }
    }
