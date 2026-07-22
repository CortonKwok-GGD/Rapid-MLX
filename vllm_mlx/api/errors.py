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
    """


def guided_schema_compile_error_detail(
    exc: BaseException,
    param: str = "response_format.json_schema.schema",
) -> dict[str, Any]:
    """Build the canonical OpenAI-shaped 400 envelope for a compile error.

    Shared by EVERY endpoint that constrains output to a caller schema (and
    the centralized exception handler) so the 400 body is byte-identical
    across ``/v1/chat/completions`` and ``/v1/responses``. ``param`` locates
    the offending field per surface: ``response_format.json_schema.schema``
    on the chat/completions API, ``text.format.schema`` on the responses API.

    The message embeds only the schema-level diagnostic carried by
    ``GuidedSchemaCompileError`` (which describes the CALLER's own malformed
    schema, confirmed by an independent validator) — never a server-internal
    exception, because the raise sites narrow to confirmed schema-invalid
    signals before constructing it.
    """
    return {
        "error": {
            "message": f"{param} failed to compile: {exc}",
            "type": "invalid_request_error",
            "code": "invalid_response_format_schema",
            "param": param,
        }
    }
