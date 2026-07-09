# SPDX-License-Identifier: Apache-2.0
"""``model_type`` → MTP inject router.

Historically, callers of the vendored MTP path
(:mod:`vllm_mlx.spec_decode.mtp`) reached directly into
:mod:`vllm_mlx.spec_decode.mtp.qwen3_5_inject`. This dispatcher keeps
family implementations behind a small registry so new architectures can
be added incrementally.

Registered families:

* Qwen3.5 / Qwen3.6 (``qwen3_5`` / ``qwen3_5_moe``) — production,
  MTP baked into the target checkpoint.
* Gemma 4 (``gemma4`` / ``gemma4_unified`` / ``gemma4_text`` /
  ``gemma4_unified_text``) — EXPERIMENTAL (0.10.6 A3 spike). Routes
  through :mod:`~vllm_mlx.spec_decode.mtp.gemma4_inject`. Runtime
  activation is gated at the CLI eligibility check
  (``--mtp-gemma4-sidecar-experimental`` + ``has_external_sidecar``
  path in :func:`~vllm_mlx.spec_decode.mtp.detect.detect_mtp_eligibility`),
  not here — the dispatcher's job is purely to find the family
  implementation. Direct low-level callers (tests, bench harnesses)
  can invoke ``dispatch_mtp_inject`` with the Gemma 4 model_type and
  a sidecar path; the CLI is where the "should this run" policy lives.

This module is intentionally the smallest possible dispatcher — no
config mutation, no monkey-patching. It resolves the family, forwards
the call, and returns the bool the caller expects (see the
``bench/bench_spec_decode_mtp.py`` and ``vllm_mlx.utils.tokenizer``
caller shape).

Adding a new architecture
-------------------------

1. Write the family-specific inject module.
2. Add the ``model_type`` string(s) to :data:`_MTP_INJECT_DISPATCH`
   below, mapping to the module path + entry function name.
3. Add the ``model_type`` to ``detect._SUPPORTED_MODEL_TYPES`` so the
   CLI can accept ``--speculative-config '{"method":"mtp"}'`` for that
   architecture at parse time.

All three steps are strictly additive — existing architectures keep
their current call sites.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Map ``config.json::model_type`` → ``(module_dotted_path,
# entry_function_name)``. The module is imported lazily on first
# dispatch so this file stays cheap to import from top-level MTP.
_MTP_INJECT_DISPATCH: dict[str, tuple[str, str]] = {
    # Qwen3.5 dense + MoE (vendored mlx-lm PR #990 MTP head).
    "qwen3_5": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "inject_mtp_support",
    ),
    "qwen3_5_moe": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "inject_mtp_support",
    ),
    # Gemma 4 (EXPERIMENTAL — 0.10.6 A3 spike). Paired with Google's
    # ``google/gemma-4-<size>-it-assistant`` drafter checkpoints,
    # Apache 2.0. Both the outer wrapper model_types (``gemma4`` /
    # ``gemma4_unified``) and the inner text model_types
    # (``gemma4_text`` / ``gemma4_unified_text``) route to the same
    # ``gemma4_inject`` module so callers that resolve model_type on
    # the inner ``language_model.args`` still land correctly. The CLI
    # eligibility gate blocks these paths from firing unless the
    # operator opts in via ``--mtp-gemma4-sidecar-experimental``.
    "gemma4": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_unified": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_unified_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
}


# Same schema, but for ``validate_mtp_support``. Kept as a separate
# table so a family with a bespoke validator can register it
# independently of the inject entry.
_MTP_VALIDATE_DISPATCH: dict[str, tuple[str, str]] = {
    "qwen3_5": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "validate_mtp_support",
    ),
    "qwen3_5_moe": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "validate_mtp_support",
    ),
    "gemma4": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
    "gemma4_unified": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
    "gemma4_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
    "gemma4_unified_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
}


def dispatch_mtp_inject(
    model: Any,
    model_type: str,
    *,
    mtp_sidecar: str | Path | None = None,
    allow_random_init: bool = False,
) -> bool:
    """Route an inject call to the family-specific implementation.

    .. warning::

        **INTERNAL LOW-LEVEL PRIMITIVE — NOT A PRODUCTION ACTIVATION
        SURFACE.** For the Gemma 4 family, this dispatcher will invoke
        the family inject on any registered ``model_type`` regardless
        of whether the "should this run" experimental policy has been
        satisfied. The policy gate lives in the CLI eligibility layer
        (``detect_mtp_eligibility(experimental_gemma4=True,
        has_external_sidecar=True)``) and in the scheduler's
        :func:`vllm_mlx.scheduler._config_vetted_mtp_supports_spec_decode`
        twin gate, NOT here. If you are wiring a new in-process
        integration path (embedders, test harnesses, non-CLI servers),
        replicate BOTH gates upstream of your ``dispatch_mtp_inject``
        call — otherwise you can attach the drafter to a Gemma 4 target
        without the operator having opted in. Codex round-B NIT #5
        called this out; the fix here is docstring + tests, because
        threading the CLI policy into the dispatcher itself would
        conflate a low-level router with a high-level activation
        decision.

    Args:
        model: Loaded model instance (from ``mlx_lm.load()``).
        model_type: The ``config.json::model_type`` string.
        mtp_sidecar: Optional sidecar reference — forwarded verbatim.
        allow_random_init: Test-only escape hatch — forwarded verbatim.

    Returns:
        ``True`` when the family inject succeeded and the model now
        exposes the four MTP contract surfaces. ``False`` on any
        refusal — unknown ``model_type``, family-level fail-closed
        default, missing sidecar, or import failure.

    Never raises — an unknown ``model_type`` is treated as "no MTP
    support for this arch" (log-and-return False), matching the
    fail-closed default the qwen3_5 side installed.
    """
    key = _MTP_INJECT_DISPATCH.get(model_type)
    if key is None:
        logger.info(
            "[mtp.dispatch] model_type=%r has no registered MTP inject; "
            "skipping. Registered: %s",
            model_type,
            sorted(_MTP_INJECT_DISPATCH),
        )
        return False

    module_path, func_name = key
    # Codex round-15 nit fix: family modules may raise ANY exception
    # at import time (dependency import errors, syntax errors on a
    # broken feature branch, top-level code that raises). The
    # dispatcher's own docstring promises "Never raises"; catching
    # only ``ImportError`` here would let a bare-``Exception`` module
    # bug escape the boundary. Match the ``except Exception`` used
    # later around the family call.
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        logger.warning(
            "[mtp.dispatch] could not import %s for model_type=%r: %s",
            module_path,
            model_type,
            exc,
        )
        return False

    fn = getattr(module, func_name, None)
    if fn is None:
        logger.warning(
            "[mtp.dispatch] %s has no %s(); skipping.", module_path, func_name
        )
        return False

    # The family injector is expected to return bool (never raise per
    # its own docstring), but the dispatcher's own "never raises"
    # contract is stricter than the family's — any loader crash,
    # weight-shape mismatch, or bug inside the family must still land
    # here as a clean False. Wrap defensively.
    try:
        return bool(
            fn(
                model,
                mtp_sidecar=mtp_sidecar,
                allow_random_init=allow_random_init,
            )
        )
    except Exception as exc:
        logger.warning(
            "[mtp.dispatch] %s.%s raised for model_type=%r: %s. "
            "Treating as inject failure.",
            module_path,
            func_name,
            model_type,
            exc,
        )
        return False


def dispatch_mtp_validate(model: Any, model_type: str) -> bool:
    """Route a ``validate_mtp_support`` call to the family validator.

    Returns ``False`` for any unknown ``model_type``. Never raises.
    """
    key = _MTP_VALIDATE_DISPATCH.get(model_type)
    if key is None:
        logger.info(
            "[mtp.dispatch] model_type=%r has no registered validator; "
            "returning False.",
            model_type,
        )
        return False

    module_path, func_name = key
    # Codex round-15 nit fix: mirror the ``except Exception`` guard
    # applied above on the inject side. Any import-time failure in the
    # family module (not just ImportError) must land as a clean False.
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        logger.warning(
            "[mtp.dispatch] could not import %s for validator: %s",
            module_path,
            exc,
        )
        return False

    fn = getattr(module, func_name, None)
    if fn is None:
        return False
    try:
        return bool(fn(model))
    except Exception as exc:
        logger.warning(
            "[mtp.dispatch] %s.%s raised for validate: %s. Returning False.",
            module_path,
            func_name,
            exc,
        )
        return False
