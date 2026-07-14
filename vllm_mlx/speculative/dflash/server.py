# SPDX-License-Identifier: Apache-2.0
"""DFlash server — dedicated single-user mode that bypasses BatchedEngine.

When DFlash is enabled, the CLI launches this server instead of the
standard ``vllm_mlx.server.app``. It hosts a minimal OpenAI-compatible
surface (``/healthz``, ``/v1/models``, ``/v1/chat/completions``) and routes
generation through mlx-vlm's ``stream_generate`` with the loaded DFlash
drafter.

Why a separate server (not a fork of the standard route)?
  - mlx-vlm's ``generate_step`` is a per-request Python generator with its
    own ``prompt_cache`` argument. BatchedEngine merges per-request KV
    caches into a ``BatchKVCache``. Grafting one onto the other would
    invent batched-DFlash that doesn't exist upstream and would risk
    regressing the non-DFlash path under attention layout changes.
  - DFlash today only validates on B=1 anyway (see PoC: 1.83-2.18× on
    Qwen3.5-27B-8bit; no batched-DFlash kernel exists in mlx-vlm 0.5.0).
  - A separate, opt-in server is a clean blast-radius boundary: turning
    on DFlash can never break a request that doesn't use it.

v1 limitations (documented in README + ``rapid-mlx info``):
  - Single-user serial. Concurrent requests queue on an ``asyncio.Lock``.
  - No tool calling, MCP, embeddings, or audio in this server (the
    standard server handles those).
  - No prefix cache (per-request KV cache built fresh each call).

These limitations are deliberate for v1 — the target user is someone
running ``rapid-mlx serve qwen3.5-27b-8bit --speculative-config
'{"method":"dflash"}'`` to get a ~2× speedup on code/long-form
completions on a single Apple Silicon box.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
)
from vllm_mlx.config import get_config

from .eligibility import have_runtime
from .runtime import DFlashRuntime, load_runtime

logger = logging.getLogger(__name__)


# Global serial lock — DFlash is single-stream by design (mlx-vlm doesn't
# expose a batched DFlash kernel in 0.5.0). The second concurrent request
# waits its turn; this matches the PoC reality.
_dflash_lock = asyncio.Lock()


# Dedicated single-thread executor so every mlx-vlm call (drafter loading,
# generate, stream_generate's ``next``) executes on ONE thread for the
# lifetime of the process. Reason: mlx-lm 0.31.3+ keeps the GPU Stream
# in thread-local storage; iterating a generator across threads (which
# would happen if we used the default ThreadPoolExecutor with N workers)
# trips "There is no Stream(gpu, N) in current thread" mid-stream. Pinning
# to one worker preserves thread affinity and matches the serial-only
# contract enforced by ``_dflash_lock``.
_dflash_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dflash-worker"
)


class _DFlashAdmissionReservation:
    """One request slot from a :class:`_DFlashAdmission` gate."""

    def __init__(self, admission: _DFlashAdmission) -> None:
        self._admission = admission
        self._deferred = False
        self._released = False

    def defer_release(self) -> None:
        """Keep the slot until a timed-out worker has actually stopped."""
        self._deferred = True

    def release(self, *, force: bool = False) -> None:
        if self._deferred and not force:
            return
        with self._admission._lock:
            if self._released:
                return
            self._released = True
            self._admission._reservations = max(0, self._admission._reservations - 1)


class _DFlashAdmission:
    """Bound DFlash's serial-lock queue before it consumes worker memory."""

    def __init__(self, max_concurrent_requests: int) -> None:
        self._max_concurrent_requests = max(0, int(max_concurrent_requests))
        self._reservations = 0
        self._lock = threading.Lock()

    def reserve(self) -> _DFlashAdmissionReservation:
        with self._lock:
            if (
                self._max_concurrent_requests > 0
                and self._reservations >= self._max_concurrent_requests
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "DFlash is at its max_concurrent_requests admission "
                        "limit; retry shortly."
                    ),
                    headers={"Retry-After": "1"},
                )
            self._reservations += 1
        return _DFlashAdmissionReservation(self)


class _DFlashStreamLease:
    """Keep DFlash resources owned until a cancelled worker is cleaned up."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        reservation: _DFlashAdmissionReservation | None,
        deadline: float | None,
    ) -> None:
        self._loop = loop
        self._reservation = reservation
        self._deadline = deadline
        self.timed_out = False
        self._lock_acquired = False
        self._released = False
        self._cleanup_deferred = False
        self._active_future: asyncio.Future[Any] | None = None
        self._active_future_makes_generator = False
        self._generator: Any | None = None

    async def __aenter__(self) -> _DFlashStreamLease:
        if self._deadline is None:
            await _dflash_lock.acquire()
        else:
            remaining = self._deadline - self._loop.time()
            if remaining <= 0:
                self.timed_out = True
                return self
            try:
                await asyncio.wait_for(_dflash_lock.acquire(), timeout=remaining)
            except asyncio.TimeoutError:
                self.timed_out = True
                return self
        self._lock_acquired = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        future = self._active_future
        if future is not None and not future.done():
            self._defer_cleanup(future)
            return False
        if future is not None:
            # Cancellation can land after the executor future resolves but
            # before its caller stores a constructed generator.
            self._capture_generator(future)
            self.clear_future(future)
        await self._close_generator_and_release()
        return False

    def track_future(
        self, future: asyncio.Future[Any], *, makes_generator: bool = False
    ) -> None:
        self._active_future = future
        self._active_future_makes_generator = makes_generator

    def clear_future(self, future: asyncio.Future[Any]) -> None:
        if self._active_future is future:
            self._active_future = None
            self._active_future_makes_generator = False

    def set_generator(self, generator: Any) -> None:
        self._generator = generator

    def _capture_generator(self, future: asyncio.Future[Any]) -> None:
        if not self._active_future_makes_generator or self._generator is not None:
            return
        try:
            candidate = future.result()
        except Exception:  # noqa: BLE001 -- worker error has no generator to close
            return
        if not isinstance(candidate, Exception) and hasattr(candidate, "close"):
            self._generator = candidate

    def _defer_cleanup(self, future: asyncio.Future[Any]) -> None:
        if self._cleanup_deferred:
            return
        self._cleanup_deferred = True
        if self._reservation is not None:
            self._reservation.defer_release()

        def _on_worker_done(done: asyncio.Future[Any]) -> None:
            self._capture_generator(done)
            self._active_future = None
            self._active_future_makes_generator = False
            self._loop.create_task(self._close_generator_and_release())

        future.add_done_callback(_on_worker_done)

    async def _close_generator_and_release(self) -> None:
        generator = self._generator
        if generator is not None:

            def _close_gen() -> None:
                try:
                    generator.close()
                except Exception:  # noqa: BLE001 -- cleanup is best-effort
                    logger.debug(
                        "DFlash generator close raised; ignoring", exc_info=True
                    )

            close_future = self._loop.run_in_executor(_dflash_executor, _close_gen)
            try:
                await asyncio.shield(close_future)
            except asyncio.CancelledError:
                # The client task can be cancelled a second time while it is
                # already unwinding. The queued close still owns the GPU
                # ordering, so release only from its completion callback.
                close_future.add_done_callback(lambda _done: self._release())
                return
        self._release()

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._lock_acquired:
            _dflash_lock.release()
            self._lock_acquired = False
        if self._reservation is not None:
            self._reservation.release(force=True)


@atexit.register
def _shutdown_dflash_executor() -> None:
    """Drain the DFlash worker on interpreter exit. Python registers an
    implicit atexit for ThreadPoolExecutor, but registering ours
    explicitly makes shutdown order deterministic and silences
    "unfinished thread" warnings during graceful uvicorn termination."""
    _dflash_executor.shutdown(wait=False, cancel_futures=True)


def _build_app(
    *,
    model: Any,
    processor: Any,
    runtime: DFlashRuntime,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    no_thinking: bool = False,
    api_key: str | None = None,
    rate_limit: int = 0,
    max_request_bytes: int = 8 * 1024 * 1024,
    body_receive_timeout_seconds: float = 15.0,
    default_timeout: float = 1800.0,
    max_concurrent_requests: int = 256,
    cors_policy: Any | None = None,
) -> FastAPI:
    """Create the FastAPI application for DFlash mode.

    Per-app model state (``model``, ``processor``, ``runtime``,
    ``served_model_name``) is captured by closure. The security policy
    deliberately uses the shared server config and rate limiter, matching
    the unified server; DFlash therefore supports one active app per
    process.

    Note: ``_dflash_lock`` and ``_dflash_executor`` are *module-level*
    by design — every DFlash invocation must serialise through the
    same single-thread worker because mlx's GPU Stream is thread-local
    (see the ``_dflash_executor`` docstring at module top). A future
    multi-model deployment would still share that worker; one model
    can't run while another's generator is mid-step.
    """
    # DFlash owns a separate FastAPI application, so it cannot inherit the
    # unified server's dependencies or middleware implicitly. Copy the
    # already-resolved security settings into the shared config singleton
    # before wiring those common protections onto this app. In particular,
    # ``verify_api_key`` and ``RequestBodyLimitMiddleware`` read this
    # singleton at request time.
    cfg = get_config()
    cfg.api_key = api_key
    cfg.max_request_bytes = max(0, int(max_request_bytes))
    cfg.body_receive_timeout_seconds = max(0.0, float(body_receive_timeout_seconds))
    cfg.default_timeout = max(0.0, float(default_timeout))

    from ...middleware.auth import (
        check_rate_limit,
        configure_rate_limiter,
        verify_api_key,
    )
    from ...middleware.body_depth import install_request_body_depth_middleware
    from ...middleware.body_size import install_request_body_limit_middleware

    configure_rate_limiter(rate_limit, enabled=rate_limit > 0)

    app = FastAPI(title="Rapid-MLX (DFlash)")
    # DFlash has one GPU worker and serializes generation with
    # ``_dflash_lock``. Bound its waiting room as well so a burst cannot
    # accumulate an unbounded number of requests and their parsed bodies.
    admission = _DFlashAdmission(max_concurrent_requests)
    app.state.dflash_admission = admission
    # D-ANTHRO-VALIDATION F11: install the shared exception handlers so
    # Pydantic validation errors return the canonical
    # ``{"error":{"type":"invalid_request_error","code":"invalid_request",
    # ...}}`` envelope at HTTP 400 instead of FastAPI's default 422 with
    # an unbounded ``detail`` array. Same handlers the main server uses.
    from ...middleware.exception_handlers import install_exception_handlers

    install_exception_handlers(app)
    # Match the main server's generic JSON request defenses. The body-size
    # middleware also enforces the configured body-receive idle timeout, so a
    # DFlash request cannot bypass the slow-client protection by taking this
    # dedicated app path.
    install_request_body_depth_middleware(app)
    install_request_body_limit_middleware(app)
    # F-090/F-091: register CORS only when an explicit origin allowlist is
    # configured. ``cors_origins=[]`` (the new default — see
    # ``vllm_mlx/server.py::configure_cors_from_env``) skips the middleware
    # entirely so preflight returns 405 and no ``Access-Control-*`` header
    # leaks. The dflash path mirrors the main server's stance.
    if cors_policy is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_policy.origins,
            allow_credentials=cors_policy.allow_credentials,
            allow_methods=cors_policy.methods,
            allow_headers=cors_policy.headers,
            max_age=cors_policy.max_age,
        )
    elif cors_origins:
        wildcard = "*" in cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            # Fetch spec: wildcard + credentials is invalid; flip off
            # credentials when ``*`` is present so the response stays
            # browser-valid.
            allow_credentials=not wildcard,
            # F-091: previously ``["*"]`` (DELETE/GET/HEAD/OPTIONS/PATCH/
            # POST/PUT). The dflash server only serves the OpenAI-compat
            # chat surface, so POST/GET/OPTIONS is the correct allowlist.
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Rapid-MLX-Internal"],
            max_age=3600,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "dflash",
            "mode": "single-user-serial",
            "drafter": runtime.drafter_repo,
        }

    @app.get("/v1/models", dependencies=[Depends(verify_api_key)])
    async def list_models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=served_model_name,
                    created=int(time.time()),
                    owned_by="rapid-mlx",
                )
            ]
        )

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if request.n is not None and request.n > 1:
            raise HTTPException(status_code=400, detail="n > 1 is not supported")
        if request.tools:
            # DFlash server doesn't run a tool-call parser. Surface this so
            # users don't think their tools "silently worked" when in fact
            # the model just emitted free-form text.
            raise HTTPException(
                status_code=400,
                detail=(
                    "Tool calling is not supported in DFlash mode (v1 "
                    "limitation). Restart without DFlash to use tools."
                ),
            )
        # Surface unsupported params explicitly rather than silently
        # ignoring — silent-drop is the bug class that makes users think
        # they got logprobs / JSON-schema / etc. when they didn't.
        if request.logprobs:
            raise HTTPException(
                status_code=400,
                detail="logprobs is not supported in DFlash mode. Restart without DFlash.",
            )
        if request.response_format is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format (structured output) is not supported "
                    "in DFlash mode. Restart without DFlash."
                ),
            )
        reservation = admission.reserve()
        try:
            # Render chat messages into a single prompt string via mlx-vlm's
            # processor. We pass through the model's chat template so the
            # tokenizer-side reasoning/tool markers match what the model was
            # trained on; no rapid-mlx-side prompt mutation happens here.
            #
            # Resolve enable_thinking (#387). The dflash app captures its own
            # ``no_thinking`` by closure rather than going through the
            # ServerConfig singleton, so we apply that override first then
            # delegate the request-side precedence (chat_template_kwargs >
            # request.enable_thinking > None) to the shared extractor — same
            # source of truth as the OpenAI/anthropic helper, but without the
            # ``cfg.no_thinking`` consult that doesn't apply to dflash.
            from ...service.helpers import _extract_thinking_from_request

            if no_thinking:
                enable_thinking: bool | None = False
            else:
                enable_thinking = _extract_thinking_from_request(request)
            prompt = _render_prompt(
                processor, model, request, enable_thinking=enable_thinking
            )

            max_tokens = (
                request.max_tokens
                if request.max_tokens is not None
                else default_max_tokens
            )
            temperature = (
                request.temperature if request.temperature is not None else 0.0
            )
            top_p = request.top_p if request.top_p is not None else 1.0

            gen_kwargs = dict(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                draft_model=runtime.drafter,
                draft_kind=runtime.kind,
            )
        except BaseException:
            reservation.release()
            raise

        if request.stream:
            return StreamingResponse(
                _stream_with_admission(
                    _stream_completion(
                        prompt=prompt,
                        request=request,
                        served_model_name=served_model_name,
                        gen_kwargs=gen_kwargs,
                        model=model,
                        processor=processor,
                        timeout=request.timeout or default_timeout,
                        admission_reservation=reservation,
                    ),
                    reservation,
                ),
                media_type="text/event-stream",
            )

        try:
            return await _non_stream_completion(
                prompt=prompt,
                request=request,
                served_model_name=served_model_name,
                gen_kwargs=gen_kwargs,
                model=model,
                processor=processor,
                timeout=request.timeout or default_timeout,
                admission_reservation=reservation,
            )
        finally:
            reservation.release()

    return app


def _render_prompt(
    processor: Any,
    model: Any,
    request: ChatCompletionRequest,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply the model's chat template via mlx-vlm's helper.

    mlx-vlm's ``apply_chat_template`` mirrors mlx-lm's but accepts the
    multimodal kwargs the VLM models need (we pass ``num_images=0`` since
    DFlash-eligible aliases are text-only Qwen3.5/3.6 variants today).

    ``enable_thinking`` resolution (caller-side; we just thread through):
      None  → defer to mlx-vlm default (Qwen3 family = True).
      True  → force chain-of-thought on.
      False → force chain-of-thought off (server --no-thinking or per-
              request ``enable_thinking=false`` body field).
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = []
    for m in request.messages:
        content = m.content
        if isinstance(content, list):
            # Multimodal payload — DFlash server is text-only. Collapse
            # text parts; non-text parts (image/audio/video) are
            # dropped. A 400 would surprise users mid-prompt, but a
            # silent drop hides "why is my model ignoring the image?"
            # debugging — so we degrade with a visible WARN log per
            # request that hits this path.
            text_pieces = []
            dropped_kinds: list[str] = []
            for part in content:
                part_type = part.type if hasattr(part, "type") else part.get("type", "")
                if part_type == "text":
                    text_pieces.append(
                        part.text if hasattr(part, "text") else part.get("text", "")
                    )
                elif part_type:
                    dropped_kinds.append(part_type)
            if dropped_kinds:
                logger.warning(
                    "DFlash server is text-only; dropped %d non-text "
                    "content part(s) of type(s) %s. The request will be "
                    "served using text parts only — switch to the standard "
                    "server without DFlash for full multimodal support.",
                    len(dropped_kinds),
                    sorted(set(dropped_kinds)),
                )
            content = "".join(text_pieces)
        messages.append({"role": m.role, "content": content})

    # Preserve historic default (enable_thinking=True) when neither the
    # server-level --no-thinking nor a per-request body override is set,
    # to keep behaviour stable for callers that never opt out.
    effective_thinking = True if enable_thinking is None else enable_thinking
    return apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=0,
        num_audios=0,
        enable_thinking=effective_thinking,
    )


async def _stream_with_admission(
    stream: AsyncIterator[bytes], reservation: _DFlashAdmissionReservation
) -> AsyncIterator[bytes]:
    """Release an admission slot even when Starlette cancels an SSE stream."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        # ``StreamingResponse`` runs its background task only on a normal
        # return. Closing the inner generator here covers client disconnects
        # and send failures too, so its GPU cleanup runs before the slot is
        # made available again.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 -- release remains mandatory
                logger.debug(
                    "DFlash stream close raised; releasing admission", exc_info=True
                )
        reservation.release()


async def _stream_completion(
    *,
    prompt: str,
    request: ChatCompletionRequest,
    served_model_name: str,
    gen_kwargs: dict[str, Any],
    model: Any,
    processor: Any,
    timeout: float | None = None,
    admission_reservation: _DFlashAdmissionReservation | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI-format chunks. Generation happens under the serial
    lock; chunks are forwarded as ``data: ...\\n\\n`` SSE events."""
    from mlx_vlm import stream_generate

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # First chunk — role marker
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model_name,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first)}\n\n".encode()

    finish_reason = "stop"
    total_completion_tokens = 0
    prompt_tokens = 0
    # Track the last token id to disambiguate "hit max_tokens but the
    # final token was actually EOS" — without this we'd falsely flag a
    # natural-stop response as truncated when it lands on exactly the
    # budget. None means "no token observed yet".
    last_token_id: int | None = None

    # Track max_tokens so we can report ``finish_reason="length"`` when
    # generation was truncated (OpenAI clients distinguish "stop"
    # = natural end / stop sequence from "length" = token-budget hit;
    # presenting "stop" for a truncated reply misleads downstream tools).
    _max_tokens = gen_kwargs.get("max_tokens")

    # Resolve the model's EOS token id (best-effort). Used by the
    # length-vs-stop disambiguation below; falls back to None when the
    # processor doesn't expose a tokenizer (the heuristic then degrades
    # to pure token-count comparison).
    _eos_ids: set[int] = set()
    _tok = getattr(processor, "tokenizer", processor)
    _eos = getattr(_tok, "eos_token_id", None)
    if isinstance(_eos, int):
        _eos_ids.add(_eos)
    elif isinstance(_eos, (list, tuple, set)):
        _eos_ids.update(int(t) for t in _eos if isinstance(t, int))

    error_message: str | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout if timeout and timeout > 0 else None
    timed_out = object()
    lease = _DFlashStreamLease(loop, admission_reservation, deadline)

    async def _await_worker(func: Any, *, makes_generator: bool = False) -> Any:
        """Wait without cancelling an mlx operation on deadline expiry."""
        if lease.timed_out:
            return timed_out
        future = loop.run_in_executor(_dflash_executor, func)
        lease.track_future(future, makes_generator=makes_generator)
        if deadline is None:
            result = await future
            lease.clear_future(future)
            return result
        remaining = deadline - loop.time()
        if remaining <= 0:
            # The future was submitted by the caller already. It may be a
            # GPU step, so wait for it before releasing the serial lock.
            result = await asyncio.shield(future)
            lease.clear_future(future)
            return result, timed_out
        done, _pending = await asyncio.wait({future}, timeout=remaining)
        if done:
            result = future.result()
            lease.clear_future(future)
            return result

        # mlx work is not preemptible. Let the in-flight token step finish
        # while the serial lock remains held, then close its generator rather
        # than letting another request overlap it on the GPU worker.
        result = await asyncio.shield(future)
        lease.clear_future(future)
        return result, timed_out

    async with lease:
        # mlx-vlm's stream_generate is a sync generator — run it in a
        # thread pool so we don't block the FastAPI event loop. Iterate
        # by polling with ``run_in_executor`` per chunk. We're already
        # inside a coroutine, so use ``get_running_loop`` (the 3.10+
        # idiom; ``get_event_loop`` is deprecated for in-coroutine use).
        # The executor MUST be ``_dflash_executor`` (single-thread) so
        # consecutive ``next(gen)`` calls land on the same worker —
        # mlx's GPU Stream is thread-local and a hand-off across worker
        # threads would crash mid-generation.
        # Create the generator on the same worker that will drive it,
        # not on the event-loop thread — otherwise the first ``next``
        # crosses a thread boundary just like the rest.
        #
        # Wrap construction in a sentinel pattern too: if
        # ``stream_generate`` raises at setup time (OOM, missing kernel,
        # bad arg) the exception would otherwise propagate out of the
        # async generator and leave the SSE client hanging without a
        # ``[DONE]``. Surfacing it as an error SSE keeps the contract
        # the same as the mid-stream error path below.
        def _make_gen():
            try:
                return stream_generate(model, processor, prompt, **gen_kwargs)
            except Exception as e:  # noqa: BLE001 — surface upstream; outer code converts to error SSE
                return e

        gen_or_err = await _await_worker(_make_gen, makes_generator=True)
        timed_out_after_generation = False
        if gen_or_err is timed_out:
            timed_out_after_generation = True
        elif isinstance(gen_or_err, tuple):
            gen_or_err, _timeout_marker = gen_or_err
            timed_out_after_generation = _timeout_marker is timed_out
        if timed_out_after_generation:
            error_message = f"DFlash stream timed out after {timeout:.1f} seconds."
            finish_reason = "length"
            gen = (
                None
                if gen_or_err is timed_out or isinstance(gen_or_err, Exception)
                else gen_or_err
            )
            if gen is not None:
                lease.set_generator(gen)
        elif isinstance(gen_or_err, Exception):
            logger.exception(
                "DFlash stream_generate raised at construction: %s",
                gen_or_err,
                exc_info=gen_or_err,
            )
            error_message = f"{type(gen_or_err).__name__}: {gen_or_err}"
            # OpenAI ChatCompletion only accepts {stop, length, tool_calls,
            # content_filter, function_call}. The error block on the final
            # SSE chunk carries the abort details for clients.
            finish_reason = "length"
            gen = None
        else:
            gen = gen_or_err
            lease.set_generator(gen)

        # Sentinels distinguish "generator exhausted" (None) from
        # "generator raised mid-stream" (an Exception instance). Catching
        # only StopIteration would let any other mlx-vlm error propagate
        # through run_in_executor, abort the response coroutine, and
        # leave the SSE client hanging without a final ``[DONE]`` — the
        # client then either times out or holds the connection forever.
        def _next_chunk():
            try:
                return next(gen)
            except StopIteration:
                return None
            except Exception as e:  # noqa: BLE001 — surface upstream; loop converts to error SSE
                return e

        while gen is not None:
            chunk = await _await_worker(_next_chunk)
            timed_out_after_generation = False
            if isinstance(chunk, tuple):
                chunk, _timeout_marker = chunk
                timed_out_after_generation = _timeout_marker is timed_out
            if timed_out_after_generation:
                error_message = f"DFlash stream timed out after {timeout:.1f} seconds."
                finish_reason = "length"
                break
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                logger.exception(
                    "DFlash stream_generate raised mid-stream: %s",
                    chunk,
                    exc_info=chunk,
                )
                error_message = f"{type(chunk).__name__}: {chunk}"
                # See above for OpenAI spec literal-set rationale.
                finish_reason = "length"
                break
            # Always sync token counts from the chunk — even when text
            # is empty (mlx-vlm occasionally emits trailing flush
            # chunks carrying the final token counters but no
            # incremental text). Skipping the update would leave the
            # final usage block with stale numbers.
            total_completion_tokens = chunk.generation_tokens
            prompt_tokens = chunk.prompt_tokens
            _ct = getattr(chunk, "token", None)
            if isinstance(_ct, int):
                last_token_id = _ct
            if not chunk.text:
                continue
            piece = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(piece)}\n\n".encode()

    # Length-truncation detection — mlx-vlm's GenerationResult has no
    # ``finish_reason`` field, so we infer "length" by comparing the
    # completion token count to the budget. Only set when we exited the
    # loop normally (StopIteration), not when the generator errored or
    # produced fewer tokens (natural stop).
    #
    # Subtle case: if the model emitted EOS exactly at ``max_tokens``,
    # the stop was natural and reporting "length" would mislead clients
    # into auto-continuing (only to get an immediate EOS again). Check
    # the last token id against the resolved EOS set to keep the
    # classification honest in this edge case.
    if (
        finish_reason == "stop"
        and _max_tokens is not None
        and total_completion_tokens >= _max_tokens
        and last_token_id not in _eos_ids
    ):
        finish_reason = "length"

    # Final chunk — finish_reason + usage. If we broke out of the loop
    # because the underlying generator raised, attach an OpenAI-style
    # error block so the client gets a readable failure instead of
    # silent truncation.
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }
    if error_message is not None:
        final["error"] = {"type": "dflash_runtime_error", "message": error_message}
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _non_stream_completion(
    *,
    prompt: str,
    request: ChatCompletionRequest,
    served_model_name: str,
    gen_kwargs: dict[str, Any],
    model: Any,
    processor: Any,
    timeout: float = 1800.0,
    admission_reservation: _DFlashAdmissionReservation | None = None,
) -> ChatCompletionResponse:
    """Run generation under the serial lock and enforce a safe deadline.

    ``mlx_vlm.generate`` cannot be preempted once it is running on the
    dedicated worker. On timeout we return a 504, but keep both the serial
    lock and admission slot until the worker exits; otherwise a second call
    could overlap the first GPU operation on the same thread.
    """
    from mlx_vlm import generate

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Keep the helper usable by direct programmatic callers and existing
    # focused tests. The request route always supplies its real admission
    # reservation; this private fallback only matters outside the ASGI path.
    if admission_reservation is None:
        admission_reservation = _DFlashAdmission(0).reserve()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    lock_acquired = False
    worker_future: asyncio.Future[Any] | None = None

    def _release_after_worker(_future: asyncio.Future[Any]) -> None:
        """Run in the event loop after a timed-out worker exits."""
        _dflash_lock.release()
        admission_reservation.release(force=True)

    def _defer_worker_cleanup() -> None:
        nonlocal lock_acquired
        assert worker_future is not None
        admission_reservation.defer_release()
        worker_future.add_done_callback(_release_after_worker)
        # The callback above is now responsible for the serial lock. Do not
        # release it in this coroutine's finally block.
        lock_acquired = False

    try:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        await asyncio.wait_for(_dflash_lock.acquire(), timeout=remaining)
        lock_acquired = True

        # mlx-vlm's ``generate`` blocks; offload to the dedicated
        # single-thread DFlash executor so every mlx-vlm call lands on
        # the same worker (matches ``_stream_completion`` — see the
        # _dflash_executor comment at module top).
        #
        # Wrap in a sentinel pattern so generate-time errors (OOM, bad
        # arg, drafter mismatch) come back as a clean HTTP 500 with a
        # readable detail string rather than a raw stack trace. Mirrors
        # the stream path's error handling.
        def _generate_safely():
            try:
                return generate(model, processor, prompt, **gen_kwargs)
            except Exception as e:  # noqa: BLE001 — surface as HTTPException below
                return e

        worker_future = loop.run_in_executor(_dflash_executor, _generate_safely)
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        result = await asyncio.wait_for(
            asyncio.shield(worker_future), timeout=remaining
        )
    except asyncio.TimeoutError as exc:
        if lock_acquired and worker_future is not None:
            _defer_worker_cleanup()
        raise HTTPException(
            status_code=504,
            detail=f"DFlash request timed out after {timeout:.1f} seconds.",
        ) from exc
    except asyncio.CancelledError:
        if lock_acquired and worker_future is not None:
            _defer_worker_cleanup()
        raise
    finally:
        if lock_acquired:
            _dflash_lock.release()

    if isinstance(result, Exception):
        logger.exception(
            "DFlash non-stream generate raised: %s", result, exc_info=result
        )
        raise HTTPException(
            status_code=500,
            detail=f"DFlash runtime error: {type(result).__name__}: {result}",
        )

    # OpenAI distinguishes "stop" (natural end / stop sequence) from
    # "length" (token-budget hit). mlx-vlm doesn't surface that on
    # GenerationResult, so infer from token-count vs requested budget.
    #
    # Known v1 limitation: unlike the streaming path which can read
    # ``chunk.token`` and check against EOS, ``mlx_vlm.generate``
    # returns only the concatenated text + token counts. If the model
    # emits EOS at exactly ``max_tokens`` the non-stream response will
    # still report ``finish_reason="length"`` (false truncation). A
    # client that auto-continues will issue one more request that
    # immediately returns EOS — annoying but not corrupt. Fix requires
    # an upstream mlx-vlm change to expose the final token id; tracked
    # as a v2 follow-up.
    _max_tokens = gen_kwargs.get("max_tokens")
    finish_reason = (
        "length"
        if _max_tokens is not None and result.generation_tokens >= _max_tokens
        else "stop"
    )

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=served_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(role="assistant", content=result.text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.generation_tokens,
            total_tokens=result.prompt_tokens + result.generation_tokens,
        ),
    )


def run_dflash_server(
    *,
    main_model_repo: str,
    drafter_repo: str,
    host: str,
    port: int,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    uvicorn_log_level: str,
    no_thinking: bool = False,
    api_key: str | None = None,
    rate_limit: int = 0,
    max_request_bytes: int = 8 * 1024 * 1024,
    body_receive_timeout_seconds: float = 15.0,
    default_timeout: float = 1800.0,
    max_concurrent_requests: int = 256,
    cors_policy: Any | None = None,
) -> None:
    """Load the model + DFlash drafter via mlx-vlm and start uvicorn.

    The mlx-vlm load path is mandatory: the DFlash hooks
    (``capture_layer_ids``, ``_dflash_rounds``) live on the mlx-vlm
    model classes, not mlx-lm's. Loading via ``mlx_lm.load`` would give
    us a model without the hooks and DFlash would silently fall back to
    AR — exactly the kind of "silent regression" the eligibility gate
    is meant to prevent. We surface a clear error if mlx-vlm is missing
    or too old.

    Eligibility re-check: even though the CLI's ``serve_command`` gates
    on the alias before calling here, a *programmatic* caller (e.g. a
    notebook or test harness) can bypass the CLI entirely. We re-run
    the path-detectable gates (4-bit quant via repo-name heuristic;
    non-empty drafter). MoE detection requires the AliasProfile (an
    ``is_moe`` flag aliases.json maintains by hand) and is therefore
    only enforced via the CLI entrypoint — callers serving an
    arbitrary ``main_model_repo`` programmatically are responsible for
    not pointing it at a MoE model. Documented in CALLERS.md.
    """
    if not have_runtime():
        raise RuntimeError(
            "DFlash server requires mlx-vlm 0.5.0+ — install with "
            "``pip install 'rapid-mlx[dflash]'``."
        )

    # Belt-and-suspenders eligibility re-check for programmatic callers
    # (the CLI's serve_command already gates on the alias upstream, but
    # we don't want to depend on it being the only entrypoint).
    from .eligibility import (
        DFlashUnavailable,
        _looks_like_4bit,  # noqa: PLC2701 — internal helper
    )

    if _looks_like_4bit(main_model_repo):
        raise DFlashUnavailable(
            f"DFlash cannot run on a 4-bit quantized model "
            f"(main_model_repo={main_model_repo!r}); upstream PoC measured "
            "regression to 0.63-0.96× on Qwen3.5-4B-MLX-4bit. Use the "
            "8-bit variant."
        )
    if not drafter_repo:
        raise DFlashUnavailable(
            "DFlash requires a non-empty drafter_repo — pass the DFlash "
            "drafter HF path (e.g. 'z-lab/Qwen3.5-27B-DFlash')."
        )

    import uvicorn
    from mlx_vlm import load

    # CRITICAL: load model + drafter on the dedicated DFlash executor
    # thread (not the main thread). mlx-lm 0.31.3+ keeps GPU streams in
    # thread-local storage, so weights loaded on thread A cannot be
    # evaluated on thread B — generate() raises ``RuntimeError: There
    # is no Stream(gpu, N) in current thread``. By pinning load AND all
    # subsequent generate() calls to the same single-worker executor,
    # streams stay reachable for the lifetime of the process.
    def _load_all():
        t0 = time.perf_counter()
        m, p = load(main_model_repo)
        logger.info("DFlash: main model loaded in %.1fs", time.perf_counter() - t0)
        rt = load_runtime(drafter_repo)
        return m, p, rt

    logger.info("DFlash: loading main model via mlx-vlm: %s", main_model_repo)
    model, processor, runtime = _dflash_executor.submit(_load_all).result()

    app = _build_app(
        model=model,
        processor=processor,
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        no_thinking=no_thinking,
        api_key=api_key,
        rate_limit=rate_limit,
        max_request_bytes=max_request_bytes,
        body_receive_timeout_seconds=body_receive_timeout_seconds,
        default_timeout=default_timeout,
        max_concurrent_requests=max_concurrent_requests,
        cors_policy=cors_policy,
    )

    print()
    host_display = "localhost" if host == "0.0.0.0" else host
    print(f"  Ready: http://{host_display}:{port}/v1  (DFlash mode)")
    print(f"  Docs:  http://{host_display}:{port}/docs")
    print()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )
