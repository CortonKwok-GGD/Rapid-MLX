# SPDX-License-Identifier: Apache-2.0
"""Shared async helpers for the generation route lanes.

These lanes all have the same shape: an ``async def`` handler that must
hand seconds-to-minutes of blocking engine work to a worker thread, while
holding a lock and owning a temp file for exactly as long as that work
runs. Getting the cancellation semantics right is subtle enough that it
lives in one place rather than each route carrying a copy that can drift.
"""

import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


async def run_to_completion(func, /, *args):
    """``asyncio.to_thread(func, *args)`` that survives cancellation.

    A plain ``await asyncio.to_thread(...)`` is NOT cancellable — the
    worker thread keeps running — but the await returns immediately when
    the surrounding task is cancelled. That combination is dangerous for
    the generation lanes, where a client disconnect cancels the handler
    mid-run:

    * an ``async with`` lock around the await would unwind and admit
      another request while the abandoned thread is still using the
      cached engine, destroying the one-at-a-time guarantee the lock
      exists to provide, and
    * a ``finally`` block would delete the temp file or directory the
      abandoned worker is still writing to.

    So on cancellation we wait for the worker to actually finish before
    propagating, keeping "lock held" and "output path alive" true for
    exactly as long as the thread is running. The client is already gone,
    so the extra wait costs nothing user-visible.

    Completion is tracked by a ``threading.Event`` the WORKER sets, not by
    the state of the wrapper task. Task state is not trustworthy here: if
    something cancels the inner ``to_thread`` task directly — a shutdown
    routine walking ``asyncio.all_tasks()`` is the realistic case — the task
    reports cancelled/done while its thread keeps executing, and draining on
    task state would return early and let the caller tear down the live
    worker's resources. A thread-owned event cannot be cancelled, so it is
    the only signal that actually tracks the thread.

    A cancel that lands before the pool picks up the job is handled by
    ``started``: without it, draining on ``done`` alone would wait on a
    worker that never ran and hang at shutdown.

    NOTE on test coverage: the two-cancel path IS covered (see the route
    tests), but the inner-task-cancellation scenario is not. I could not
    build a test that reliably reproduces it — the inner task is created
    inside this coroutine and never became visible to
    ``asyncio.all_tasks()`` sweeps in time, so every attempt passed against
    the buggy task-state version too. Rather than keep a test that proves
    nothing, the reasoning is recorded here: a thread's own completion
    signal cannot be cancelled out from under it, so it is strictly safer
    than task state either way.
    """
    # ``started`` distinguishes "the executor never ran our job" from "the
    # job is running": ``to_thread`` submits to a pool, so a cancel that
    # lands before a worker picks it up means ``_wrapped`` never executes —
    # and waiting on ``done`` alone would then hang forever at shutdown.
    started = threading.Event()
    done = threading.Event()

    def _wrapped():
        started.set()
        try:
            return func(*args)
        finally:
            done.set()

    task = asyncio.ensure_future(asyncio.to_thread(_wrapped))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Drain only if a worker actually began. Poll with ``asyncio.sleep``
        # rather than blocking on ``done.wait()`` in the executor: that would
        # occupy a second pool thread for up to the render ceiling, adding to
        # the very starvation this helper exists to avoid.
        while started.is_set() and not done.is_set():
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                # Absorb repeated cancels — a bare await here would let a
                # second cancel end the drain mid-render.
                logger.debug("Ignoring cancellation while draining an abandoned worker")
        # Retrieve the outcome so asyncio doesn't log "Task exception was
        # never retrieved" at GC time, which would also lose the reason the
        # abandoned work failed.
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.debug(
                    "Abandoned worker finished with an error: %r",
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        raise
