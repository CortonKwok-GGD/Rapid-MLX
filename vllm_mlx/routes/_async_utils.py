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
      cached engine, destroying the one-at-a-time memory guarantee the
      lock exists to provide, and
    * a ``finally`` block would delete the temp file the abandoned
      worker is still reading from.

    So on cancellation we wait for the worker to actually finish before
    propagating, keeping "lock held" and "input path alive" true for
    exactly as long as the thread is running. The client is already gone,
    so the extra wait costs nothing user-visible, and it is bounded by
    the engine's own work.
    """
    task = asyncio.ensure_future(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Drain the worker (ignoring its outcome — nobody is listening)
        # before letting the cancellation unwind our lock + cleanup.
        #
        # The drain has to be a SHIELDED LOOP, not a bare ``await task``.
        # A bare await is itself cancellable, so a second ``Task.cancel()``
        # — a shutdown signal, a supervisor giving up on a hung request —
        # would interrupt the drain and hand control back to the caller
        # while the thread is still running. That releases the lock and
        # unlinks the temp file underneath a live worker, i.e. exactly the
        # failure this helper exists to prevent, just one cancel later.
        #
        # ``asyncio.to_thread`` cannot actually be interrupted, so the
        # thread WILL finish; looping until ``task.done()`` simply refuses
        # to return before it has. Repeated cancels are absorbed here and
        # the original CancelledError is re-raised at the end, so the
        # caller's cancellation semantics are unchanged.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    break
                logger.debug("Ignoring cancellation while draining an abandoned worker")
            except Exception:
                logger.debug("Abandoned worker finished with an error", exc_info=True)
                break
        # Retrieve the outcome even when the loop never ran (the worker had
        # already finished when cancellation arrived). Without this the
        # exception stays unretrieved and asyncio logs "Task exception was
        # never retrieved" at GC time — noise that also loses the reason the
        # abandoned work failed.
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.debug(
                    "Abandoned worker finished with an error: %r", exc, exc_info=exc
                )
        raise
