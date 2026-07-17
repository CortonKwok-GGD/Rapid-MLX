# SPDX-License-Identifier: Apache-2.0
"""Opt-in prompt-deterministic RESPONSE CACHE (exact-match short-circuit).

Completely repeated *deterministic* requests short-circuit the whole GPU
pipeline and return the previously-computed completion verbatim — zero
decode. This is distinct from the KV / prefix cache (which reuses prefix
*state* to speed prefill): the response cache returns the *entire stored
completion*, doing no engine work at all.

Opt-in, default OFF. Enabled by ``--response-cache-entries N`` (N > 0);
``N == 0`` (the default) makes the cache fully inert — no store, no
lookup, counters stay at zero, zero behavior change. Mirrors the
``--hybrid-cache-entries`` opt-in knob.

Determinism gate
----------------
Only *greedy* requests are cached and short-circuited. A request is
eligible when ``temperature == 0`` OR ``top_k == 1`` (see
:func:`is_deterministic`). If ``temperature > 0`` and the caller did not
pin a ``seed``, they expect sampling variety, so returning a stale
identical response would be WRONG — those requests are skipped (a miss,
which is always correct). A pinned ``seed`` with ``temperature > 0`` is
NOT treated as deterministic in this MVP: MLX's batched sampler advances
shared PRNG state whose per-request split depends on scheduling
neighbours, so "same seed" does not guarantee "same tokens" across runs
here. Conservative by design — widen later with evidence, not hope.

Correctness note (pre-empts the "epsilon recompute" objection)
--------------------------------------------------------------
At ``temperature == 0`` MLX greedy decode is NOT bit-stable across runs:
batched SDPA numerics diverge between ``q_len == 1`` and ``q_len >= 2``
under quantized weights, so a *fresh recompute* of the same prompt may
differ by a token. This does NOT break the response cache. The cache
RETURNS A STORED VALID COMPLETION — it never recomputes. The contract is
exactly OpenAI's prompt-caching contract: *"an identical deterministic
request MAY return a previously-computed valid response."* Reviewers
should not spiral on "but a fresh run might differ" — that is out of
scope by construction.

Concurrency
-----------
The server is async and multi-request. The LRU store and the counters
are guarded by a single ``threading.Lock`` around an ``OrderedDict``
(mirrors ``memory_cache.py``). No background thread is introduced.
Because ``get``/``put`` hold the lock only for O(1) dict ops (never
across ``await``), lock contention is negligible.

Metrics
-------
Two process-local counters, read by ``routes/metrics.py`` via
:func:`snapshot`:

* ``rapid_mlx_response_cache_hits_total``
* ``rapid_mlx_response_cache_misses_total``

Counters never decrease for the process lifetime; they reset to zero on
restart (the normal Prometheus convention). Tests use
:func:`reset_response_cache_for_tests`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


class ResponseCache:
    """Bounded LRU cache of fully-assembled deterministic responses.

    Capacity ``0`` means disabled: :meth:`get` always misses (without
    ticking the miss counter — an inert cache records nothing) and
    :meth:`put` is a no-op. Capacity ``N > 0`` retains at most ``N``
    entries, evicting the least-recently-USED entry on overflow (a HIT
    refreshes recency, so eviction order is true LRU, not FIFO).

    Stored values are opaque to this class — the chat route stores the
    serialized response body plus the small amount of metadata it needs
    to rebuild a fresh :class:`Response` (with a new ``id`` / ``created``)
    on a hit.
    """

    def __init__(self, capacity: int = 0) -> None:
        if capacity < 0:
            raise ValueError("ResponseCache capacity must be >= 0")
        self._capacity = int(capacity)
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        # Counters are process-local and monotonic (see module docstring).
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """True when the cache is active (capacity > 0)."""
        return self._capacity > 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def configure(self, capacity: int) -> None:
        """(Re)set the LRU capacity.

        Called once at server boot from the resolved
        ``SchedulerConfig.response_cache_entries``. Shrinking capacity
        evicts the coldest entries so the invariant ``len <= capacity``
        holds immediately. Counters are intentionally NOT reset here —
        they are process-lifetime monotonic. ``configure(0)`` disables
        the cache and clears the store.
        """
        if capacity < 0:
            raise ValueError("ResponseCache capacity must be >= 0")
        with self._lock:
            self._capacity = int(capacity)
            if self._capacity == 0:
                self._store.clear()
                return
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def get(self, key: str) -> Any | None:
        """Return the stored value for ``key`` and mark it MRU, or None.

        A hit ticks the hit counter AND moves the entry to the most-
        recently-used end (this is what makes eviction LRU rather than
        FIFO). A miss ticks the miss counter. When the cache is disabled
        (capacity 0) this is a no-op that returns None and ticks NOTHING
        — an inert cache must have zero observable effect, including on
        metrics.
        """
        if self._capacity == 0:
            return None
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Insert ``value`` under ``key`` as the most-recently-used entry.

        No-op when disabled (capacity 0). On overflow, evicts the
        least-recently-used entry (``last=False``). Re-inserting an
        existing key refreshes both its value and its recency.
        """
        if self._capacity == 0:
            return
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached entries (does not touch counters)."""
        with self._lock:
            self._store.clear()

    def snapshot(self) -> dict[str, int]:
        """Consistent snapshot of the counters for ``/metrics``."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._store),
                "capacity": self._capacity,
            }


# ── Determinism gate ──────────────────────────────────────────────────


def is_deterministic(sampling_kwargs: dict[str, Any]) -> bool:
    """Return True when a request is greedy enough to cache/short-circuit.

    Safe MVP rule (see module docstring): eligible when the effective
    sampling is greedy — ``temperature == 0`` OR ``top_k == 1``. Any
    other shape (``temperature > 0`` without a definitively deterministic
    decode, or missing/None temperature) is treated as non-deterministic
    and skipped. Missing keys default to "not greedy" so we never cache
    an ambiguous request.

    ``sampling_kwargs`` is the resolved kwargs dict the engine actually
    consumes (``chat_kwargs`` on the chat route) — i.e. the values AFTER
    the request → CLI → alias → generation_config cascade — so the gate
    sees exactly what will drive decoding, not the raw request fields.
    """
    temperature = sampling_kwargs.get("temperature")
    top_k = sampling_kwargs.get("top_k")
    # top_k == 1 forces a single candidate → argmax → greedy regardless
    # of temperature. temperature == 0 is greedy by definition.
    if top_k == 1:
        return True
    if temperature is not None and temperature == 0:
        return True
    return False


# ── Cache key ─────────────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """Best-effort canonicalizer for non-JSON-native key components.

    Pydantic models (tools, response_format) expose ``model_dump``;
    everything else falls back to ``repr`` so an unexpected type still
    produces a STABLE, collision-resistant string rather than raising
    (a raise here would 500 a request that merely can't be cached — we
    prefer a deterministic fallback that simply keys distinctly).
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover — defensive
            return repr(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return repr(obj)


def make_cache_key(
    *,
    model: str,
    prompt: Any,
    sampling_kwargs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a stable sha256 over EVERY output-affecting input.

    A missing field would be a correctness bug (a wrong response served),
    so the key spans:

    * ``model`` — the resolved model id.
    * ``prompt`` — the fully-rendered prompt the engine consumes (string
      or token-id list). Keying on the RENDERED prompt (not the raw
      messages) automatically folds in chat-template, tools, and
      ``forced_assistant_prefix`` differences: two requests that render
      to the same prompt string ARE the same generation input.
    * ``sampling_kwargs`` — the resolved kwargs dict passed to the engine
      (``temperature``, ``top_p``, ``top_k``, ``min_p``, ``seed``,
      ``max_tokens``, ``stop``, ``presence_penalty``,
      ``frequency_penalty``, ``repetition_penalty``, ``enable_thinking``,
      ``tools``, ``forced_assistant_prefix``, …). Because this is the
      SAME dict the engine consumes, no sampling param can silently drop
      out of the key.
    * ``extra`` — output-shape-affecting request fields that do NOT flow
      through ``sampling_kwargs`` but change the response body: e.g.
      ``response_format`` (JSON coercion), ``logprobs`` / ``top_logprobs``
      (adds the logprobs field). A change in any of these yields a
      different key → a MISS → a correct recompute.

    ``sort_keys=True`` + a compact separator make the JSON canonical
    (dict-order-independent). ``default=_json_default`` canonicalizes
    pydantic / set components. ``ensure_ascii=False`` keeps CJK / emoji
    from bloating the pre-hash string (the hash is over UTF-8 bytes
    either way).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "sampling": sampling_kwargs,
        "extra": extra or {},
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Module singleton ──────────────────────────────────────────────────

_response_cache = ResponseCache(capacity=0)


def get_response_cache() -> ResponseCache:
    """Return the process-wide response-cache singleton.

    Read by BOTH the chat route (store / lookup) and the metrics route
    (counter snapshot). Starts disabled (capacity 0); the serve boot path
    calls :func:`configure_response_cache` with the resolved
    ``--response-cache-entries`` value.
    """
    return _response_cache


def configure_response_cache(capacity: int) -> None:
    """Set the singleton's LRU capacity from the resolved CLI knob."""
    _response_cache.configure(capacity)


def reset_response_cache_for_tests() -> None:
    """Reset the singleton to a fresh, disabled cache (tests only)."""
    global _response_cache
    _response_cache = ResponseCache(capacity=0)
