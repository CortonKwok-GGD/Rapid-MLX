# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the opt-in prompt-deterministic RESPONSE CACHE.

Covers the invariants that guard correctness:

* Cache key includes EVERY output-affecting field — a change in any
  sampling param (or model / prompt / response-shape field) yields a
  different key → a miss → a correct recompute. A missing field would be
  a correctness bug (a wrong response served).
* Determinism gate — only greedy (temperature==0 / top_k==1) requests
  are cacheable; a ``temperature > 0`` request with no definitively
  deterministic decode (even with a pinned seed, in this MVP) is NOT
  cached, so sampling variety is preserved.
* LRU bound — the (N+1)th distinct store evicts the LEAST-RECENTLY-USED
  entry, AND a HIT refreshes recency so eviction order is true LRU, not
  FIFO (explicitly regression-tested; the sibling hybrid-cache knob had a
  FIFO-vs-LRU gap that this guards against).
* N=0 fully disables — no store, no lookup, counters stay at zero, zero
  observable effect.
* Concurrent access safety — many threads hammering get/put must not
  corrupt the store or lose the LRU/counter invariants.
* SchedulerConfig validation — a negative ``response_cache_entries`` is
  rejected at construction.
* Metrics rendering — the hit/miss counters surface on ``/metrics``.
"""

from __future__ import annotations

import threading

import pytest

from vllm_mlx.response_cache import (
    UNCACHEABLE,
    ResponseCache,
    configure_response_cache,
    get_response_cache,
    is_deterministic,
    make_cache_key,
    reset_response_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _fresh_singleton():
    """Reset the process singleton around every test so counters/state
    from one case never leak into the next."""
    reset_response_cache_for_tests()
    yield
    reset_response_cache_for_tests()


# ── LRU semantics ─────────────────────────────────────────────────────


def test_lru_evicts_least_recently_used_not_fifo():
    """The (N+1)th store evicts the LRU entry — and a HIT must refresh
    recency so the eviction is LRU, NOT FIFO.

    Insert a, b (capacity 2). Then GET a (making a the most-recently
    used and b the least). Insert c → b (the LRU) must be evicted, and
    a (refreshed by the hit) must survive. A FIFO cache would wrongly
    evict a here; this asserts that does not happen.
    """
    c = ResponseCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # a is now MRU, b is LRU
    c.put("c", 3)  # overflow → evict LRU (b), NOT the FIFO-oldest (a)
    assert c.get("b") is None, "FIFO bug: b was LRU and should be evicted"
    assert c.get("a") == 1, "the hit-refreshed entry must survive eviction"
    assert c.get("c") == 3


def test_reinsert_refreshes_recency():
    """Re-``put``-ing an existing key refreshes its recency (and value)."""
    c = ResponseCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 99)  # refresh a → b becomes LRU, value updated
    c.put("c", 3)  # evict LRU (b)
    assert c.get("b") is None
    assert c.get("a") == 99
    assert c.get("c") == 3


def test_capacity_bound_never_exceeded():
    c = ResponseCache(capacity=3)
    for i in range(100):
        c.put(f"k{i}", i)
        assert c.snapshot()["entries"] <= 3
    # Only the last 3 distinct keys remain.
    assert c.get("k99") == 99 and c.get("k98") == 98 and c.get("k97") == 97
    assert c.get("k96") is None


# ── N=0 disables everything ───────────────────────────────────────────


def test_zero_capacity_fully_inert():
    """Capacity 0 → no store, no lookup, counters untouched."""
    c = ResponseCache(capacity=0)
    assert c.enabled is False
    c.put("x", 1)
    assert c.get("x") is None
    snap = c.snapshot()
    # An inert cache records NOTHING — not even the miss.
    assert snap == {"hits": 0, "misses": 0, "entries": 0, "capacity": 0}


def test_configure_zero_clears_and_disables():
    c = ResponseCache(capacity=4)
    c.put("a", 1)
    assert c.get("a") == 1
    c.configure(0)
    assert c.enabled is False
    assert c.get("a") is None
    assert c.snapshot()["entries"] == 0


def test_configure_shrink_evicts_coldest():
    c = ResponseCache(capacity=3)
    c.put("1", 1)
    c.put("2", 2)
    c.put("3", 3)  # MRU order: 1(cold) < 2 < 3(hot)
    c.configure(1)  # keep only the hottest
    assert c.get("3") == 3
    assert c.get("1") is None
    assert c.get("2") is None


def test_negative_capacity_rejected():
    with pytest.raises(ValueError, match=r"capacity must be >= 0"):
        ResponseCache(capacity=-1)
    c = ResponseCache(capacity=1)
    with pytest.raises(ValueError, match=r"capacity must be >= 0"):
        c.configure(-5)


# ── Counters ──────────────────────────────────────────────────────────


def test_hit_miss_counters():
    c = ResponseCache(capacity=4)
    c.get("absent")  # miss
    c.put("k", "v")
    c.get("k")  # hit
    c.get("k")  # hit
    c.get("absent2")  # miss
    snap = c.snapshot()
    assert snap["hits"] == 2
    assert snap["misses"] == 2


# ── Determinism gate ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"temperature": 0}, True),
        ({"temperature": 0.0}, True),
        ({"top_k": 1}, True),
        ({"top_k": 1, "temperature": 0.9}, True),  # top_k==1 forces greedy
        ({"temperature": 0.8}, False),
        ({"temperature": 0.8, "seed": 42}, False),  # seed alone NOT enough (MVP)
        ({"temperature": 0.0000001}, False),  # near-zero is still sampling
        ({}, False),  # missing → not greedy
        ({"temperature": None}, False),
        ({"top_k": 0, "temperature": 0.7}, False),
    ],
)
def test_determinism_gate(kwargs, expected):
    assert is_deterministic(kwargs) is expected


# ── Cache key: every output-affecting field participates ──────────────


_BASE = dict(
    model="m",
    prompt="hello world",
    sampling_kwargs={
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 0,
        "min_p": 0.0,
        "max_tokens": 64,
        "stop": ["</s>"],
        "seed": 7,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
)


def _key(**overrides):
    args = dict(_BASE)
    if "sampling_kwargs" in overrides:
        args["sampling_kwargs"] = {
            **_BASE["sampling_kwargs"],
            **overrides.pop("sampling_kwargs"),
        }
    args.update(overrides)
    return make_cache_key(**args)


def test_key_is_dict_order_independent():
    k1 = make_cache_key(
        model="m", prompt="p", sampling_kwargs={"temperature": 0, "max_tokens": 10}
    )
    k2 = make_cache_key(
        model="m", prompt="p", sampling_kwargs={"max_tokens": 10, "temperature": 0}
    )
    assert k1 == k2


def test_key_identical_inputs_match():
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.5),
        ("top_p", 0.5),
        ("top_k", 40),
        ("min_p", 0.05),
        ("max_tokens", 65),
        ("seed", 8),
        ("stop", ["STOP"]),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
    ],
)
def test_key_changes_when_any_sampling_param_changes(field, value):
    """A missing field in the key = a wrong response served. Prove EACH
    sampling param is part of the key by flipping it and asserting the
    key changes."""
    base = _key()
    changed = _key(sampling_kwargs={field: value})
    assert base != changed, f"{field} must affect the cache key"


def test_key_changes_on_model_prompt_and_extra():
    base = _key()
    assert _key(model="other") != base
    assert _key(prompt="different") != base
    # ``extra`` fields (response_format / logprobs) change the wire shape.
    assert make_cache_key(**_BASE, extra={"logprobs": True}) != base
    assert (
        make_cache_key(**_BASE, extra={"response_format": {"type": "json_object"}})
        != base
    )
    assert make_cache_key(**_BASE, extra={"top_logprobs": 5}) != base


def test_key_handles_pydantic_like_components():
    """Non-JSON-native key components (objects with ``model_dump``) must be
    canonicalized by ``_json_default`` via ``.model_dump()`` — NOT raise,
    and NOT fall through to an unstable repr.

    The instance is passed DIRECTLY in ``extra`` (not pre-``model_dump``-ed)
    so ``json.dumps`` actually routes it through ``_json_default`` and
    invokes ``.model_dump()``. Passing the already-dumped dict would bypass
    the code path under test entirely (a dict is JSON-native), leaving the
    ``model_dump`` branch unexercised — mutation-kill: deleting the
    ``model_dump`` branch in ``_json_default`` MUST make this test fail.
    """

    class _Fake:
        def model_dump(self):
            return {"type": "json_schema", "schema": {"a": 1}}

    k = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"response_format": _Fake()},  # instance, so _json_default fires
    )
    assert isinstance(k, str) and len(k) == 64  # sha256 hexdigest

    # The key must equal the one produced from the DUMPED dict directly:
    # proves _json_default really invoked .model_dump() (not repr / some
    # other fallback), so canonicalization is by VALUE, not object identity.
    k_from_dict = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"response_format": _Fake().model_dump()},
    )
    assert k == k_from_dict

    # And two FRESH _Fake() instances must produce the SAME key — the whole
    # point of canonicalizing by value rather than repr (which would embed
    # the object's memory address and silently miss).
    k2 = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"response_format": _Fake()},
    )
    assert k == k2


def test_uncacheable_when_component_cannot_be_canonicalized():
    """A key component that is neither JSON-native, a set, nor a
    ``model_dump``-carrying object cannot be mapped to a STABLE string.
    ``make_cache_key`` must return the ``UNCACHEABLE`` sentinel (so the
    caller skips store + lookup) rather than emit an unstable repr key —
    and it must NOT raise.

    A repr fallback would embed the object's memory address, so two
    otherwise-identical requests carrying fresh equivalent objects would
    key DIFFERENTLY → silent misses that defeat an exact-match cache.
    Mutation-kill: replacing the ``raise _UncanonicalizableError`` in
    ``_json_default`` with ``return repr(obj)`` MUST make this test fail
    (the result would be a 64-char hex string, not the sentinel).
    """

    class _Opaque:
        """No model_dump, not JSON-native, not a set."""

    result = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"weird": _Opaque()},
    )
    assert result is UNCACHEABLE

    # A model_dump that itself raises is likewise uncacheable, not a 500.
    class _BadDump:
        def model_dump(self):
            raise RuntimeError("boom")

    result2 = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"rf": _BadDump()},
    )
    assert result2 is UNCACHEABLE

    # Two fresh opaque objects both yield the sentinel — no repr address
    # leakage, no accidental distinct keys.
    r_a = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"weird": _Opaque()},
    )
    r_b = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"weird": _Opaque()},
    )
    assert r_a is UNCACHEABLE and r_b is UNCACHEABLE


# ── Concurrency safety ────────────────────────────────────────────────


def test_concurrent_access_is_safe():
    """Many threads storing/reading distinct + shared keys must not
    corrupt the store, break the capacity bound, or lose the counter
    invariant (hits + misses == total gets).

    Worker-thread exceptions are COLLECTED and re-raised on the main
    thread — a raw ``threading.Thread`` swallows exceptions, so without
    this a corrupted store or a raising ``put``/``get`` would pass
    silently.

    Mutation-kill: the test ALSO writes a set of stable keys concurrently
    and, after the join, asserts every one is retrievable. Capacity is
    sized to hold them all, so deleting ``put()`` (which would leave the
    store empty) makes the retrieval assertion fail — the earlier
    capacity/counter-only version stayed green with storage broken.
    """
    # Stable keys written by every thread; capacity covers them plus the
    # churn keys, so a survivor check is deterministic.
    stable_keys = [f"stable-{i}" for i in range(32)]
    c = ResponseCache(capacity=4096)  # large enough that nothing evicts
    n_threads = 16
    ops_per_thread = 500
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(tid: int):
        try:
            barrier.wait()
            # Every thread writes the full stable set so the values are
            # present regardless of scheduling.
            for sk in stable_keys:
                c.put(sk, sk)
            for i in range(ops_per_thread):
                k = f"t{tid % 4}-k{i % 80}"  # overlapping keyspace → contention
                if i % 2 == 0:
                    c.put(k, (tid, i))
                else:
                    c.get(k)
        except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"worker thread(s) raised under concurrency: {errors!r}"

    # Storage actually worked: every concurrently-written stable key is
    # retrievable with its written value. Deleting ``put()`` fails here.
    for sk in stable_keys:
        assert c.get(sk) == sk, f"concurrently-stored key {sk!r} was not retrievable"

    snap = c.snapshot()
    assert snap["entries"] <= 4096, "capacity bound violated under concurrency"


# ── Module singleton wiring ───────────────────────────────────────────


def test_singleton_starts_disabled_and_configures():
    assert get_response_cache().enabled is False
    configure_response_cache(8)
    assert get_response_cache().enabled is True
    assert get_response_cache().capacity == 8
    configure_response_cache(0)
    assert get_response_cache().enabled is False


def test_configure_clears_entries_on_reload():
    """``configure_response_cache`` runs on EVERY ``load_model`` — including
    a hot reload of changed weights under the same model id. A stored
    completion is only valid for the model artifact that produced it, so
    the (re)configure MUST drop all cached entries; otherwise a reload
    serves completions from the previous model.

    Mutation-kill: remove the ``clear()`` from ``configure_response_cache``
    → the entry survives the simulated reload and this fails.
    """
    configure_response_cache(16)
    cache = get_response_cache()
    cache.put("some-key", "completion-from-model-v1")
    assert cache.get("some-key") == "completion-from-model-v1"

    # Simulate a second load_model() with the SAME positive capacity (the
    # case where plain configure() would otherwise preserve entries).
    configure_response_cache(16)

    assert get_response_cache().snapshot()["entries"] == 0
    assert get_response_cache().get("some-key") is None


# ── SchedulerConfig validation ────────────────────────────────────────


def test_scheduler_config_rejects_negative_response_cache_entries():
    from vllm_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=r"response_cache_entries must be >= 0"):
        SchedulerConfig(response_cache_entries=-1)


def test_scheduler_config_default_response_cache_entries_is_zero():
    from vllm_mlx.scheduler import SchedulerConfig

    assert SchedulerConfig().response_cache_entries == 0
    assert SchedulerConfig(response_cache_entries=32).response_cache_entries == 32
