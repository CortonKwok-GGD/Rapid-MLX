# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests: the response-cache hit/miss counters on ``/metrics``.

Mirrors ``tests/test_metrics_route.py`` — no real engine, just the
metrics router mounted on a throwaway FastAPI app. The counters live in
the ``vllm_mlx.response_cache`` module singleton (NOT the engine), so
they must render even when the engine is absent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def metrics_client():
    from vllm_mlx.config import reset_config
    from vllm_mlx.response_cache import reset_response_cache_for_tests
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    _reset_accumulator_for_tests()
    reset_response_cache_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()
    reset_response_cache_for_tests()


def test_response_cache_counters_present_even_without_engine(metrics_client):
    """Both series render (at zero) even when the engine is not loaded —
    the counters are engine-independent module state."""
    metrics_client.cfg.engine = None
    body = metrics_client.client.get("/metrics").text
    assert "rapid_mlx_response_cache_hits_total" in body
    assert "rapid_mlx_response_cache_misses_total" in body
    # Disabled cache → both at zero.
    assert "rapid_mlx_response_cache_hits_total 0" in body
    assert "rapid_mlx_response_cache_misses_total 0" in body


def test_response_cache_counters_reflect_singleton_state(metrics_client):
    """Driving the singleton's counters must surface on the scrape."""
    from vllm_mlx.response_cache import get_response_cache

    c = get_response_cache()
    c.configure(4)
    c.get("miss-a")  # miss
    c.put("k", object())
    c.get("k")  # hit
    c.get("k")  # hit
    c.get("miss-b")  # miss

    body = metrics_client.client.get("/metrics").text
    assert "rapid_mlx_response_cache_hits_total 2" in body
    assert "rapid_mlx_response_cache_misses_total 2" in body


def test_response_cache_counters_have_help_and_type_lines(metrics_client):
    """Prometheus HELP + TYPE metadata must accompany each series."""
    body = metrics_client.client.get("/metrics").text
    assert "# TYPE rapid_mlx_response_cache_hits_total counter" in body
    assert "# TYPE rapid_mlx_response_cache_misses_total counter" in body
    assert "# HELP rapid_mlx_response_cache_hits_total" in body
    assert "# HELP rapid_mlx_response_cache_misses_total" in body
