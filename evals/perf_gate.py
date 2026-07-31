#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode-throughput perf-regression gate — measures generation speed against a
REAL running ``rapid-mlx serve`` and (optionally) fails when it drops below a
reviewed floor.

This is the perf half of the release gate that the Tier-1 agent smoke
(``tests/integrations/agent_smoke.sh``) runs on the SAME warm serve it already
booted for the agentic checks — so there is no second model load. Like
``evals/coherence_gate.py`` it requires a server to already be listening (it does
**not** boot one) and reads ``$RAPID_MLX_BASE_URL`` by default.

Why a long, fixed generation
----------------------------
A hybrid MoE model's tokens/sec on a SHORT response is dominated by prefill and
kernel-warmup noise (#284), so a short-prompt measurement is not a reliable
regression signal. This gate uses a short prompt and a LONG generation
(``--max-tokens``, default 512) so decode dominates the wall clock; on a fixed
workload at temperature 0 the end-to-end tokens/sec is a stable proxy for decode
throughput. A small warmup request precedes the measured one so a cold
long-context kernel does not skew the number.

Baseline is a reviewed human decision, never automatic
------------------------------------------------------
This gate NEVER invents a baseline. With no floor it runs ADVISORY: it prints the
measured tokens/sec and exits 0, so the first Studio run yields the number to
review. Enforcement turns on only when a floor is supplied via ``--min-tps`` or
``$RAPID_MLX_PERF_MIN_TPS`` (set it to the reviewed number, e.g. ~85% of the
observed warm rate to absorb run-to-run variance).

Usage
-----
    # advisory (prints tokens/sec, always exits 0):
    python evals/perf_gate.py --base-url http://127.0.0.1:8000/v1

    # enforcing (fails if below the reviewed floor):
    RAPID_MLX_PERF_MIN_TPS=19.5 python evals/perf_gate.py

Exit codes:
    0 — measured tokens/sec >= floor, OR advisory mode (no floor set)
    1 — a floor was set and the measured tokens/sec fell below it (regression)
    2 — no server reachable, or the response lacked usable token accounting
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

_DEFAULT_BASE_URL = os.environ.get("RAPID_MLX_BASE_URL", "http://127.0.0.1:8000/v1")

# Short prompt, long deterministic generation — decode dominates the wall clock.
_DEFAULT_PROMPT = (
    "Write a thorough technical explanation of how a modern CPU memory cache "
    "works. Cover the L1/L2/L3 hierarchy, cache lines, associativity, write-back "
    "vs write-through, and eviction policies such as LRU. Be detailed and precise."
)


class InvalidServerResponseError(RuntimeError):
    """The server replied, but without usable token accounting."""


def _server_reachable(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _complete(
    base_url: str, prompt: str, *, max_tokens: int, timeout: float
) -> tuple[int, float]:
    """One non-streaming completion at temperature 0. Returns
    ``(completion_tokens, elapsed_seconds)``. Raises on a malformed reply."""
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        # Match the gauntlet's --no-thinking boot: measure answer-token decode,
        # not thinking-mode expansion.
        "enable_thinking": False,
    }
    start = time.monotonic()
    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout
    )
    elapsed = time.monotonic() - start
    resp.raise_for_status()
    try:
        data = resp.json()
        completion_tokens = int(data["usage"]["completion_tokens"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidServerResponseError(
            "response lacked usage.completion_tokens — cannot measure throughput"
        ) from exc
    return completion_tokens, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL (default: $RAPID_MLX_BASE_URL or "
        "http://127.0.0.1:8000/v1)",
    )
    ap.add_argument(
        "--min-tps",
        type=float,
        default=(
            float(os.environ["RAPID_MLX_PERF_MIN_TPS"])
            if os.environ.get("RAPID_MLX_PERF_MIN_TPS")
            else None
        ),
        help="reviewed decode tokens/sec floor; below it the gate fails. "
        "Default: $RAPID_MLX_PERF_MIN_TPS, else advisory-only.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="generation length for the measured request (default: 512)",
    )
    ap.add_argument(
        "--timeout", type=float, default=180.0, help="per-request timeout (s)"
    )
    args = ap.parse_args()

    base_url = args.base_url
    print("=" * 60)
    print("  perf-regression gate (decode throughput)")
    print(f"  base_url: {base_url}")
    floor_str = (
        f"{args.min_tps:.2f} tok/s" if args.min_tps is not None else "(advisory)"
    )
    print(f"  floor: {floor_str}   max_tokens: {args.max_tokens}")
    print("=" * 60)

    if not _server_reachable(base_url):
        print(
            f"ERROR: no rapid-mlx server reachable at {base_url}. "
            "Start one with: rapid-mlx serve <model> --port 8000",
            file=sys.stderr,
        )
        return 2

    try:
        # Warm the long-context decode path so the measured run is not skewed by
        # a first-touch kernel compile (the agent smoke already exercised the
        # serve, so this is usually a no-op).
        _complete(base_url, "Say ready.", max_tokens=8, timeout=args.timeout)
        tokens, elapsed = _complete(
            base_url, _DEFAULT_PROMPT, max_tokens=args.max_tokens, timeout=args.timeout
        )
    except (httpx.HTTPError, InvalidServerResponseError) as exc:
        print(f"ERROR: perf measurement failed: {exc}", file=sys.stderr)
        return 2

    if tokens <= 0 or elapsed <= 0:
        print(
            f"ERROR: unusable measurement (tokens={tokens}, elapsed={elapsed:.3f}s).",
            file=sys.stderr,
        )
        return 2

    tps = tokens / elapsed
    print(f"  measured: {tokens} tokens in {elapsed:.2f}s -> {tps:.2f} tok/s")
    print("=" * 60)

    if args.min_tps is None:
        print(
            "  ADVISORY: no floor set (RAPID_MLX_PERF_MIN_TPS unset) — record this "
            "number, review it, then set the floor to enforce."
        )
        return 0

    if tps < args.min_tps:
        print(
            f"PERF GATE FAILED — {tps:.2f} tok/s is below the reviewed floor of "
            f"{args.min_tps:.2f} tok/s. The served model regressed; do NOT release.",
            file=sys.stderr,
        )
        return 1

    print(f"  PASS: {tps:.2f} tok/s >= floor {args.min_tps:.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
