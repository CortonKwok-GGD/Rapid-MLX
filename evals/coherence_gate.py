#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Output-coherence release gate — feeds golden prompts through a REAL running
``rapid-mlx serve`` and asserts the generated text is coherent (#1247).

This is the serve-path half of the coherence gate; the pure predicates and the
garbage detector live in :mod:`vllm_mlx.coherence` and are unit-tested in
ordinary CI. This script requires a server to already be listening (it does
**not** boot one) — mirroring ``evals/run_eval.py`` and
``tests/integrations/test_anthropic_sdk.py``. The release gauntlet
(``scripts/release_check_m3.sh``) boots ``rapid-mlx serve <model> --no-thinking``
and exports ``RAPID_MLX_BASE_URL``; this gate reads that env by default.

Usage
-----
    # against the server the release gauntlet already booted:
    python evals/coherence_gate.py

    # or point it explicitly:
    python evals/coherence_gate.py --base-url http://127.0.0.1:8000/v1

Exit codes:
    0 — every golden case coherent + correct
    1 — one or more cases failed (garbage, wrong answer, or think-leak)
    2 — no server reachable at the base URL
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

# Make ``vllm_mlx`` importable when run as ``python evals/coherence_gate.py``
# from a bare checkout (sys.path[0] is evals/, not the repo root). Harmless when
# rapid-mlx is already installed — an editable/site-packages copy still resolves
# first only if this insert is skipped, but preferring the checkout is correct
# for a gate that must test THIS tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_mlx.coherence import GOLDEN, GoldenCase, evaluate_case  # noqa: E402

_DEFAULT_BASE_URL = os.environ.get("RAPID_MLX_BASE_URL", "http://127.0.0.1:8000/v1")


def _generate(base_url: str, case: GoldenCase, *, timeout: float) -> str:
    """Non-streaming completion for ``case`` at temperature 0. Returns the
    visible assistant text (empty string if the model returned no content)."""
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": case.max_tokens,
        "temperature": 0.0,
        "stream": False,
        # Match the gauntlet's --no-thinking boot: the gate measures answer
        # coherence, not thinking-mode behavior. The no-think-leak case still
        # asserts no raw <think> tag survives into the visible message.
        "enable_thinking": False,
    }
    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    return content or ""


def _server_reachable(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL (default: $RAPID_MLX_BASE_URL or "
        "http://127.0.0.1:8000/v1)",
    )
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="per-request timeout (s)"
    )
    args = ap.parse_args()

    base_url = args.base_url
    print("=" * 60)
    print("  output-coherence gate (#1247)")
    print(f"  base_url: {base_url}")
    print(f"  cases:    {len(GOLDEN)}")
    print("=" * 60)

    if not _server_reachable(base_url):
        print(
            f"ERROR: no rapid-mlx server reachable at {base_url}. "
            "Start one with: rapid-mlx serve <model> --port 8000",
            file=sys.stderr,
        )
        return 2

    failures: list[tuple[str, str, str]] = []  # (id, reason, snippet)
    passed_n = 0
    transport_failed = False
    for case in GOLDEN:
        try:
            text = _generate(base_url, case, timeout=args.timeout)
            passed, reason = evaluate_case(case, text)
        except httpx.TransportError as exc:
            passed, reason = False, f"server transport error: {exc}"
            transport_failed = True
            text = ""
        except Exception as exc:  # server/protocol error mid-run -> a gate failure
            passed, reason = False, f"request error: {exc}"
            text = ""

        status = "PASS" if passed else "FAIL"
        snippet = " ".join(text.split())[:80]
        print(f"  [{status}] {case.id:<16} {reason}")
        if passed:
            passed_n += 1
        else:
            print(f"           output: {snippet!r}")
            failures.append((case.id, reason, snippet))
        if transport_failed:
            break

    print("=" * 60)
    print(f"  {passed_n}/{len(GOLDEN)} coherent")
    print("=" * 60)

    if transport_failed:
        print(
            "ERROR: the rapid-mlx server became unreachable while the "
            "coherence gate was running.",
            file=sys.stderr,
        )
        return 2

    if failures:
        print(
            "COHERENCE GATE FAILED — the served model produced incoherent or "
            "incorrect output. This is the class that shipped as garbage in "
            "#1234; do NOT release.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
