#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct-Python chain-of-K MTP smoke + micro-bench for Gemma 4 12B.

Loads the Gemma 4 12B target via mlx-vlm's loader, injects Google's
official assistant sidecar via our Gemma 4 MTP inject path, then drives
``mtp_generate_step`` with configurable ``num_draft_tokens`` (K).

Usage:
    python scripts/chain_of_k_smoke.py \\
      --sidecar ~/rapid-mlx-staging/gemma-4-12B-it-assistant \\
      --ks 1 2 3 4 5 \\
      --runs 3 \\
      --max-tokens 128

Reports per-K + per-prompt: decode tok/s, accept ratio, tokens_saved,
first-run output token sequence hash (for cross-run determinism check).

Constraints:
    * Batch=1 only (matches mtp_max_batch_size=1 in gemma4_inject).
    * Greedy sampling only (temp=0).
    * Baseline is chain-of-1 with K=1 (byte-identical to pre-PR legacy).

Output format: JSON to stdout unless --markdown is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from typing import Any

# Prompts — mixed workload for the smoke + micro-bench.
_PROMPTS: list[tuple[str, str]] = [
    (
        "sky_blue_40w",
        "Explain in exactly 40 words why the sky appears blue at midday. "
        "Be concise; do not exceed 40 words.",
    ),
    (
        "robot_paint_story",
        "Tell a short story (about 100 words) about a robot who learns "
        "to paint. Include a beginning, middle, and end.",
    ),
    (
        "fib_python_code",
        "Write a Python function that computes the n-th Fibonacci number "
        "using memoization. Include type hints, a docstring, and a small "
        "example call. Keep it under 30 lines.",
    ),
    (
        "json_top3_nosql",
        "Return a JSON object describing the top 3 NoSQL databases by "
        "popularity, with fields name, category, primary_use_case, and "
        "year_released. No prose, just the JSON.",
    ),
]


def _load_model(sidecar: str, model_path: str = "mlx-community/gemma-4-12B-it-4bit"):
    """Load Gemma 4 12B target + inject Google assistant sidecar."""
    print(f"[smoke] Loading {model_path} target ...", file=sys.stderr)
    t0 = time.perf_counter()

    from vllm_mlx.models.gemma4_text import load_gemma4_text

    model, tokenizer = load_gemma4_text(model_path)
    load_t = time.perf_counter() - t0
    print(f"[smoke] Target loaded in {load_t:.1f}s.", file=sys.stderr)

    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )

    t0 = time.perf_counter()
    ok = inject_mtp_support(model, mtp_sidecar=sidecar)
    if not ok:
        raise RuntimeError(f"inject_mtp_support failed on sidecar {sidecar!r}")
    if not validate_mtp_support(model):
        raise RuntimeError("validate_mtp_support returned False after inject.")
    inj_t = time.perf_counter() - t0
    print(f"[smoke] MTP injected in {inj_t:.1f}s.", file=sys.stderr)

    return model, tokenizer


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_tokens: int,
    K: int,
    use_controller: bool = False,
    warmup: bool = False,
) -> dict[str, Any]:
    """Run one MTP generation and return telemetry."""
    import mlx.core as mx

    from vllm_mlx.spec_decode.mtp import (
        MTPAcceptCounter,
        clear_global_controller,
        install_global_controller,
    )
    from vllm_mlx.spec_decode.mtp.draft_k_controller import DraftKController
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    # Isolate per-run controller singleton state.
    clear_global_controller()
    if use_controller:
        # Default-ish controller with k_start = K (i.e., we're benching
        # a specific K seed under auto-tune).
        controller = DraftKController(
            k_min=1,
            k_max=5,
            k_start=K,
            window=48,
            upshift_threshold=0.82,
            downshift_threshold=0.55,
            cooldown=64,
        )
        install_global_controller(controller)

    # Encode via tokenizer. Use chat template when available so the
    # results are comparable to server-side calls.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                return_tensors=None,
            )
            # apply_chat_template returns a python list or numpy — normalize.
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            if isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
        except Exception:
            prompt_ids = tokenizer.encode(prompt_text)
    else:
        prompt_ids = tokenizer.encode(prompt_text)

    prompt_arr = mx.array(prompt_ids, mx.uint32)

    # If the wrapper exposes ``language_model``, drive that so
    # mtp_generate_step reaches the patched inner text model.
    inner_model = getattr(model, "language_model", model)

    counter = MTPAcceptCounter()

    # Warm-up: eval to compile kernels.
    _ = mx.array([1.0])
    mx.eval(_)

    t0 = time.perf_counter()
    n = 0
    tokens: list[int] = []
    gen = mtp_generate_step(
        prompt_arr,
        inner_model,
        max_tokens=max_tokens,
        temp=0.0,
        accept_counter=counter,
        num_draft_tokens=K,
    )
    for tok, _lp, _from_draft in gen:
        tokens.append(tok)
        n += 1
        if n >= max_tokens:
            break
    elapsed = time.perf_counter() - t0

    snap = counter.snapshot()
    tok_per_s = n / elapsed if elapsed > 0 else 0.0

    # Determinism hash: first 64 tokens' hex fingerprint.
    tok_hash = hashlib.sha256(
        b",".join(str(t).encode() for t in tokens[:64])
    ).hexdigest()[:16]

    if use_controller:
        from vllm_mlx.spec_decode.mtp.draft_k_controller import (
            get_global_controller,
        )

        ctrl = get_global_controller()
        ctrl_snap = ctrl.snapshot() if ctrl is not None else {}
    else:
        ctrl_snap = {}
    clear_global_controller()

    # Decode the tokens to a short preview for eyeball verification.
    try:
        preview = tokenizer.decode(tokens[:120])
    except Exception:
        preview = "[decode error]"

    return {
        "K": K,
        "prompt_len_tokens": len(prompt_ids),
        "output_len_tokens": n,
        "elapsed_seconds": round(elapsed, 3),
        "tok_per_second": round(tok_per_s, 2),
        "attempts": snap.attempts,
        "accepts": snap.accepts,
        "tokens_saved": snap.tokens_saved,
        "accept_ratio": round(snap.accept_ratio, 4),
        "tok_hash": tok_hash,
        "controller_snapshot": ctrl_snap,
        "output_preview": preview[:400],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", required=True)
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Draft-K values to sweep.",
    )
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--prompts",
        type=int,
        default=len(_PROMPTS),
        help="Number of prompts to sweep (from the fixed 4-prompt list).",
    )
    p.add_argument("--markdown", action="store_true")
    p.add_argument(
        "--auto-tune",
        action="store_true",
        help="Enable the auto-tune controller. K becomes k_start seed.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    prompts = _PROMPTS[: args.prompts]
    model, tokenizer = _load_model(args.sidecar)

    # Warmup: 1 short pass on the first prompt + K=1 to compile kernels.
    print("[smoke] Warmup pass ...", file=sys.stderr, flush=True)
    _ = _generate(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompts[0][1],
        max_tokens=16,
        K=1,
        use_controller=False,
        warmup=True,
    )

    all_results: list[dict[str, Any]] = []
    # Interleave the (prompt, K) matrix by run_idx to reduce thermal
    # bias — each run pass touches every (prompt, K) once before the
    # next run pass starts.
    for run_idx in range(args.runs):
        for prompt_id, prompt_text in prompts:
            for K in args.ks:
                print(
                    f"[smoke] run={run_idx} prompt={prompt_id} K={K} ...",
                    file=sys.stderr,
                    flush=True,
                )
                res = _generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_tokens=args.max_tokens,
                    K=K,
                    use_controller=args.auto_tune,
                )
                res["prompt_id"] = prompt_id
                res["run_idx"] = run_idx
                all_results.append(res)
                print(
                    f"[smoke]   -> {res['tok_per_second']} tok/s "
                    f"(n={res['output_len_tokens']}, "
                    f"accept={res['accept_ratio']}, "
                    f"hash={res['tok_hash']})",
                    file=sys.stderr,
                    flush=True,
                )

    if args.markdown:
        _print_markdown(all_results, prompts, args)
    else:
        print(
            json.dumps(
                {
                    "config": {
                        "sidecar": args.sidecar,
                        "ks": args.ks,
                        "runs": args.runs,
                        "max_tokens": args.max_tokens,
                        "auto_tune": args.auto_tune,
                    },
                    "results": all_results,
                },
                indent=2,
            )
        )
    return 0


def _print_markdown(
    results: list[dict[str, Any]],
    prompts: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    print("# Chain-of-K smoke report")
    print()
    print(f"- Sidecar: `{args.sidecar}`")
    print(f"- Ks swept: {args.ks}")
    print(f"- Runs per K: {args.runs}")
    print(f"- Max tokens: {args.max_tokens}")
    print(f"- Auto-tune controller: {args.auto_tune}")
    print()
    # Determinism check per (prompt, K) — all runs should have same hash.
    print("## Determinism (runs share the same first-64-token hash?)")
    print()
    print("| prompt | K | runs share hash? | hash |")
    print("| --- | --- | --- | --- |")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault((r["prompt_id"], r["K"]), []).append(r)
    for (pid, K), rs in sorted(grouped.items()):
        hashes = [r["tok_hash"] for r in rs]
        same = "YES" if len(set(hashes)) == 1 else "NO"
        print(f"| {pid} | {K} | {same} | {hashes[0]} |")
    print()

    # Speedup table — pooled tok/s per (prompt, K), speedup vs K=1.
    print("## Speedup (pooled tok/s per prompt vs K=1)")
    print()
    print(
        "| prompt | K=1 tok/s | K=1 accept | "
        + " | ".join(
            f"K={K} tok/s | K={K} accept | K={K} spd"
            for K in args.ks
            if K != 1
        )
        + " |"
    )
    hdr_sep = "| --- | --- | --- | " + " | ".join(
        "--- | --- | ---" for K in args.ks if K != 1
    ) + " |"
    print(hdr_sep)
    for pid, _ in prompts:
        row = f"| {pid} |"
        baseline = grouped.get((pid, 1), [])
        if not baseline:
            row += " (no K=1 baseline) |"
            k1_tps = None
        else:
            k1_tps = sum(r["tok_per_second"] for r in baseline) / len(baseline)
            k1_acc = sum(r["accept_ratio"] for r in baseline) / len(baseline)
            row += f" {k1_tps:.2f} | {k1_acc:.3f} |"
        for K in args.ks:
            if K == 1:
                continue
            grp = grouped.get((pid, K), [])
            if not grp:
                row += " -- | -- | -- |"
                continue
            tps = sum(r["tok_per_second"] for r in grp) / len(grp)
            acc = sum(r["accept_ratio"] for r in grp) / len(grp)
            spd = tps / k1_tps if k1_tps else 0.0
            row += f" {tps:.2f} | {acc:.3f} | {spd:.2f}x |"
        print(row)


if __name__ == "__main__":
    sys.exit(main())
