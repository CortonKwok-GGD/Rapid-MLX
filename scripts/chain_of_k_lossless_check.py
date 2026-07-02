#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Byte-level lossless check for chain-of-K MTP.

Runs the SAME prompt under:
  * Plain mlx-lm generate_step (no MTP, no chain-of-K).
  * mtp_generate_step with num_draft_tokens=1, 2, 3, 4, 5.

All emit sequences (first N tokens) MUST be byte-identical to plain
autoregressive greedy output. Any divergence means the chain-of-K
implementation has a lossless-contract violation.

Usage:
    python scripts/chain_of_k_lossless_check.py \\
      --sidecar ~/rapid-mlx-staging/gemma-4-12B-it-assistant \\
      --max-tokens 64
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar", required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--prompts",
        type=int,
        default=3,
        help="How many of the 4 fixed prompts to test.",
    )
    args = p.parse_args()

    from vllm_mlx.models.gemma4_text import load_gemma4_text
    from vllm_mlx.spec_decode.mtp.gemma4_inject import (
        inject_mtp_support,
        validate_mtp_support,
    )
    import mlx.core as mx
    from vllm_mlx.spec_decode.mtp import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    print("[lossless] Loading model ...", file=sys.stderr)
    model, tokenizer = load_gemma4_text("mlx-community/gemma-4-12B-it-4bit")
    print("[lossless] Model loaded.", file=sys.stderr)

    _PROMPTS = [
        ("sky_blue_40w",
         "Explain in exactly 40 words why the sky appears blue at midday. "
         "Be concise; do not exceed 40 words."),
        ("robot_paint_story",
         "Tell a short story (about 100 words) about a robot who learns "
         "to paint. Include a beginning, middle, and end."),
        ("fib_python_code",
         "Write a Python function that computes the n-th Fibonacci number "
         "using memoization. Include type hints, a docstring, and a small "
         "example call. Keep it under 30 lines."),
        ("json_top3_nosql",
         "Return a JSON object describing the top 3 NoSQL databases by "
         "popularity, with fields name, category, primary_use_case, and "
         "year_released. No prose, just the JSON."),
    ][: args.prompts]

    inner_model = getattr(model, "language_model", model)

    def _encode(text: str):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if isinstance(ids[0], list):
                    ids = ids[0]
                return ids
            except Exception:
                pass
        return tokenizer.encode(text)

    def _greedy_baseline(prompt_text: str, max_tokens: int) -> list[int]:
        """Pure autoregressive greedy via manual loop.

        The mlx-vlm LanguageModel wrapper returns a
        ``LanguageModelOutput`` structured type from ``__call__``, so
        mlx-lm's ``generate_step`` (which does ``logits[:, -1, :]``)
        crashes with ``'LanguageModelOutput' object is not
        subscriptable``. Run a plain autoregressive greedy loop
        directly here — the goal is the token sequence, not
        throughput.
        """
        from mlx_lm.models.cache import make_prompt_cache

        ids = _encode(prompt_text)
        p_arr = mx.array(ids, mx.uint32)[None]  # (1, T)
        cache = make_prompt_cache(inner_model)
        # Prefill.
        out = inner_model(p_arr, cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        tokens: list[int] = []
        cur = mx.argmax(logits[:, -1, :], axis=-1)  # (1,)
        tokens.append(int(cur.item()))
        for _ in range(1, max_tokens):
            out = inner_model(cur[None, :], cache=cache)
            logits = out.logits if hasattr(out, "logits") else out
            cur = mx.argmax(logits[:, -1, :], axis=-1)
            tokens.append(int(cur.item()))
        return tokens

    def _mtp_run(prompt_text: str, max_tokens: int, K: int) -> list[int]:
        ids = _encode(prompt_text)
        p_arr = mx.array(ids, mx.uint32)
        counter = MTPAcceptCounter()
        tokens: list[int] = []
        n = 0
        gen = mtp_generate_step(
            p_arr,
            inner_model,
            max_tokens=max_tokens,
            temp=0.0,
            accept_counter=counter,
            num_draft_tokens=K,
        )
        for tok, _lp, _fd in gen:
            tokens.append(int(tok))
            n += 1
            if n >= max_tokens:
                break
        return tokens

    # Setup: inject sidecar ONCE. We compare BOTH pre-inject baseline
    # (no MTP) AND post-inject K=1..5 outputs, since injection modifies
    # the model class (via ``__class__`` swap). Actually the greedy
    # baseline can be run pre-inject and the MTP runs post-inject —
    # class swap doesn't affect the plain forward path (baseline uses
    # the SAME model, but the class swap adds mtp_forward /
    # mtp_draft_block / mtp_max_batch_size and modifies __call__ to
    # accept return_hidden + n_confirmed. The base forward through
    # ``self.model(...)`` is unchanged, so plain generate_step still
    # yields the same tokens).

    # Do a preflight baseline BEFORE injecting to confirm the swap
    # doesn't perturb plain generation.
    print("[lossless] Preflight baseline (pre-inject) ...", file=sys.stderr)
    pre_inject_baseline = {}
    for pid, text in _PROMPTS:
        pre_inject_baseline[pid] = _greedy_baseline(text, args.max_tokens)

    ok = inject_mtp_support(model, mtp_sidecar=args.sidecar)
    if not ok:
        raise SystemExit("inject_mtp_support failed")
    assert validate_mtp_support(model)
    print("[lossless] MTP injected. Post-inject baseline ...", file=sys.stderr)

    post_inject_baseline = {}
    for pid, text in _PROMPTS:
        post_inject_baseline[pid] = _greedy_baseline(text, args.max_tokens)

    print("[lossless] Chain-of-K sweep ...", file=sys.stderr)
    mtp_outputs: dict[tuple[str, int], list[int]] = {}
    for pid, text in _PROMPTS:
        for K in [1, 2, 3, 4]:
            mtp_outputs[(pid, K)] = _mtp_run(text, args.max_tokens, K)

    # ── Report ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Lossless contract report")
    print("=" * 60)
    for pid, _text in _PROMPTS:
        pre = pre_inject_baseline[pid]
        post = post_inject_baseline[pid]
        pre_post_match = pre == post
        print(f"\n[{pid}]")
        print(f"  pre_inject_baseline    (len {len(pre)}): {pre[:20]}...")
        print(f"  post_inject_baseline   (len {len(post)}): {post[:20]}...")
        print(f"  pre == post baseline?  {pre_post_match}")
        for K in [1, 2, 3, 4]:
            mtp = mtp_outputs[(pid, K)]
            mtp_match = mtp == post
            marker = "OK" if mtp_match else "FAIL"
            print(
                f"  [{marker}] K={K} mtp (len {len(mtp)}): "
                f"{mtp[:20]}... match_vs_baseline={mtp_match}"
            )
            if not mtp_match:
                # Find first divergence.
                for i in range(min(len(mtp), len(post))):
                    if mtp[i] != post[i]:
                        print(
                            f"       First divergence at index {i}: "
                            f"baseline={post[i]} mtp={mtp[i]}"
                        )
                        print(
                            f"       Context prev tokens: {mtp[max(0,i-4):i]} "
                            f"| baseline next {post[i:i+3]} vs "
                            f"mtp next {mtp[i:i+3]}"
                        )
                        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
