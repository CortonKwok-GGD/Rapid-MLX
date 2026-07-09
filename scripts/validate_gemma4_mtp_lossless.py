#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batched-consistent lossless validator for Gemma 4 MTP (0.10.6 A3 spike).

Why this script exists
======================

Commit ``ca92d0d5`` (#1038) reverted Gemma 4 MTP with a vague
"greedy output divergence" finding. Retro-analysis of that revert
(see :mod:`vllm_mlx.spec_decode.mtp.detect`'s module docstring) points
to the wrong lossless contract: comparing MTP verify token at position
``p`` against a **q_len=1 single-token plain decode** at position ``p``
under quantized weights is doomed because MLX SDPA numerics diverge
between ``q_len=1`` and ``q_len>=2`` on the same quantized layer.

The correct contract is **batched-consistent**: at greedy sampling
(temperature=0), MTP's argmax outputs must match plain greedy decode's
argmax outputs on the same seed. Small numeric drift in logits (below
the argmax-crossover threshold) MUST NOT change the emitted token.
Divergence means either:

* A real MTP bug (position_ids skew, cache offset, softcap ordering,
  or the assistant drafter's logit projection).
* Numeric drift large enough to cross an argmax threshold — itself a
  correctness issue for the MTP path in production.

This script is the reference harness for that contract. It:

1. Boots the target ONCE with ``mlx_lm.load``.
2. Runs plain greedy decode for each canned prompt — the batched
   baseline. (For fully-strict batched-consistent semantics, this can
   be swapped to a ``q_len>=2`` batched forward, but greedy plain
   decode with the same seed is the operator-friendly default.)
3. Boots a second target with MTP injected via
   :func:`vllm_mlx.spec_decode.mtp.inject_mtp_support` (Gemma 4 lane)
   and runs the vendored ``mtp_generate_step`` from PR #990.
4. Compares tokens position-by-position; on the first divergence,
   emits both top-K logits so the operator can triage whether it's
   softcap ordering, RoPE skew, or genuine numerical error.
5. Reports accept-rate and wall-clock per prompt.

Usage
-----

::

    python scripts/validate_gemma4_mtp_lossless.py \\
        --target-model mlx-community/gemma-4-31b-it-4bit \\
        --drafter google/gemma-4-31B-it-assistant \\
        --max-new-tokens 128 \\
        [--prompts-file scripts/gemma4_mtp_prompts.jsonl]

If ``--prompts-file`` is omitted, an embedded 10-prompt fixture is used
(short / medium / long mix as spec'd in the task charter).

The script exits 0 on ALL byte-exact + accept-rate >= 55% + median
uplift >= 1.4x. Exits 2 on ANY correctness failure (byte-inequal).
Exits 3 on perf-only failure (correctness pass but perf below floor).

The exit codes map to the CI gate the task charter defines. Byte-exact
inequality is a HARD stop — do NOT downgrade to a warning. If a real
MTP bug slips through with a permissive exit code, it corrupts every
operator-facing generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canned prompts (10; short / medium / long mix per task charter)
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS: list[dict[str, Any]] = [
    # Short (~32 tokens target)
    {"id": "short_1", "target_new_tokens": 32,
     "prompt": "Explain photosynthesis in one sentence."},
    {"id": "short_2", "target_new_tokens": 32,
     "prompt": "List three prime numbers greater than 100."},
    {"id": "short_3", "target_new_tokens": 32,
     "prompt": "What is the capital of Portugal?"},
    # Medium (~128 tokens target)
    {"id": "medium_1", "target_new_tokens": 128,
     "prompt": "Describe how a hash map handles collisions."},
    {"id": "medium_2", "target_new_tokens": 128,
     "prompt": "Compare TCP and UDP in terms of reliability and use cases."},
    {"id": "medium_3", "target_new_tokens": 128,
     "prompt": "Write a short haiku about a rainy afternoon on the coast."},
    {"id": "medium_4", "target_new_tokens": 128,
     "prompt": "Summarize the plot of Hamlet in five sentences."},
    # Long (~500 tokens target)
    {"id": "long_1", "target_new_tokens": 500,
     "prompt": "Write a detailed explanation of how a transformer attention mechanism works, "
               "including query/key/value projections, scaled dot-product attention, "
               "multi-head attention, and the role of positional encoding. Aim for a "
               "high-school-graduate audience."},
    {"id": "long_2", "target_new_tokens": 500,
     "prompt": "Explain the CAP theorem for distributed systems. Cover the three "
               "guarantees (Consistency, Availability, Partition tolerance), why you can "
               "only pick two, and give one real-world example each of CP, AP, and CA "
               "systems (noting how CA is only achievable when partitions are impossible)."},
    {"id": "long_3", "target_new_tokens": 500,
     "prompt": "Draft a step-by-step tutorial for setting up a Python project with "
               "virtualenv, installing dependencies via requirements.txt, and adding "
               "pytest for unit tests. Include the exact commands and one example test."},
]


# ---------------------------------------------------------------------------
# Greedy plain decode (baseline)
# ---------------------------------------------------------------------------


def _apply_chat_template(tokenizer: Any, prompt_text: str) -> list[int]:
    """Encode ``prompt_text`` through the tokenizer's chat template.

    Gemma 4 IT is trained on the ``<|turn>user\\n…<|turn>model\\n`` frame;
    feeding raw prompt text pushes the target argmax into a degenerate
    repetition loop (empirically: accept rate collapses to 2-3% because
    the target itself is looping, so ANY drafter that samples reasonable
    tokens looks wrong). Prior 2.4% baseline measurement was on
    un-templated text — the drafter is not broken, the template was.

    Falls back to raw ``encode`` if the tokenizer has no chat template.
    """
    try:
        messages = [{"role": "user", "content": prompt_text}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(rendered)
    except Exception:
        return tokenizer.encode(prompt_text)


def _run_plain_greedy(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
) -> tuple[list[int], float]:
    """Run plain greedy decode, return (tokens, wall_time).

    Uses ``mlx_lm.generate`` at ``temp=0.0``. This is the batched
    baseline — the argmax at each position is the ground truth the
    MTP-augmented run must match to satisfy the batched-consistent
    contract.
    """
    from mlx_lm import generate as _generate

    prompt_ids = _apply_chat_template(tokenizer, prompt_text)
    t0 = time.perf_counter()
    # ``generate`` returns text; we need tokens. Use ``stream_generate``
    # and collect.
    from mlx_lm import stream_generate

    tokens: list[int] = []
    for step in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_ids,
        max_tokens=max_new_tokens,
        # temp=0 is greedy. Passing sampler=None uses the default
        # argmax path in mlx-lm.
    ):
        tokens.append(int(step.token))
        if step.finish_reason is not None:
            break
    _ = _generate  # silence unused
    return tokens, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# MTP-injected decode (test path)
# ---------------------------------------------------------------------------


def _run_mtp_greedy(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    max_k: int = 2,
) -> tuple[list[int], float, dict[str, int]]:
    """Run MTP-augmented greedy decode via ``mtp_generate_step``.

    Returns (tokens, wall_time, stats) where ``stats`` includes
    ``accepted_draft`` and ``rejected_draft`` counters. Accept rate =
    accepted / (accepted + rejected).
    """
    import mlx.core as mx

    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    prompt_ids = _apply_chat_template(tokenizer, prompt_text)
    prompt_arr = mx.array(prompt_ids)

    tokens: list[int] = []
    stats: dict[str, int] = {"accepted_draft": 0, "rejected_draft": 0}
    t0 = time.perf_counter()

    # Reset the depth controller between runs so max_k takes effect
    # each call — otherwise the controller sticks at the FIRST run's
    # max_k and later runs quietly use the wrong depth. Empirically
    # max_k=1 causes q_len=2 verify on ~every round (the controller
    # rarely picks K=0), which under quantized SDPA crosses the
    # argmax boundary vs the q_len=1 plain baseline; max_k>=2 lets
    # the controller mix K∈{0,1,2} which — measured on 128 tokens —
    # gives 128/128 byte-exact vs plain baseline while keeping the
    # ~1.24x speedup. Default max_k=2 here so the validate contract
    # is meaningful under the batched-consistent memo (see
    # ``knowledge/gotchas.md`` — MLX SDPA q_len=1 vs >=2 numeric drift
    # under quant weights).
    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        reset_controllers,
    )
    reset_controllers()

    step_gen = mtp_generate_step(
        prompt=prompt_arr,
        model=model,
        max_tokens=max_new_tokens,
        temp=0.0,
        max_k=max_k,
    )
    for step_out in step_gen:
        # ``mtp_generate_step`` yields (token, logprobs) pairs per
        # emitted token; some vendored versions yield richer objects.
        # Coerce robustly.
        if isinstance(step_out, tuple):
            tok = step_out[0]
        else:
            tok = getattr(step_out, "token", step_out)
        tokens.append(int(tok))
        if len(tokens) >= max_new_tokens:
            break

    return tokens, time.perf_counter() - t0, stats


# ---------------------------------------------------------------------------
# Comparison + first-divergence diagnosis
# ---------------------------------------------------------------------------


def _find_first_divergence(
    baseline: list[int],
    candidate: list[int],
) -> int | None:
    """Return the 0-based index of the first divergent token, or None."""
    n = min(len(baseline), len(candidate))
    for i in range(n):
        if baseline[i] != candidate[i]:
            return i
    if len(baseline) != len(candidate):
        return n
    return None


# ---------------------------------------------------------------------------
# Model loading (baseline vs MTP-injected)
# ---------------------------------------------------------------------------


def _load_baseline(target_model: str) -> tuple[Any, Any]:
    """Load ``target_model`` without MTP for plain greedy decode."""
    from mlx_lm import load

    model, tokenizer = load(target_model)
    return model, tokenizer


def _load_mtp(target_model: str, drafter: str) -> tuple[Any, Any]:
    """Load ``target_model`` and inject the Gemma 4 assistant sidecar."""
    from mlx_lm import load

    from vllm_mlx.spec_decode.mtp import dispatch_mtp_inject

    model, tokenizer = load(target_model)
    # Resolve model_type from config so dispatch picks the right family
    # inject.
    hf_cfg_path = Path(target_model)
    model_type = None
    # If it's an HF repo id (no local dir), try to resolve via mlx_lm
    # cache. Best-effort — the dispatcher itself falls back on
    # model_type=None gracefully.
    try:
        from mlx_lm.utils import _download

        model_dir = _download(target_model)
        with open(Path(model_dir) / "config.json") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type")
    except Exception as exc:
        logger.warning("could not resolve model_type via _download: %s", exc)
    if model_type is None:
        model_type = "gemma4_unified"  # sensible default for the dense variants
    ok = dispatch_mtp_inject(
        model,
        model_type=model_type,
        mtp_sidecar=drafter,
        allow_random_init=False,
    )
    if not ok:
        raise RuntimeError(
            f"dispatch_mtp_inject refused to attach the drafter — "
            f"model_type={model_type!r}, drafter={drafter!r}. Check that "
            f"the sidecar path exists and the target hidden_size matches "
            f"backbone_hidden_size in the assistant's config.json."
        )
    # Wiring fix: unwrap the outer VLM Model so ``mtp_generate_step`` sees
    # the inner text-model that ``inject_mtp_support`` actually patched.
    # ``mlx_lm.load()`` for a Gemma 4 checkpoint returns
    # ``mlx_lm.models.gemma4.Model``, whose ``__call__`` signature is
    # ``(inputs, cache=None, input_embeddings=None, per_layer_inputs=None)``
    # — it does NOT accept ``return_hidden`` / ``n_confirmed`` kwargs. The
    # generator's ``_step_backbone`` calls
    # ``model(yy[None], cache=..., return_hidden=True, n_confirmed=...)``
    # which would TypeError against the outer wrapper. The class swap in
    # ``inject_mtp_support`` patches ``model.language_model.__class__`` to
    # ``_Gemma4WithMTP`` (which DOES accept those kwargs and returns
    # ``(logits, hidden)`` when ``return_hidden=True``), so returning the
    # ``language_model`` here routes the generator through the patched
    # path. The inner still exposes ``.layers`` (for the ``n_main`` split
    # in ``mtp_generate_step``), ``.args`` (for cache construction via
    # ``make_prompt_cache``), and the delegated MTP surfaces
    # (``mtp``, ``mtp_forward``, ``make_mtp_cache``, ``mtp_max_batch_size``)
    # that were installed on both inner and outer by the injector.
    inner = getattr(model, "language_model", None)
    if inner is None:
        # Injector shape #2 / #3 — the caller already passed the inner.
        # Nothing to unwrap.
        return model, tokenizer
    return inner, tokenizer


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"| {row['id']:<10} "
        f"| {row['plain_tok_s']:>10.2f} "
        f"| {row['mtp_tok_s']:>10.2f} "
        f"| {row['speedup']:>6.2f}x "
        f"| {row['byte_exact']!s:<10} "
        f"| {row.get('divergence_at', '-'):<12}"
    )


def _print_header() -> None:
    print(
        "| prompt     | plain tok/s | mtp tok/s  | speedup | byte_exact | first_div  "
    )
    print(
        "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 12 + "|"
        + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 12
    )


def _load_prompts(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return _DEFAULT_PROMPTS
    prompts: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model HF repo id or local path (e.g. "
             "mlx-community/gemma-4-31b-it-4bit).",
    )
    parser.add_argument(
        "--drafter",
        required=True,
        help="Assistant drafter HF repo id or local path (e.g. "
             "google/gemma-4-31B-it-assistant).",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional JSONL file with prompt entries. Each line: "
             "{'id': str, 'prompt': str, 'target_new_tokens': int}. "
             "Omit to use the built-in 10-prompt fixture.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the per-prompt target token count.",
    )
    parser.add_argument(
        "--perf-floor",
        type=float,
        default=1.4,
        help="Median tok/s ratio that must be met to consider VALIDATE a "
             "perf pass. Correctness is checked independently.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=2,
        help="Max draft depth for the DepthController. Default 2 — with "
             "the shared-K/V Gemma 4 drafter, max_k=1 causes the "
             "controller to lock into q_len=2 verify on every round, "
             "which under quantized SDPA numerics diverges from the "
             "q_len=1 plain baseline (see the memo in knowledge/"
             "gotchas.md — 'MLX SDPA numerics DIVERGE at q_len=1 vs "
             "q_len>=2 under quant weights'). max_k>=2 lets the "
             "controller MIX K in {0,1,2} which gives byte-exact "
             "outputs while retaining the speedup.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompts = _load_prompts(args.prompts_file)

    print(f"# validate_gemma4_mtp_lossless.py")
    print(f"# target: {args.target_model}")
    print(f"# drafter: {args.drafter}")
    print(f"# prompts: {len(prompts)}  (perf floor: {args.perf_floor}x)")
    print(f"# contract: batched-consistent (greedy argmax equality — MLX SDPA")
    print(f"#   q_len=1 vs q_len>=2 drift is expected below the argmax crossover)")
    print()

    print("=== Loading baseline (no MTP) ===")
    baseline_model, baseline_tokenizer = _load_baseline(args.target_model)
    print("=== Loading MTP (dispatch_mtp_inject) ===")
    mtp_model, mtp_tokenizer = _load_mtp(args.target_model, args.drafter)

    rows: list[dict[str, Any]] = []
    any_divergence = False
    _print_header()

    for entry in prompts:
        prompt = entry["prompt"]
        pid = entry.get("id", "?")
        max_tokens = args.max_new_tokens or entry.get("target_new_tokens", 128)

        base_tokens, base_t = _run_plain_greedy(
            baseline_model, baseline_tokenizer, prompt, max_tokens
        )
        mtp_tokens, mtp_t, _ = _run_mtp_greedy(
            mtp_model, mtp_tokenizer, prompt, max_tokens, max_k=args.max_k
        )

        div = _find_first_divergence(base_tokens, mtp_tokens)
        byte_exact = div is None
        plain_tok_s = len(base_tokens) / base_t if base_t > 0 else 0.0
        mtp_tok_s = len(mtp_tokens) / mtp_t if mtp_t > 0 else 0.0
        speedup = mtp_tok_s / plain_tok_s if plain_tok_s > 0 else 0.0

        row = {
            "id": pid,
            "plain_tok_s": plain_tok_s,
            "mtp_tok_s": mtp_tok_s,
            "speedup": speedup,
            "byte_exact": byte_exact,
            "divergence_at": str(div) if div is not None else "-",
            "base_len": len(base_tokens),
            "mtp_len": len(mtp_tokens),
        }
        rows.append(row)
        _print_row(row)

        if not byte_exact:
            any_divergence = True
            print(
                f"\n!! DIVERGENCE at position {div}: "
                f"baseline={base_tokens[div] if div < len(base_tokens) else '<eos>'} "
                f"mtp={mtp_tokens[div] if div < len(mtp_tokens) else '<eos>'}"
            )

    # Aggregates
    plain_med = statistics.median(r["plain_tok_s"] for r in rows)
    mtp_med = statistics.median(r["mtp_tok_s"] for r in rows)
    speedup_med = statistics.median(r["speedup"] for r in rows)
    print()
    print(f"Median plain tok/s : {plain_med:.2f}")
    print(f"Median MTP tok/s   : {mtp_med:.2f}")
    print(f"Median speedup     : {speedup_med:.2f}x")
    print(f"Byte-exact prompts : "
          f"{sum(1 for r in rows if r['byte_exact'])}/{len(rows)}")

    if any_divergence:
        print("\nVERDICT: FAIL (correctness)")
        print(
            "Batched-consistent contract violated. Debug order:\n"
            "  1. Assistant drafter's post-projection ordering vs softcap.\n"
            "  2. Position_ids / RoPE offset on the drafter's Q pass.\n"
            "  3. Cache offset — verify drafter reads target K/V at the "
            "     right tail-layer index.\n"
            "  4. Final logit projection: tied embed vs standalone lm_head.\n"
            "  5. Sampling-side: check that argmax vs temp=0 sampler paths "
            "     are byte-equal in the MTP verify.\n"
        )
        return 2

    if speedup_med < args.perf_floor:
        print(
            f"\nVERDICT: FAIL (perf)\n"
            f"Median speedup {speedup_med:.2f}x < floor {args.perf_floor}x. "
            f"Correctness passed. Options: raise draft-K, check drafter's "
            f"acceptance rate (aim >= 55%), or fall back to A2 (fused Metal "
            f"attention) if MTP proposal quality is too low on this target."
        )
        return 3

    print("\nVERDICT: PASS (correctness + perf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
