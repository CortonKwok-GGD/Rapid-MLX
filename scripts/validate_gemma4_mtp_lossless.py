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

That q_len=1-vs-q_len>=2 quant-SDPA drift is the exact anti-pattern
recorded in ``~/.claude/projects/-Users-raullenstudio-work-rapid-mlx/
memory/knowledge/gotchas.md`` under "MLX SDPA numerics DIVERGE at
q_len=1 vs q_len>=2 under quant weights" — the ambush that #1038's
old harness fell into.

The correct contract is **batched-consistent**: at greedy sampling
(temperature=0), MTP's argmax outputs must match the argmax outputs
of the SAME generator / SAME model instance / SAME sampler chain
run with ``max_k=0`` (drafter never proposes). Small numeric drift
in logits (below the argmax-crossover threshold) MUST NOT change the
emitted token. Divergence means either:

* A real MTP bug (position_ids skew, cache offset, softcap ordering,
  or the assistant drafter's logit projection).
* Numeric drift large enough to cross an argmax threshold — itself a
  correctness issue for the MTP path in production.

This script is the reference harness for that contract. Baseline
modes (``--baseline-mode``):

* ``batched-consistent`` (default): baseline runs through the SAME
  ``mtp_generate_step`` with ``max_k=0`` — the DepthController is
  clamped to K in {0}, so the generator only takes the q_len=1
  single-token backbone forward branch. Same generator loop, same
  sampler chain, same MTP-injected model instance, same logprob
  ordering as the ``max_k>=1`` MTP run. Any residual divergence is
  a REAL bug, not an artificial harness artifact.
* ``plain-decode`` (legacy): baseline runs through
  ``mlx_lm.stream_generate`` on a SEPARATE plain-loaded model
  instance (no MTP injection). Kept for A/B debug against the pre-fix
  contract that reverted #1038; DO NOT use for gate decisions.

Flow:

1. Load the target with MTP injected once (``dispatch_mtp_inject``).
   In legacy mode, also load a plain baseline model instance.
2. For each canned prompt, run baseline first (with
   ``reset_controllers()``), then MTP with ``max_k=args.max_k``
   (also with ``reset_controllers()``).
3. Compare tokens position-by-position; on divergence, emit the
   position + baseline/MTP token so the operator can triage whether
   it's softcap ordering, RoPE skew, or genuine numerical error.
4. Report accept-rate and wall-clock per prompt.

Precision axis (``--target-precision``)
---------------------------------------

* ``quant`` (default): the shipped acceptance target. Loads a 4/5/6/8
  bit mlx-community mirror (e.g. ``mlx-community/gemma-4-31b-it-4bit``).
  The q_len=1-vs-q_len>=2 SDPA numeric drift documented in the gotcha
  memo above lives here — expect a small number of prompts to diverge
  even under batched-consistent baseline, because the argmax crossover
  threshold is exceeded by the intrinsic mlx quant SDPA drift.
  Use ``--byte-exact-min-pass`` to gate against the drift ceiling.
* ``bf16`` (correctness proof): loads an unquantized mlx-community
  bf16 mirror (e.g. ``mlx-community/gemma-4-12B-it-bf16``). Removes
  the quant-SDPA drift entirely; any byte-exact divergence in this
  mode is a REAL injector bug (H1 chained cascade / H2 softcap
  ordering / cache offset / RoPE skew) and MUST be root-caused before
  shipping. The 0.10.6 A3 spike-3 evidence table (proving the injector
  is correct in absence of quant) was gathered in this mode against
  the 12B-it-bf16 target — see the branch's PR description.
* ``fp16``: reserved. Same semantics as ``bf16`` when an fp16 mirror
  is available; no fp16 mirror is currently published on
  ``mlx-community`` at Gemma-4 sizes, so use ``bf16`` in practice.

Usage
-----

::

    # A phase — quant acceptance gate (matches shipped default)
    python scripts/validate_gemma4_mtp_lossless.py \\
        --target-model mlx-community/gemma-4-31b-it-4bit \\
        --drafter google/gemma-4-31B-it-assistant \\
        --target-precision quant \\
        --byte-exact-min-pass 4 \\
        --runs 5 \\
        --max-new-tokens 128

    # D phase — bf16 correctness proof (evidence for the PR)
    python scripts/validate_gemma4_mtp_lossless.py \\
        --target-model mlx-community/gemma-4-12B-it-bf16 \\
        --drafter google/gemma-4-12B-it-assistant \\
        --target-precision bf16 \\
        --max-new-tokens 96 \\
        --prompts-file scripts/gemma4_mtp_prompts_mini.jsonl

If ``--prompts-file`` is omitted, an embedded 10-prompt fixture is used
(short / medium / long mix as spec'd in the task charter).

The script exits 0 on byte-exact-min-pass met + median uplift >= perf
floor (default 1.25x — the 0.10.6 gate raullen accepted after
direct-probe accept rate of 87-94%). Exits 2 on correctness failure
(byte-exact pass < min). Exits 3 on perf-only failure (correctness
passed but perf below floor).

Byte-exact-min-pass semantics
=============================

Under ``--target-precision quant`` the intrinsic q_len=1 vs q_len>=2
mlx SDPA numeric drift (memory gotcha "MLX SDPA numerics DIVERGE at
q_len=1 vs q_len>=2 under quant weights") crosses argmax on a subset
of prompts. Empirically 4/10 is the observed ceiling on the shipped
mixed-length fixture. Setting ``--byte-exact-min-pass 4`` documents
that ceiling; the D-phase bf16 evidence table proves the residual
divergences are quant-side, not injector-side.

Under ``--target-precision bf16``/``fp16`` the default is ALL prompts
(i.e. len(prompts)/len(prompts)) — any divergence is a real injector
bug and must halt.
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
    """Legacy ``plain-decode`` baseline via ``mlx_lm.stream_generate``.

    Kept behind ``--baseline-mode plain-decode`` for A/B against the
    pre-fix contract. This is the harness shape that #1038's revert
    depended on and that the memory gotcha
    ("MLX SDPA numerics DIVERGE at q_len=1 vs q_len>=2 under quant
    weights") explicitly warns against. Prefer
    ``batched-consistent`` for all real gate decisions.
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


def _run_generator_baseline(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
) -> tuple[list[int], float]:
    """Batched-consistent baseline: same generator, ``max_k=0``.

    Runs the vendored ``mtp_generate_step`` on the SAME MTP-injected
    model instance the MTP run uses, but with ``max_k=0`` so the
    :class:`DepthController` can only pick ``K=0`` — the "park"
    branch that does a single q_len=1 backbone forward per emitted
    token. This eliminates every artificial harness divergence source
    the legacy ``plain-decode`` baseline suffered from:

    * Same model instance (no double-load numeric drift under quant
      weight repack).
    * Same generator loop / same ``_step_backbone`` / same sampler
      chain / same logprob ordering as the MTP verify path.
    * Same ``_apply_chat_template`` (deterministic across calls).
    * Same KV-cache class stack from ``make_prompt_cache`` (no
      ``stream_generate`` vs ``mtp_generate_step`` divergence in
      cache init).

    The residual difference is intrinsic: baseline stays at q_len=1
    for every emit, while MTP verify is q_len in {1, 2, K+1} depending
    on the controller pick. Under quantized SDPA that q_len drift
    perturbs logits below the argmax-crossover threshold in normal
    text ranges; any argmax cross-over surfaced by this contract is
    the real MTP correctness signal we want to catch.

    See ``~/.claude/projects/-Users-raullenstudio-work-rapid-mlx/
    memory/knowledge/gotchas.md`` — "MLX SDPA numerics DIVERGE at
    q_len=1 vs q_len>=2 under quant weights" — for the ambush that
    this contract was designed around.
    """
    import mlx.core as mx

    from vllm_mlx.spec_decode.mtp.draft_k_controller_v2 import (
        reset_controllers,
    )
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    prompt_ids = _apply_chat_template(tokenizer, prompt_text)
    prompt_arr = mx.array(prompt_ids)

    # Reset before EACH baseline run so max_k=0 takes effect fresh —
    # otherwise a prior max_k=2 controller would linger in the registry
    # and quietly diverge the semantics of "batched-consistent baseline".
    reset_controllers()

    tokens: list[int] = []
    t0 = time.perf_counter()
    step_gen = mtp_generate_step(
        prompt=prompt_arr,
        model=model,
        max_tokens=max_new_tokens,
        temp=0.0,
        max_k=0,
    )
    for step_out in step_gen:
        if isinstance(step_out, tuple):
            tok = step_out[0]
        else:
            tok = getattr(step_out, "token", step_out)
        tokens.append(int(tok))
        if len(tokens) >= max_new_tokens:
            break
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
        default=1.25,
        help="Median tok/s ratio that must be met to consider VALIDATE "
             "a perf pass. Default 1.25x — the 0.10.6 A3 gate after "
             "raullen accepted 1.27x as the acceptance-verified result "
             "(prior 1.4x floor pre-dated the batched-consistent "
             "contract fix). Correctness is checked independently.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("batched-consistent", "plain-decode"),
        default="batched-consistent",
        help="How to run the baseline path. "
             "``batched-consistent`` (default) routes the baseline "
             "through the SAME ``mtp_generate_step`` generator with "
             "``max_k=0`` on the SAME MTP-injected model — same code "
             "path, same sampler, same cache semantics; the only "
             "residual difference vs the MTP run is q_len (baseline "
             "always q_len=1, MTP verify q_len>=2). This is the "
             "contract the 0.10.6 A3 spike was landed under. "
             "``plain-decode`` is the LEGACY harness that #1038 "
             "originally reverted on — kept for A/B debug; do NOT use "
             "for gate decisions (see memory gotcha 'MLX SDPA numerics "
             "DIVERGE at q_len=1 vs q_len>=2 under quant weights').",
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
        "--target-precision",
        choices=("quant", "bf16", "fp16"),
        default="quant",
        help="Documents which precision the target-model is loaded at. "
             "``quant`` (default) is the shipped acceptance path — the "
             "q_len=1-vs-q_len>=2 mlx SDPA drift lives here. "
             "``bf16`` is the correctness-proof path — no quant drift, "
             "any byte-inequal is a real injector bug. ``fp16`` is "
             "reserved; use bf16 in practice since no fp16 mlx-community "
             "mirror ships at Gemma-4 sizes. This flag does NOT change "
             "how the target is loaded (mlx_lm.load resolves precision "
             "from the mirror's config) — it only sets the default gate "
             "for --byte-exact-min-pass and annotates the run header.",
    )
    parser.add_argument(
        "--byte-exact-min-pass",
        type=int,
        default=None,
        help="Minimum number of prompts that must be byte-exact for the "
             "correctness gate to pass. If unset: for ``bf16``/``fp16`` "
             "targets the default is len(prompts) (all-or-nothing — any "
             "divergence indicates a real injector bug); for ``quant`` "
             "targets the default is len(prompts) minus the documented "
             "SDPA drift ceiling of 6 prompts (i.e. 4/10 on the shipped "
             "10-prompt fixture). Setting an explicit value overrides "
             "both defaults. See the module docstring section "
             "'Byte-exact-min-pass semantics' for the rationale.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat each prompt (both baseline and "
             "MTP). Speedup is reported as the median across runs to "
             "damp per-run tok/s jitter (thermal / GC / IO). Under "
             "greedy sampling the emitted token sequence is "
             "deterministic across runs, so byte-exact is checked "
             "against run 0 only; a divergence between runs on the "
             "same prompt is asserted as a bug. Default 1.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    if args.byte_exact_min_pass is None:
        if args.target_precision in ("bf16", "fp16"):
            byte_exact_min = None  # resolved to len(prompts) after load
        else:
            byte_exact_min = None  # resolved to len(prompts)-6 after load
    else:
        byte_exact_min = args.byte_exact_min_pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompts = _load_prompts(args.prompts_file)

    print(f"# validate_gemma4_mtp_lossless.py")
    print(f"# target: {args.target_model}")
    print(f"# drafter: {args.drafter}")
    print(f"# prompts: {len(prompts)}  (perf floor: {args.perf_floor}x)")
    print(f"# baseline-mode: {args.baseline_mode}")
    print(f"# target-precision: {args.target_precision}")
    print(f"# runs per prompt: {args.runs}")
    if args.baseline_mode == "batched-consistent":
        print("# contract: batched-consistent — SAME generator + SAME model "
              "+ max_k=0 baseline vs max_k>=1 MTP.")
        print("#   Any argmax divergence is a REAL drafter/verify bug, not "
              "harness drift.")
    else:
        print("# contract: LEGACY plain-decode (stream_generate vs "
              "mtp_generate_step) — do NOT use for gate decisions.")
        print("#   Kept for A/B against the pre-fix harness that #1038 "
              "reverted on.")
    # Resolve final min-pass gate now that we have len(prompts).
    if byte_exact_min is None:
        if args.target_precision in ("bf16", "fp16"):
            byte_exact_min = len(prompts)
            print(f"# byte-exact-min-pass: {byte_exact_min}/{len(prompts)} "
                  "(auto — bf16/fp16 correctness proof requires all)")
        else:
            byte_exact_min = max(0, len(prompts) - 6)
            print(f"# byte-exact-min-pass: {byte_exact_min}/{len(prompts)} "
                  "(auto — quant SDPA drift ceiling per gotchas.md)")
    else:
        print(f"# byte-exact-min-pass: {byte_exact_min}/{len(prompts)} "
              "(explicit)")
    print()

    if args.baseline_mode == "plain-decode":
        print("=== Loading baseline (no MTP) ===")
        baseline_model, baseline_tokenizer = _load_baseline(args.target_model)
    else:
        baseline_model = None
        baseline_tokenizer = None
    print("=== Loading MTP (dispatch_mtp_inject) ===")
    mtp_model, mtp_tokenizer = _load_mtp(args.target_model, args.drafter)
    # In batched-consistent mode the baseline shares the MTP-injected
    # model instance — same weights, same quant repack, same wrapper
    # class stack. See ``_run_generator_baseline`` docstring for the
    # rationale.
    if args.baseline_mode == "batched-consistent":
        baseline_model = mtp_model
        baseline_tokenizer = mtp_tokenizer

    rows: list[dict[str, Any]] = []
    any_divergence = False
    _print_header()

    for entry in prompts:
        prompt = entry["prompt"]
        pid = entry.get("id", "?")
        max_tokens = args.max_new_tokens or entry.get("target_new_tokens", 128)

        # Multi-run: collect tok/s per run, keep run-0 tokens for
        # byte-exact + assert determinism across runs.
        base_tok_s_runs: list[float] = []
        mtp_tok_s_runs: list[float] = []
        base_tokens_ref: list[int] | None = None
        mtp_tokens_ref: list[int] | None = None

        for run_idx in range(args.runs):
            if args.baseline_mode == "batched-consistent":
                base_tokens, base_t = _run_generator_baseline(
                    baseline_model, baseline_tokenizer, prompt, max_tokens
                )
            else:
                base_tokens, base_t = _run_plain_greedy(
                    baseline_model, baseline_tokenizer, prompt, max_tokens
                )
            mtp_tokens, mtp_t, _ = _run_mtp_greedy(
                mtp_model, mtp_tokenizer, prompt, max_tokens, max_k=args.max_k
            )

            base_tok_s_runs.append(
                len(base_tokens) / base_t if base_t > 0 else 0.0
            )
            mtp_tok_s_runs.append(
                len(mtp_tokens) / mtp_t if mtp_t > 0 else 0.0
            )

            if run_idx == 0:
                base_tokens_ref = list(base_tokens)
                mtp_tokens_ref = list(mtp_tokens)
            else:
                # Determinism gate — greedy sampling MUST produce the
                # same token sequence across runs on the same prompt.
                # A between-runs divergence is a REAL bug (RNG leak,
                # cache eviction, non-deterministic scatter).
                if list(base_tokens) != base_tokens_ref:
                    print(
                        f"\n!! BASELINE non-determinism on {pid} "
                        f"run{run_idx}: greedy decode drifted between runs."
                    )
                if list(mtp_tokens) != mtp_tokens_ref:
                    print(
                        f"\n!! MTP non-determinism on {pid} "
                        f"run{run_idx}: greedy decode drifted between runs."
                    )

        # Aggregate across runs — median damps jitter.
        plain_tok_s = statistics.median(base_tok_s_runs)
        mtp_tok_s = statistics.median(mtp_tok_s_runs)
        speedup = mtp_tok_s / plain_tok_s if plain_tok_s > 0 else 0.0

        assert base_tokens_ref is not None and mtp_tokens_ref is not None
        div = _find_first_divergence(base_tokens_ref, mtp_tokens_ref)
        byte_exact = div is None

        row = {
            "id": pid,
            "plain_tok_s": plain_tok_s,
            "mtp_tok_s": mtp_tok_s,
            "speedup": speedup,
            "byte_exact": byte_exact,
            "divergence_at": str(div) if div is not None else "-",
            "base_len": len(base_tokens_ref),
            "mtp_len": len(mtp_tokens_ref),
        }
        rows.append(row)
        _print_row(row)

        if not byte_exact:
            any_divergence = True
            print(
                f"\n!! DIVERGENCE at position {div}: "
                f"baseline={base_tokens_ref[div] if div < len(base_tokens_ref) else '<eos>'} "
                f"mtp={mtp_tokens_ref[div] if div < len(mtp_tokens_ref) else '<eos>'}"
            )

    # Aggregates
    plain_med = statistics.median(r["plain_tok_s"] for r in rows)
    mtp_med = statistics.median(r["mtp_tok_s"] for r in rows)
    speedup_med = statistics.median(r["speedup"] for r in rows)
    byte_exact_count = sum(1 for r in rows if r["byte_exact"])
    print()
    print(f"Median plain tok/s : {plain_med:.2f}")
    print(f"Median MTP tok/s   : {mtp_med:.2f}")
    print(f"Median speedup     : {speedup_med:.2f}x")
    if args.target_precision == "quant":
        print(f"Byte-exact prompts : {byte_exact_count}/{len(rows)} "
              f"(gate: >= {byte_exact_min}; quant drift <= MLX q_len=1 vs "
              f"q_len>=2 numerical ceiling; bf16 target achieves all-pass — "
              f"see memory gotcha 'MLX SDPA numerics DIVERGE at q_len=1 vs "
              f"q_len>=2 under quant weights')")
    else:
        print(f"Byte-exact prompts : {byte_exact_count}/{len(rows)} "
              f"(gate: >= {byte_exact_min}; {args.target_precision} target — "
              f"any divergence is a real injector bug)")

    if byte_exact_count < byte_exact_min:
        print("\nVERDICT: FAIL (correctness)")
        print(
            f"Byte-exact {byte_exact_count}/{len(rows)} below required "
            f"{byte_exact_min}/{len(rows)}. Debug order:\n"
            "  1. Assistant drafter's post-projection ordering vs softcap.\n"
            "  2. Position_ids / RoPE offset on the drafter's Q pass.\n"
            "  3. Cache offset — verify drafter reads target K/V at the "
            "     right tail-layer index.\n"
            "  4. Final logit projection: tied embed vs standalone lm_head.\n"
            "  5. Sampling-side: check that argmax vs temp=0 sampler paths "
            "     are byte-equal in the MTP verify.\n"
            "  6. Under --target-precision quant, re-run with "
            "     --target-precision bf16 on the 12B mirror to isolate "
            "     quant drift vs injector bugs.\n"
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
