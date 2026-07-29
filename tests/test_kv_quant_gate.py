# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the KV-quant differential quality gate (absorb #4).

All metric tests use SYNTHETIC baseline/candidate token streams + logprob
vectors + byte counts — no model inference. A single ``@pytest.mark.slow`` smoke
runs the real harness on ONE small already-cached model and SKIPS cleanly when
the model isn't in the local HF cache (never downloads).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vllm_mlx.kv_quant_gate import (
    FAIL,
    NA,
    PASS,
    AgreementResult,
    LogitDivergence,
    RetentionResult,
    build_report,
    default_thresholds,
    extract_json_candidate,
    greedy_agreement_rate,
    is_valid_json,
    logit_divergence,
    memory_delta,
    structured_output_retention,
)


def _logsoftmax(logits: list[float]) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    return arr - float(np.log(np.sum(np.exp(arr))))


# ---------------------------------------------------------------------------
# Greedy agreement
# ---------------------------------------------------------------------------
def test_greedy_agreement_identical():
    r = greedy_agreement_rate([1, 2, 3, 4], [1, 2, 3, 4])
    assert r.rate == 1.0
    assert r.matched == 4
    assert r.total == 4
    assert r.first_divergence_index is None


def test_greedy_agreement_one_token_diff():
    # Diverges at index 2 -> 2 leading matches out of 4 compared.
    r = greedy_agreement_rate([1, 2, 3, 4], [1, 2, 9, 4])
    assert r.first_divergence_index == 2
    assert r.matched == 2
    assert r.total == 4
    assert r.rate == 0.5


def test_greedy_agreement_diverge_immediately():
    r = greedy_agreement_rate([5, 6], [9, 6])
    assert r.first_divergence_index == 0
    assert r.matched == 0
    assert r.rate == 0.0


def test_greedy_agreement_empty_streams():
    r = greedy_agreement_rate([], [])
    assert r.total == 0
    assert r.rate == 1.0
    assert r.first_divergence_index is None


def test_greedy_agreement_premature_eos_scores_below_one():
    """RED-GREEN: a candidate that ends early after an all-matching prefix must
    NOT score 1.0. With the old ``min`` denominator this returned 1.0, hiding the
    premature-EOS degradation the gate exists to catch; the ``max`` denominator
    counts the missing suffix positions as disagreement.
    """
    # baseline 5 tokens, candidate terminated at 3 (premature EOS), prefix matches
    r = greedy_agreement_rate([1, 2, 3, 4, 5], [1, 2, 3])
    assert r.total == 5  # denominator is the LONGER stream, not the shorter
    assert r.matched == 3
    assert r.rate == 0.6
    assert r.first_divergence_index == 3  # boundary where the short stream ended


def test_greedy_agreement_candidate_longer_also_penalized():
    r = greedy_agreement_rate([1, 2], [1, 2, 3, 4])
    assert r.total == 4
    assert r.matched == 2
    assert r.rate == 0.5
    assert r.first_divergence_index == 2


def test_greedy_agreement_empty_vs_nonempty_scores_zero():
    r = greedy_agreement_rate([], [1, 2, 3])
    assert r.total == 3
    assert r.matched == 0
    assert r.rate == 0.0


# ---------------------------------------------------------------------------
# Logit divergence
# ---------------------------------------------------------------------------
def test_logit_divergence_identical_is_zero():
    base = [_logsoftmax([2.0, 1.0, 0.0]), _logsoftmax([0.0, 3.0, 1.0])]
    r = logit_divergence(base, base)
    assert r.compared_steps == 2
    assert r.mean_kl == pytest.approx(0.0, abs=1e-9)
    assert r.max_kl == pytest.approx(0.0, abs=1e-9)
    assert r.top1_agreement_rate == 1.0


def test_logit_divergence_degraded_positive_kl_and_top1_drop():
    base = [_logsoftmax([5.0, 0.0, 0.0])]  # argmax 0
    cand = [_logsoftmax([0.0, 5.0, 0.0])]  # argmax 1
    r = logit_divergence(base, cand)
    assert r.compared_steps == 1
    assert r.mean_kl > 0.1
    assert r.top1_agreement_rate == 0.0


def test_logit_divergence_compare_len_bounds_steps():
    base = [_logsoftmax([3.0, 0.0]), _logsoftmax([0.0, 3.0]), _logsoftmax([1.0, 0.0])]
    cand = list(base)
    r = logit_divergence(base, cand, compare_len=1)
    assert r.compared_steps == 1


def test_logit_divergence_catastrophic_neg_inf_is_finite():
    # Baseline has mass at index 0; candidate assigns it -inf logprob.
    base = [_logsoftmax([10.0, 0.0])]
    cand = [np.array([-np.inf, 0.0])]
    r = logit_divergence(base, cand)
    assert math.isfinite(r.mean_kl)
    assert r.mean_kl > 1.0  # clearly flags the divergence


def test_logit_divergence_empty():
    r = logit_divergence([], [])
    assert r.compared_steps == 0
    assert r.mean_kl == 0.0
    assert r.top1_agreement_rate == 1.0


def test_logit_divergence_shape_mismatch_step_skipped():
    base = [_logsoftmax([1.0, 2.0, 3.0])]
    cand = [_logsoftmax([1.0, 2.0])]  # wrong vocab size -> skipped
    r = logit_divergence(base, cand)
    assert r.compared_steps == 0


# ---------------------------------------------------------------------------
# Structured-output retention + JSON helpers
# ---------------------------------------------------------------------------
def test_is_valid_json_plain_and_fenced_and_embedded():
    assert is_valid_json('{"a": 1}')
    assert is_valid_json('```json\n{"a": 1}\n```')
    assert is_valid_json('Sure! Here you go: {"a": [1, 2, 3]} — done.')
    assert is_valid_json("[1, 2, 3]")
    assert not is_valid_json("not json at all")
    assert not is_valid_json("")


def test_extract_json_candidate_prefers_fence():
    assert extract_json_candidate('```json\n{"x": 1}\n```') == '{"x": 1}'
    assert extract_json_candidate('prefix {"x": 1} suffix') == '{"x": 1}'


def test_is_valid_json_embedded_despite_other_brackets():
    """RED-GREEN: a valid object survives even when another bracketed fragment in
    the prose would break a naive first-brace-through-last-bracket span. The old
    span logic yielded ``[x] then {"a": 1}`` / ``[1, 2] and {"ok": true}`` (both
    unparseable); the raw_decode scan recovers the real JSON value.
    """
    assert is_valid_json('see [x] then {"a": 1}')
    assert is_valid_json('result: [1, 2] and {"ok": true}')
    # A genuinely JSON-free string still returns False.
    assert not is_valid_json("no brackets, no json here")


def test_retention_both_valid():
    r = structured_output_retention([('{"a": 1}', '{"a": 2}')])
    assert r.attributable == 1
    assert r.retained == 1
    assert r.rate == 1.0
    assert r.baseline_invalid == 0


def test_retention_candidate_breaks():
    r = structured_output_retention([('{"a": 1}', "sorry, no json")])
    assert r.attributable == 1
    assert r.retained == 0
    assert r.rate == 0.0


def test_retention_baseline_invalid_is_excluded():
    r = structured_output_retention([("garbage", "garbage")])
    assert r.attributable == 0
    assert r.baseline_invalid == 1
    assert r.rate is None  # N/A — nothing attributable


def test_retention_mixed():
    pairs = [
        ('{"a": 1}', '{"a": 1}'),  # attributable, retained
        ('{"b": 2}', "broke"),  # attributable, lost
        ("not json", "not json"),  # excluded
    ]
    r = structured_output_retention(pairs)
    assert r.attributable == 2
    assert r.retained == 1
    assert r.rate == 0.5
    assert r.baseline_invalid == 1


# ---------------------------------------------------------------------------
# Memory delta
# ---------------------------------------------------------------------------
def test_memory_delta_typical_int8():
    d = memory_delta(1000, 500)
    assert d.saved_bytes == 500
    assert d.reduction_ratio == 2.0
    assert d.saved_pct == 50.0


def test_memory_delta_candidate_zero_is_degenerate():
    d = memory_delta(1000, 0)
    assert d.reduction_ratio == 0.0  # treated as no saving, not div-by-zero


def test_memory_delta_no_saving():
    d = memory_delta(1000, 1000)
    assert d.reduction_ratio == 1.0
    assert d.saved_bytes == 0


# ---------------------------------------------------------------------------
# Thresholds + report schema + PASS/FAIL logic
# ---------------------------------------------------------------------------
def test_default_thresholds_int4_more_lenient_than_int8():
    t8 = default_thresholds("int8")
    t4 = default_thresholds("int4")
    assert t4.min_greedy_agreement < t8.min_greedy_agreement
    assert t4.max_mean_kl > t8.max_mean_kl
    assert t4.min_memory_reduction_ratio > t8.min_memory_reduction_ratio


def _passing_inputs():
    """Metric objects that sit exactly at / above the int8 thresholds -> all PASS."""
    t = default_thresholds("int8")
    return dict(
        thresholds=t,
        agreement=AgreementResult(
            total=10, matched=6, rate=t.min_greedy_agreement, first_divergence_index=6
        ),
        logits=LogitDivergence(
            compared_steps=8,
            mean_kl=t.max_mean_kl,  # exactly at bound -> PASS (<=)
            max_kl=t.max_mean_kl,
            top1_agreement_rate=0.9,
        ),
        retention=RetentionResult(
            attributable=2, retained=2, rate=1.0, baseline_invalid=0
        ),
        memory=memory_delta(2000, 1000),  # 2.0x >= 1.5
    )


def _base_report_kwargs():
    return dict(
        model="synthetic",
        hf_path="synthetic/path",
        baseline_dtype="bf16",
        candidate_dtype="int8",
        num_prompts=5,
        advisory=True,
        chip={
            "raw": "Apple M3 Ultra",
            "is_apple_silicon": True,
            "generation": 3,
            "variant": "Ultra",
            "is_m3_or_newer": True,
        },
    )


def test_report_schema_has_expected_fields():
    report = build_report(**_base_report_kwargs(), **_passing_inputs())
    d = report.to_dict()
    for key in (
        "schema",
        "kind",
        "advisory",
        "model",
        "hf_path",
        "baseline_dtype",
        "candidate_dtype",
        "num_prompts",
        "chip",
        "thresholds",
        "agreement",
        "logits",
        "retention",
        "memory",
        "niah",
        "metrics",
        "overall",
    ):
        assert key in d, key
    assert set(d["metrics"]) == {
        "greedy_agreement",
        "logit_mean_kl",
        "structured_retention",
        "memory_reduction",
    }
    # to_dict must be JSON-serializable end to end.
    import json

    json.loads(json.dumps(d))


def test_report_all_pass_at_thresholds():
    report = build_report(**_base_report_kwargs(), **_passing_inputs())
    assert report.overall == PASS
    for m in report.metrics.values():
        assert m["outcome"] == PASS


def test_report_agreement_just_below_threshold_fails():
    inputs = _passing_inputs()
    t = inputs["thresholds"]
    inputs["agreement"] = AgreementResult(
        total=100,
        matched=int((t.min_greedy_agreement - 0.01) * 100),
        rate=t.min_greedy_agreement - 0.01,
        first_divergence_index=1,
    )
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["greedy_agreement"]["outcome"] == FAIL
    assert report.overall == FAIL


def test_report_kl_above_threshold_fails():
    inputs = _passing_inputs()
    t = inputs["thresholds"]
    inputs["logits"] = LogitDivergence(
        compared_steps=4,
        mean_kl=t.max_mean_kl + 0.5,
        max_kl=t.max_mean_kl + 0.5,
        top1_agreement_rate=0.5,
    )
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["logit_mean_kl"]["outcome"] == FAIL
    assert report.overall == FAIL


def test_report_memory_below_reduction_fails():
    inputs = _passing_inputs()
    inputs["memory"] = memory_delta(1000, 900)  # 1.11x < 1.5x
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["memory_reduction"]["outcome"] == FAIL
    assert report.overall == FAIL


def test_report_na_metrics_do_not_fail_overall():
    inputs = _passing_inputs()
    # No comparable logit steps -> logit_mean_kl NA.
    inputs["logits"] = LogitDivergence(
        compared_steps=0, mean_kl=0.0, max_kl=0.0, top1_agreement_rate=1.0
    )
    # No attributable JSON prompt -> structured_retention NA.
    inputs["retention"] = RetentionResult(
        attributable=0, retained=0, rate=None, baseline_invalid=3
    )
    report = build_report(**_base_report_kwargs(), **inputs)
    assert report.metrics["logit_mean_kl"]["outcome"] == NA
    assert report.metrics["structured_retention"]["outcome"] == NA
    # Remaining graded metrics still pass -> overall PASS.
    assert report.overall == PASS


def test_report_niah_participates_in_overall():
    passing = _passing_inputs()
    skipped = build_report(
        **_base_report_kwargs(), **passing, niah={"status": "skipped"}
    )
    assert skipped.overall == PASS  # skipped NIAH is NA

    failed = build_report(**_base_report_kwargs(), **passing, niah={"status": "fail"})
    assert failed.overall == FAIL  # a failing NIAH drags overall down

    passed = build_report(**_base_report_kwargs(), **passing, niah={"status": "pass"})
    assert passed.overall == PASS


def test_human_summary_renders():
    report = build_report(**_base_report_kwargs(), **_passing_inputs())
    text = report.human_summary()
    assert "KV-quant differential quality gate" in text
    assert "ADVISORY" in text
    assert "OVERALL:" in text


# ---------------------------------------------------------------------------
# Harness guards (hermetic — no model load; the guards run before any import).
# ---------------------------------------------------------------------------
def _run_gate_kwargs(**overrides):
    base = dict(
        model_arg="dummy",
        candidate_dtypes=["int8"],
        prompts=[{"kind": "text", "prompt": "hi"}],
        max_tokens=8,
        mem_tokens=16,
        kv_group_size=64,
        advisory=True,
        run_niah=False,
        json_out=None,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "override",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"mem_tokens": 0},
        {"kv_group_size": 0},
        {"prompts": []},
    ],
)
def test_run_gate_rejects_bad_config_before_load(override):
    """RED-GREEN: non-positive token budgets / empty prompts raise up front.

    A zero/negative budget yields empty generations that would score a vacuous
    1.0 agreement — an enforced gate must never 'pass' without measuring. The
    guard runs before any mlx_lm import, so this is hermetic (no model load).
    """
    from scripts.kv_quant_quality_gate import run_gate

    with pytest.raises(ValueError):
        run_gate(**_run_gate_kwargs(**override))


def test_positive_int_argparse_type_rejects_nonpositive():
    import argparse

    from scripts.kv_quant_quality_gate import _positive_int

    assert _positive_int("48") == 48
    for bad in ("0", "-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(bad)


def test_niah_fail_closed_on_unknown_ram():
    """RED-GREEN: when RAM is undetected (None) NIAH must SKIP, not run.

    The chip qualifies (M3 Ultra) so the RAM guard is the deciding factor;
    unknown capacity must fail closed rather than assume headroom.
    """
    from scripts.kv_quant_quality_gate import _maybe_run_niah
    from vllm_mlx.chip_tier import classify_chip_tier

    chip = classify_chip_tier("Apple M3 Ultra")
    assert chip.is_m3_or_newer  # precondition — RAM guard is what decides
    result = _maybe_run_niah(
        None,
        None,
        chip,
        enabled=True,
        max_tokens=8,
        kv_bits=8,
        kv_group_size=64,
        eos_ids=set(),
        total_ram_gb=None,
    )
    assert result["status"] == "skipped"
    assert "undetected" in result["reason"] or "fail-closed" in result["reason"]


def test_niah_skips_when_not_requested_and_sub_m3():
    from scripts.kv_quant_quality_gate import _maybe_run_niah
    from vllm_mlx.chip_tier import classify_chip_tier

    m3 = classify_chip_tier("Apple M3 Ultra")
    m1 = classify_chip_tier("Apple M1")
    # Not requested -> skipped regardless of chip.
    r1 = _maybe_run_niah(
        None,
        None,
        m3,
        enabled=False,
        max_tokens=8,
        kv_bits=8,
        kv_group_size=64,
        eos_ids=set(),
        total_ram_gb=256.0,
    )
    assert r1["status"] == "skipped" and "not requested" in r1["reason"]
    # Sub-M3 chip -> skipped even when requested with ample RAM.
    r2 = _maybe_run_niah(
        None,
        None,
        m1,
        enabled=True,
        max_tokens=8,
        kv_bits=8,
        kv_group_size=64,
        eos_ids=set(),
        total_ram_gb=256.0,
    )
    assert r2["status"] == "skipped" and "below M3" in r2["reason"]


# ---------------------------------------------------------------------------
# Gated end-to-end smoke — real harness, cached small model only.
# ---------------------------------------------------------------------------
# Small, quantizable (dense, non-sliding-window) text models to try, smallest
# first. The smoke SKIPS unless one is already in the local HF cache.
_SMOKE_MODEL_CANDIDATES = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-8bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Phi-3-mini-4k-instruct-4bit",
]


# A COMPLETE snapshot needs config + tokenizer + at least one weights artifact.
# Probing only ``config.json`` would let ``mlx_lm.load`` fetch a missing shard /
# tokenizer over the network — violating the cache-only (no-download) contract.
_REQUIRED_CACHE_FILES = ("config.json", "tokenizer_config.json")
_WEIGHT_CANDIDATE_FILES = ("model.safetensors", "model.safetensors.index.json")


def _model_fully_cached(repo: str) -> bool:
    """True iff a COMPLETE local snapshot of ``repo`` is in the HF cache.

    Deterministic cache probe (``try_to_load_from_cache`` — a real ``str`` path
    means present) for config + tokenizer + a weights artifact. No network, no
    exception-message classification (the flaky pattern the gemma4 tests warned
    against).
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    def cached(filename: str) -> bool:
        try:
            return isinstance(try_to_load_from_cache(repo, filename), str)
        except Exception:
            return False

    if not all(cached(f) for f in _REQUIRED_CACHE_FILES):
        return False
    return any(cached(f) for f in _WEIGHT_CANDIDATE_FILES)


def _first_cached_model() -> str | None:
    """Return the first candidate with a COMPLETE local snapshot, else None."""
    for repo in _SMOKE_MODEL_CANDIDATES:
        if _model_fully_cached(repo):
            return repo
    return None


@pytest.mark.slow
def test_smoke_real_harness_on_cached_model(tmp_path, monkeypatch):
    """End-to-end: load a real cached small model, run the gate, assert the report.

    Cache-only (never downloads): skips unless a COMPLETE snapshot is cached, and
    forces HF offline mode so the load can NEVER reach the network. Run with
    ``pytest --run-slow -m slow tests/test_kv_quant_gate.py``.
    """
    pytest.importorskip("mlx_lm")
    repo = _first_cached_model()
    if repo is None:
        pytest.skip(
            "no COMPLETE small-model snapshot in the local HF cache — smoke is "
            "cache-only (no download). Pre-cache one of: "
            + ", ".join(_SMOKE_MODEL_CANDIDATES)
        )

    # Belt-and-suspenders: even though the snapshot is complete, force offline so
    # a stray fetch is impossible (raises instead of downloading).
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from scripts.kv_quant_quality_gate import run_gate

    out = tmp_path / "report.json"
    rc = run_gate(
        model_arg=repo,
        candidate_dtypes=["int8"],
        prompts=[
            {"kind": "text", "prompt": "Say hello in one word."},
            {
                "kind": "json",
                "prompt": 'Return ONLY {"ok": true} as JSON. No prose.',
            },
        ],
        max_tokens=16,
        mem_tokens=32,
        kv_group_size=64,
        advisory=True,  # advisory -> exit 0 regardless
        run_niah=False,
        json_out=out,
    )
    assert rc == 0  # advisory always 0

    import json

    reports = json.loads(out.read_text())
    assert len(reports) == 1
    rep = reports[0]
    assert rep["schema"] == 1
    assert rep["candidate_dtype"] == "int8"
    assert rep["baseline_dtype"] == "bf16"
    assert 0.0 <= rep["agreement"]["rate"] <= 1.0
    # int8 KV must actually save memory vs bf16 on a dense model.
    assert rep["memory"]["reduction_ratio"] > 1.0
    assert rep["memory"]["candidate_bytes"] < rep["memory"]["baseline_bytes"]
    assert rep["overall"] in {PASS, FAIL, NA}
    # chip tier must be recorded and internally consistent.
    chip = rep["chip"]
    assert "is_apple_silicon" in chip
    if chip["is_apple_silicon"]:
        assert chip["is_m3_or_newer"] == (chip["generation"] >= 3)
