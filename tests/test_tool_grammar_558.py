# SPDX-License-Identifier: Apache-2.0
"""Offline tests for grammar-constrained tool calling (#558) PILOT.

These validate the grammar BUILDER and negative-control masking without a
model — they only need the hermes tokenizer (fast) + llguidance. If the
tokenizer/llguidance is unavailable the tests skip.
"""

import numpy as np
import pytest

pytest.importorskip("llguidance")

TOOLS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
            "additionalProperties": False,
        },
    },
]

_TOKENIZER_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)
    except Exception:  # pragma: no cover - network / cache miss
        pytest.skip("hermes-wire tokenizer not available")


def _abc_default_is_none():
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    class _Dummy(ToolParser):
        EXPECTED_WIRE_FORMATS = ("tool_call_json",)

        def extract_tool_calls(self, model_output, request=None):  # noqa: D401
            raise NotImplementedError

    return _Dummy().structure_info()


def test_abc_structure_info_defaults_to_none():
    # PR-1 non-breaking contract: a parser that does not override
    # ``structure_info`` returns None (free-form fallback).
    assert _abc_default_is_none() is None


def test_hermes_builds_grammar(tok):
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    parser = HermesToolParser(tokenizer=tok)
    assert build_tool_grammar(TOOLS, "required", parser) is not None
    assert build_tool_grammar(TOOLS, "auto", parser) is not None


def test_qwen_builds_grammar(tok):
    from vllm_mlx.tool_parsers.qwen_tool_parser import QwenToolParser
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    parser = QwenToolParser(tokenizer=tok)
    assert build_tool_grammar(TOOLS, "required", parser) is not None


def _feed(proc, lltok, tok, text):
    """Feed target ids through the processor; return (blocked, blocked_idx)."""
    import mlx.core as mx

    ids = tok.encode(text, add_special_tokens=False)
    committed = []
    vocab = lltok.vocab_size
    for k, tid in enumerate(ids):
        ctx = mx.array(committed) if committed else mx.array([], dtype=mx.int32)
        masked = proc(ctx, mx.zeros((1, vocab)))
        row = np.array(masked[0])
        allowed = bool(np.isfinite(row[tid]) and row[tid] > -1e30)
        if not allowed:
            return True, k
        committed.append(tid)
    return False, len(ids)


def test_negative_controls_are_forbidden(tok):
    # The load-bearing #558 proof: the grammar MASK actively forbids a
    # hallucinated tool name and off-schema arguments — it is grammar-enforced,
    # not merely post-parsed.
    from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
    from vllm_mlx.api.tool_grammar import (
        build_tool_grammar,
        build_lltokenizer,
        GrammarLogitsProcessor,
    )

    parser = HermesToolParser(tokenizer=tok)
    grammar = build_tool_grammar(TOOLS, "required", parser)
    lltok = build_lltokenizer(tok)
    if grammar is None or lltok is None:
        pytest.skip("grammar/lltokenizer unavailable")

    # Valid call: fully allowed.
    blocked, _ = _feed(
        GrammarLogitsProcessor(lltok, grammar, tokenizer=tok),
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    )
    assert not blocked, "valid hermes call was unexpectedly blocked"

    # Hallucinated name: blocked.
    blocked, _ = _feed(
        GrammarLogitsProcessor(lltok, grammar, tokenizer=tok),
        lltok,
        tok,
        '<tool_call>\n{"name": "get_stockquote',
    )
    assert blocked, "hallucinated tool name was NOT blocked"

    # Off-schema (city as int): blocked.
    blocked, _ = _feed(
        GrammarLogitsProcessor(lltok, grammar, tokenizer=tok),
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": 4',
    )
    assert blocked, "off-schema integer argument was NOT blocked"

    # Bad enum: blocked.
    blocked, _ = _feed(
        GrammarLogitsProcessor(lltok, grammar, tokenizer=tok),
        lltok,
        tok,
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "P", "unit": "kelvin',
    )
    assert blocked, "invalid enum value was NOT blocked"
