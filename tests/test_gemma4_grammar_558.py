# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the Gemma-4 native grammar constraint (#558 E4).

The JSON-body families (hermes / qwen / harmony) constrain their ``arguments``
as a single ``%json`` object; Qwen3-Coder (E3) emits an XML arg body. Gemma-4
instead emits its own native wire ::

    <|tool_call>call:NAME{k1:<|"|>str<|"|>,k2:5,k3:true}<tool_call|>

so its ``structure_info()`` declares ``arg_style="gemma4"`` and ``build_tool_lark``
emits a DICTSORT-ordered, COMMA-separated ``key:VALUE`` body
(``_emit_gemma4_arg_body`` / ``_emit_gemma4_param_value``) instead of an XML body
or a whole-object ``%json``. These tests prove that path WITHOUT a decode loop:

  * ``build_tool_lark`` golden output for ``arg_style="gemma4"`` (``call:NAME{``
    frame, string values as the GREEDY ``gemma_str_value`` rule terminating on the
    ``<|"|>`` SPECIAL TOKEN, enum alternation with per-value ``<|"|>`` wrapping,
    ``%json`` scalars, and the first-present comma construction for required vs
    optional fields);
  * THE E4 FEASIBILITY RESULT: a string value is bounded by ``<|"|>`` — a SINGLE
    SPECIAL TOKEN, not a byte string like E3's ``</parameter>``. llguidance rejects
    a special token inside a ``[lazy]`` terminal, and a byte-literal ``<|"|>``
    cannot match the atomic token at runtime; the working construct is a GREEDY
    ``gemma_str_value: <|"|> GEMMA_STR_TEXT <|"|>`` (maximal-munch stops at the
    special-token boundary on its own). A value containing ``<``/``{``/``}``/``,``/
    quotes/newlines round-trips and terminates correctly;
  * grammar ENFORCEMENT via llguidance ``LLMatcher`` on the REAL gemma4 tokenizer:
    valid calls accepted + terminal, forced prose masked at token 0, bad enum /
    off-schema scalar rejected, and the comma construction admits exactly the valid
    optional subsets;
  * ROUND-TRIP: the ``gemma4`` parser parses the constrained wire back to
    ``{name, arguments}`` with correct types;
  * FAITHFUL-OR-OPT-OUT: the gemma4 representability guard reuses the SHARED
    strict-allowlist core (``_xml_schema_representable`` with the gemma4 wire
    policy) and differs only where the wire genuinely differs (bare ``\\w+`` keys;
    only the ``<|"|>`` marker is an unsafe enum-value byte — ``<``/newlines are
    SAFE, unlike XML);
  * REGRESSION GUARD: an ``arg_style="json"`` (hermes/qwen/harmony) build is
    byte-identical to before — the E4 change never leaks gemma4 constructs.

The enforcement / round-trip tests use the ACTUAL target model tokenizer
``mlx-community/gemma-4-e2b-it-4bit`` (pinned by revision). They skip ONLY on
genuine unavailability; the pure-Python golden / guard / regression tests never
skip.
"""

import importlib.util
import json

import pytest

_HAS_LLGUIDANCE = importlib.util.find_spec("llguidance") is not None
_requires_llguidance = pytest.mark.skipif(
    not _HAS_LLGUIDANCE, reason="llguidance ([guided] extra) not installed"
)

# The REAL Gemma-4 target model. Pin the revision so enforcement runs against an
# IMMUTABLE artifact (its <|tool_call>/<tool_call|>/<|"|> single-special-token
# layout — ids 48/49/52 — is fixed at this commit).
_TOKENIZER_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
_TOKENIZER_REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"
_GEMMA4_SENTINELS = ("<|tool_call>", "<tool_call|>", '<|"|>')

# A representative tool: required string + required enum + optional int + optional
# bool — exercises every ``_emit_gemma4_param_value`` branch (greedy string rule,
# per-value-wrapped enum alternation, ``%json`` scalar) AND the required-vs-optional
# comma construction.
GEMMA4_TOOLS = [
    {
        "name": "run",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "lang": {"type": "string", "enum": ["python", "cpp"]},
                "timeout": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
            "required": ["code", "lang"],
            "additionalProperties": False,
        },
    },
]

# All-OPTIONAL tool (no required field) — exercises the empty-body ``( ... )?`` and
# the first-present comma alternation for every optional subset.
GEMMA4_OPT_TOOL = [
    {
        "name": "cfg",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "additionalProperties": False,
        },
    },
]

# A nested-object tool with a ``$ref`` into ``$defs`` — proves ``$defs`` propagates
# into the per-value ``%json`` sub-schema.
GEMMA4_OBJ_TOOL = [
    {
        "name": "place",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"$ref": "#/$defs/point"}},
            "required": ["origin"],
            "$defs": {
                "point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    },
]


def _gemma4_structure_info(name: str):
    """The gemma4 wire triple, exactly as ``Gemma4ToolParser`` ships it
    (``arg_style="gemma4"``). Declared test-locally so the pure-Python golden
    tests need no tokenizer."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f"<|tool_call>call:{name}{{",
        end="}<tool_call|>",
        trigger="<|tool_call>",
        sentinels=_GEMMA4_SENTINELS,
        arg_style="gemma4",
    )


def _hermes_json_structure_info(name: str):
    """A hermes ``<tool_call>`` JSON-body wire triple (``arg_style="json"``, the
    default) — the regression baseline the E4 change must not perturb."""
    from vllm_mlx.api.tool_grammar import StructureInfo

    return StructureInfo(
        begin=f'<tool_call>\n{{"name": "{name}", "arguments": ',
        end="}\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
    )


# --------------------------------------------------------------------------
# Golden Lark for the gemma4 arg body (pure Python, always runs).
# --------------------------------------------------------------------------
_GEMMA4_GOLDEN_LARK = (
    "%llguidance {}\n"
    "start: (tag_0) (SEP (tag_0))* tag_end\n"
    "tag_end: TAG_TEXT\n"
    "SEP: /[ \\t\\r\\n]*/\n"
    "TAG_TEXT: /(.|\\n)*/\n"
    # The gemma4 string-value construct: GEMMA_STR_TEXT admits ANY byte; the RULE
    # (lowercase) wraps it in the <|"|> special-token markers (greedy — the special
    # token itself terminates it, so no [lazy] is needed / possible).
    "GEMMA_STR_TEXT: /(.|\\n)*/\n"
    'gemma_str_value: <|"|> GEMMA_STR_TEXT <|"|>\n'
    "\n"
    # The DICTSORT-ordered comma body, emitted as an O(n)-size RECURSIVE grammar:
    # the first-present head (required ``code`` at index 0, so it is the only
    # first-present candidate) sits in ``tag_0`` with NO leading comma, then defers
    # its comma-prefixed SUFFIX to the ``g0_rest<i>`` chain — each suffix nonterminal
    # is emitted ONCE and references the next (right recursion), so the grammar does
    # NOT regenerate the tail per alternative (previously O(n^2)). ``g0_rest1``
    # carries the required ``lang`` (bare comma separator + per-value <|"|>-wrapped
    # enum alternation); ``g0_rest2``/``g0_rest3`` carry each optional field's OWN
    # leading comma inside a ``( ... )?`` group (string -> greedy rule; scalar ->
    # bare %json).
    'tag_0: <|tool_call> "call:run{" ( "code:" gemma_str_value g0_rest1 ) "}" '
    "<tool_call|>\n"
    'g0_rest1: "," "lang:" (<|"|> "python" <|"|> | <|"|> "cpp" <|"|>) g0_rest2\n'
    'g0_rest2: ( "," "timeout:" %json {"type": "integer"} )? g0_rest3\n'
    'g0_rest3: ( "," "verbose:" %json {"type": "boolean"} )?\n'
)


def test_gemma4_lark_matches_golden():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(GEMMA4_TOOLS, "required", [_gemma4_structure_info("run")])
    assert lark == _GEMMA4_GOLDEN_LARK


def test_gemma4_lark_frame_and_sentinels():
    # The call frame: <|tool_call> trigger + <tool_call|> close + the <|"|> string
    # marker are all BARE special-token refs (never quoted byte literals the single
    # token could not satisfy). ``call:run{`` / ``}`` are ordinary byte literals.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(GEMMA4_TOOLS, "required", [_gemma4_structure_info("run")])
    assert " <|tool_call> " in lark  # bare trigger ref
    # The tag frame closes with a BARE ``<tool_call|>`` ref. (The O(n) suffix chain
    # emits ``g0_rest<i>`` rules AFTER the tag, so the grammar no longer ends on this
    # line — assert against the ``tag_0`` rule itself, not the whole grammar.)
    tag0_line = next(ln for ln in lark.splitlines() if ln.startswith("tag_0:"))
    assert tag0_line.endswith("<tool_call|>")  # bare closing ref
    assert '"<|tool_call>"' not in lark  # NOT quoted literals
    assert '"<tool_call|>"' not in lark
    assert '"<|\\"|>"' not in lark  # the <|"|> marker is a ref, never a byte literal
    assert '"call:run{"' in lark  # frame text is a byte literal
    assert '"}"' in lark


def test_gemma4_string_value_uses_greedy_rule_not_lazy():
    # THE E4 feasibility result, at the grammar-string level: a string value is the
    # GREEDY ``gemma_str_value`` RULE wrapping the ``<|"|>`` marker — NOT a
    # ``[lazy]`` construct (special tokens cannot live in a lazy terminal) and NOT a
    # byte literal for ``<|"|>`` (the atomic special token can never satisfy it).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(GEMMA4_TOOLS, "required", [_gemma4_structure_info("run")])
    assert 'gemma_str_value: <|"|> GEMMA_STR_TEXT <|"|>' in lark
    assert "GEMMA_STR_TEXT: /(.|\\n)*/" in lark
    assert '"code:" gemma_str_value' in lark
    # No lazy construct and no byte-literal marker.
    assert "[lazy]" not in lark
    assert "gemma_str_value[lazy]" not in lark


def test_gemma4_comma_and_required_optional_framing():
    # Required fields come first with a bare comma separator; each optional field
    # carries its OWN leading comma inside a ``( ... )?`` group (first-present
    # construction, now emitted as an O(n) ``g0_rest<i>`` suffix chain).
    # ``code``/``lang`` required; ``timeout``/``verbose`` optional.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(GEMMA4_TOOLS, "required", [_gemma4_structure_info("run")])
    # Required ``code`` is the first-present head (no leading comma), deferring its
    # comma-prefixed suffix to the shared ``g0_rest1`` nonterminal.
    assert '( "code:" gemma_str_value g0_rest1 )' in lark
    # The required ``lang`` suffix carries a BARE comma separator (never an optional
    # ``( )?`` wrapper — a required field can't be skipped).
    assert 'g0_rest1: "," "lang:"' in lark
    assert '( "," "lang:"' not in lark
    # Each optional scalar carries its own leading comma inside ``( ... )?``.
    assert '( "," "timeout:" %json {"type": "integer"} )?' in lark
    assert '( "," "verbose:" %json {"type": "boolean"} )?' in lark


def test_gemma4_enum_is_per_value_wrapped_alternation():
    # A STRING enum renders as an alternation with each value wrapped in its OWN
    # ``<|"|>`` marker pair (NOT one shared wrapper, NOT %json, NOT the greedy rule).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(GEMMA4_TOOLS, "required", [_gemma4_structure_info("run")])
    assert '(<|"|> "python" <|"|> | <|"|> "cpp" <|"|>)' in lark


def test_gemma4_all_optional_wraps_body_in_optional_group():
    # A tool with NO required field wraps the whole first-present alternation in an
    # outer ``( ... )?`` so an empty ``{}`` body is admitted.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(GEMMA4_OPT_TOOL, "required", [_gemma4_structure_info("cfg")])
    # First-present alternation over {a-first, b-first}, wrapped in an outer ``?``;
    # the a-first branch defers b's comma-suffix to the shared ``g0_rest1`` rule.
    assert (
        '"call:cfg{" ( "a:" gemma_str_value g0_rest1 '
        '| "b:" %json {"type": "integer"} )? "}" <tool_call|>' in lark
    )
    assert 'g0_rest1: ( "," "b:" %json {"type": "integer"} )?' in lark
    # The comma-prefixed suffix lives ONLY in the shared rule, never inlined into the
    # first-present alternation (that inlining was the O(n^2) construction).
    assert 'gemma_str_value ( "," "b:"' not in lark
    # The body group closes with the outer ``?`` right before the ``}`` frame.
    assert ')? "}" <tool_call|>' in lark


def test_gemma4_optional_grammar_is_linear_size_not_quadratic():
    # FIX #2 (codex r1): the first-present body must be an O(n)-size RECURSIVE
    # grammar — each comma-prefixed suffix emitted ONCE as a ``g0_rest<i>``
    # nonterminal — NOT the old inline construction that regenerated the entire
    # remaining suffix for every first-present alternative (O(n^2) grammar size AND
    # construction time, a request-controlled CPU/memory amplifier).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    def _build(n):
        props = {f"p{i:02d}": {"type": "integer"} for i in range(n)}
        tool = [
            {
                "name": "cfg",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "additionalProperties": False,
                },
            }
        ]
        lark = build_tool_lark(tool, "required", [_gemma4_structure_info("cfg")])
        rest_defs = sum(
            1 for ln in lark.splitlines() if ln.startswith("g0_rest") and ":" in ln
        )
        frag = lark.count('%json {"type": "integer"}')
        return lark, rest_defs, frag

    lark2, rest2, frag2 = _build(2)
    lark8, rest8, frag8 = _build(8)

    # Exactly n-1 suffix nonterminals -> each suffix emitted ONCE (linear rule count,
    # not one regenerated copy per alternative).
    assert (rest2, rest8) == (1, 7)
    # Each field's value fragment appears O(1) times (first-present head + its suffix
    # rule) -> exactly 2n-1: LINEAR. The old inline body was n(n+1)/2: QUADRATIC.
    assert (frag2, frag8) == (3, 15)
    assert frag8 < 8 * (8 + 1) // 2  # 15 < 36 (the would-be quadratic count)
    # 4x the fields must not blow up super-linearly: linear grows ~5x here, quadratic
    # would be ~12x. Guard the growth ratio well under quadratic.
    assert frag8 <= 6 * frag2  # 15 <= 18 holds for linear; quadratic (36) would fail
    # Byte size also grows sub-quadratically (each added field adds a bounded chunk).
    assert len(lark8) < 4 * len(lark2)


def test_gemma4_dictsort_order_is_case_insensitive():
    # FIX #4 (codex r1): property order must match Gemma's chat template, which uses
    # Jinja ``dictsort`` (default ``case_sensitive=False``) -> sort by the LOWERCASED
    # key. A case-SENSITIVE ``sorted`` would order uppercase keys before lowercase
    # ones and force the model off the order it was trained to emit.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    props = {  # insertion order deliberately NEITHER sorted order
        "Zebra": {"type": "string"},
        "apple": {"type": "string"},
        "Mango": {"type": "string"},
    }
    tool = [
        {
            "name": "cfg",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": ["Zebra", "apple", "Mango"],
                "additionalProperties": False,
            },
        }
    ]
    lark = build_tool_lark(tool, "required", [_gemma4_structure_info("cfg")])
    # dictsort (case-insensitive): apple (a) < Mango (m) < Zebra (z).
    i_apple, i_mango, i_zebra = (
        lark.index(f'"{k}:"') for k in ("apple", "Mango", "Zebra")
    )
    assert i_apple < i_mango < i_zebra
    # Case-SENSITIVE ``sorted`` would put uppercase Mango/Zebra before lowercase
    # ``apple`` (M, Z < a) -> the exact bug this guards. Assert we are NOT there.
    assert not (i_mango < i_apple)


def test_gemma4_dictsort_case_collision_is_stable_insertion_order():
    # A case-insensitive collision (``a`` vs ``A``) keeps the schema's INSERTION
    # order via Python's stable sort -> EXACTLY Jinja ``dictsort``'s tiebreak (a
    # secondary case-sensitive sort would instead force ``A`` before ``a``).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    props = {"a": {"type": "string"}, "A": {"type": "string"}}  # 'a' inserted first
    tool = [
        {
            "name": "cfg",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": ["a", "A"],
                "additionalProperties": False,
            },
        }
    ]
    lark = build_tool_lark(tool, "required", [_gemma4_structure_info("cfg")])
    assert lark.index('"a:"') < lark.index('"A:"')


def test_gemma4_noarg_tool_has_empty_body():
    # A no-argument tool renders the bare ``call:NAME{}`` frame (empty body).
    from vllm_mlx.api.tool_grammar import build_tool_lark

    noarg = [
        {
            "name": "ping",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]
    lark = build_tool_lark(noarg, "required", [_gemma4_structure_info("ping")])
    assert 'tag_0: <|tool_call> "call:ping{" "}" <tool_call|>' in lark
    # The string rule is still declared (arg_style is gemma4) but unused in the tag.
    assert "gemma_str_value: " in lark


def test_gemma4_ref_defs_propagated_into_value_schema():
    # A ``$ref`` object value carries the parent's ``$defs`` into its per-value
    # ``%json`` sub-schema so the ``$ref`` resolves.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    lark = build_tool_lark(
        GEMMA4_OBJ_TOOL, "required", [_gemma4_structure_info("place")]
    )
    expected_schema = {
        "$ref": "#/$defs/point",
        "$defs": {
            "point": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
    }
    assert f"%json {json.dumps(expected_schema)}" in lark


# --------------------------------------------------------------------------
# REGRESSION GUARD: arg_style="json" (hermes/qwen/harmony) byte-identical — the
# E4 change (and the shared-guard policy refactor) must not perturb JSON families.
# --------------------------------------------------------------------------
def test_json_family_grammar_has_no_gemma4_constructs():
    from vllm_mlx.api.tool_grammar import build_tool_lark

    infos = [_hermes_json_structure_info(t["name"]) for t in GEMMA4_TOOLS]
    lark = build_tool_lark(GEMMA4_TOOLS, "required", infos)
    assert "%json" in lark
    assert "GEMMA_STR_TEXT" not in lark
    assert "gemma_str_value" not in lark
    assert "call:" not in lark
    assert '<|"|>' not in lark


def test_json_family_grammar_byte_identical_to_baseline():
    # The exact hermes forced golden (identical to test_qwen3coder_xml_grammar_558's
    # baseline). Pinning it proves E4 left the JSON-family grammar byte-for-byte
    # unchanged.
    from vllm_mlx.api.tool_grammar import build_tool_lark

    tools = [
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
        }
    ]
    infos = [_hermes_json_structure_info("get_weather")]
    expected = (
        "%llguidance {}\n"
        "start: (tag_0) (SEP (tag_0))* tag_end\n"
        "tag_end: TAG_TEXT\n"
        "SEP: /[ \\t\\r\\n]*/\n"
        "TAG_TEXT: /(.|\\n)*/\n"
        "\n"
        'tag_0: <tool_call> "\\n{\\"name\\": \\"get_weather\\", '
        '\\"arguments\\": " %json {"type": "object", "properties": {"city": '
        '{"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, '
        '"required": ["city"], "additionalProperties": false} "}\\n" </tool_call>\n'
    )
    assert build_tool_lark(tools, "required", infos) == expected


# --------------------------------------------------------------------------
# Real Gemma4ToolParser.structure_info() (pure Python, hermetic tokenizer stub).
# --------------------------------------------------------------------------
class _FakeAddedToken:
    def __init__(self, content, special=True):
        self.content = content
        self.special = special


class _FakeTokenizer:
    """Models the ``are_single_special_tokens`` guard surfaces: the three gemma4
    markers as distinct single ADDED tokens that round-trip (real ids 48/49/52)."""

    def __init__(self, added=None):
        self._added = dict(added or {})
        self._id_to_str = {i: s for s, i in self._added.items()}
        self.added_tokens_decoder = {
            i: _FakeAddedToken(s) for s, i in self._added.items()
        }

    def encode(self, text, add_special_tokens=False):
        if text in self._added:
            return [self._added[text]]
        return [0, 1]  # ordinary multi-token text

    def decode(self, ids):
        return "".join(self._id_to_str.get(i, "<unk>") for i in ids)

    def get_vocab(self):
        return dict(self._added)


def _single_token_tokenizer():
    return _FakeTokenizer(added={"<|tool_call>": 48, "<tool_call|>": 49, '<|"|>': 52})


def _make_gemma4(tokenizer=None):
    from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

    return Gemma4ToolParser(tokenizer=tokenizer)


def test_gemma4_structure_info_opts_out_without_tokenizer():
    assert _make_gemma4(tokenizer=None).structure_info() is None


def test_gemma4_structure_info_opts_out_on_multitoken_tokenizer():
    # A tokenizer that encodes the markers as ordinary multi-token text -> opt out.
    assert _make_gemma4(tokenizer=_FakeTokenizer(added={})).structure_info() is None


def test_gemma4_structure_info_opts_out_when_marker_missing():
    # Missing even ONE of the three markers (here <|"|>) -> opt out (an
    # unenforceable string-value rule would otherwise be emitted).
    tok = _FakeTokenizer(added={"<|tool_call>": 48, "<tool_call|>": 49})
    assert _make_gemma4(tokenizer=tok).structure_info() is None


def test_gemma4_structure_info_returns_gemma4_wire_triple():
    from vllm_mlx.api.tool_grammar import StructureInfo

    get_info = _make_gemma4(tokenizer=_single_token_tokenizer()).structure_info()
    assert callable(get_info), "opt-in must return a name->StructureInfo factory"
    si = get_info("run")
    assert isinstance(si, StructureInfo)
    assert si.arg_style == "gemma4"
    assert si.trigger == "<|tool_call>"
    assert si.begin == "<|tool_call>call:run{"
    assert si.end == "}<tool_call|>"
    assert si.begin.startswith(si.trigger)  # builder invariant
    assert si.sentinels == _GEMMA4_SENTINELS
    assert si.trigger in si.sentinels


def test_gemma4_supports_grammar_and_auto_unsafe():
    from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

    assert Gemma4ToolParser.SUPPORTS_GRAMMAR is True
    assert Gemma4ToolParser.supports_grammar() is True
    # AUTO stays free-form (channel reasoning + special-token markers).
    assert Gemma4ToolParser.TOOL_GRAMMAR_AUTO_SAFE is False


# --------------------------------------------------------------------------
# Grammar ENFORCEMENT + ROUND-TRIP on the REAL gemma4 tokenizer.
# --------------------------------------------------------------------------
def _offline_skip_exc_types():
    # ``OSError`` is included UNCONDITIONALLY: Transformers commonly wraps an
    # unavailable tokenizer/cache as a plain ``OSError`` ("Can't load ... offline"),
    # so a genuinely-offline run must SKIP (not FAIL) on it. The previous
    # ``tuple(types) or (OSError,)`` dropped ``OSError`` whenever either optional
    # import below succeeded — turning an offline run into a failure. The narrower
    # connectivity exceptions are appended (still skip-worthy, more specific) but
    # never displace ``OSError``.
    types: list[type[BaseException]] = [OSError]
    try:
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )

        types += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:  # pragma: no cover - old hub without these names
        pass
    try:
        from requests.exceptions import ConnectionError as _ReqConnErr

        types.append(_ReqConnErr)
    except Exception:  # pragma: no cover - requests not present
        pass
    return tuple(types)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, revision=_TOKENIZER_REVISION
        )
    except _offline_skip_exc_types():  # pragma: no cover - offline & uncached
        pytest.skip(
            f"tokenizer {_TOKENIZER_MODEL}@{_TOKENIZER_REVISION[:8]} not cached "
            "and no network — gemma4 enforcement tests require it"
        )


@pytest.fixture(scope="module")
def lltok(tok):
    """Build an llguidance LLTokenizer via the module's own resolver. Skip ONLY
    when the runtime bridge is genuinely unavailable (mirrors E3 finding 5)."""
    from vllm_mlx.api.tool_grammar import HAS_LL_TOKENIZER, build_lltokenizer

    if not HAS_LL_TOKENIZER:
        pytest.skip(
            "llguidance runtime bridge (llguidance.hf / LLTokenizer) not "
            "installed — gemma4 enforcement tests require it"
        )
    built = build_lltokenizer(tok)
    assert built is not None, (
        "build_lltokenizer() returned None for the CACHED gemma4 tokenizer with "
        "the llguidance runtime bridge available — the tokenizer->llguidance "
        "integration is BROKEN (this must FAIL, not skip)."
    )
    return built


def _gemma4_grammar(tools, tool_choice, tok):
    """Compile the gemma4 grammar through the REAL parser (opted-in on ``tok``)."""
    from vllm_mlx.api.tool_grammar import build_tool_grammar

    return build_tool_grammar(tools, tool_choice, _make_gemma4(tok))


def _consume(grammar, lltok, tok, text):
    """Offline enforcement probe. Returns ``(accepted, total, is_accepting)`` —
    advances real matcher state so ``is_accepting()`` proves a COMPLETE valid
    derivation, not merely an accepted prefix."""
    from llguidance.mlx import LLMatcher

    ids = tok.encode(text, add_special_tokens=False)
    matcher = LLMatcher(lltok, grammar)
    assert not matcher.get_error(), matcher.get_error()
    accepted = 0
    for tid in ids:
        if not matcher.consume_tokens([tid]):
            break
        accepted += 1
    return accepted, len(ids), matcher.is_accepting()


def _wire(code, *, lang="python", timeout=None, verbose=None):
    """Build the gemma4 wire for ``run`` (keys in DICTSORT order)."""
    s = f'<|tool_call>call:run{{code:<|"|>{code}<|"|>,lang:<|"|>{lang}<|"|>'
    if timeout is not None:
        s += f",timeout:{timeout}"
    if verbose is not None:
        s += f",verbose:{verbose}"
    s += "}<tool_call|>"
    return s


def _parse(wire, tools):
    parser = _make_gemma4(tokenizer=None)
    req = {
        "tools": [
            {
                "type": "function",
                "function": {"name": t["name"], "parameters": t["parameters"]},
            }
            for t in tools
        ]
    }
    res = parser.extract_tool_calls(wire, request=req)
    assert res.tools_called, "parser did not detect the tool call"
    tc = res.tool_calls[0]
    return tc["name"], json.loads(tc["arguments"])


@_requires_llguidance
def test_gemma4_finding_tokenizer_llguidance_integration_not_broken(tok):
    from vllm_mlx.api.tool_grammar import HAS_LL_TOKENIZER, build_lltokenizer

    if not HAS_LL_TOKENIZER:
        pytest.skip("llguidance runtime bridge (llguidance.hf / LLTokenizer) absent")
    assert build_lltokenizer(tok) is not None


@_requires_llguidance
def test_gemma4_markers_are_single_special_tokens(tok):
    # Ground truth (probe a): the three wire markers are single special tokens.
    for marker, expected_id in (
        ("<|tool_call>", 48),
        ("<tool_call|>", 49),
        ('<|"|>', 52),
    ):
        ids = tok.encode(marker, add_special_tokens=False)
        assert ids == [expected_id], (marker, ids)
        assert tok.decode(ids) == marker


@_requires_llguidance
def test_gemma4_valid_call_accepted_and_terminates(tok, lltok):
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    assert grammar is not None
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire("print(1)"))
    assert accepted == total, f"valid gemma4 call rejected ({accepted}/{total})"
    assert accepting, "valid complete gemma4 call is not a terminal state"


@_requires_llguidance
@pytest.mark.parametrize(
    "code",
    [
        "a < b && c > d",
        "<html><body>x</body></html>",
        "vector<int> v",
        "if (a < b) { return a; }",
        "obj = {x: 1, y: 2}",
        "line1\nline2\nline3",
        'she said "hi", ok',
    ],
)
def test_gemma4_string_value_with_special_chars_is_accepted(code, tok, lltok):
    # THE E4 result: a string value containing <, >, {, }, comma, quotes, or
    # newlines is ACCEPTED in full and terminates at the closing ``<|"|>`` SPECIAL
    # TOKEN (the greedy rule stops there because a special token isn't a byte).
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    assert grammar is not None
    accepted, total, accepting = _consume(grammar, lltok, tok, _wire(code, lang="cpp"))
    assert accepted == total, (
        f"special-char string value rejected ({accepted}/{total}) for {code!r}"
    )
    assert accepting, f"special-char value {code!r} is not a terminal state"


@_requires_llguidance
def test_gemma4_optional_scalars_enforced(tok, lltok):
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    accepted, total, accepting = _consume(
        grammar, lltok, tok, _wire("x", lang="cpp", timeout=30, verbose="true")
    )
    assert accepted == total and accepting, (
        f"valid call with optional scalars rejected ({accepted}/{total})"
    )


@_requires_llguidance
def test_gemma4_bad_enum_value_is_rejected(tok, lltok):
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    accepted, total, _ = _consume(grammar, lltok, tok, _wire("x", lang="rust"))
    assert accepted < total, "invalid enum value was NOT rejected by the grammar"


@_requires_llguidance
def test_gemma4_off_schema_scalar_is_rejected(tok, lltok):
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    bad = '<|tool_call>call:run{code:<|"|>x<|"|>,lang:<|"|>python<|"|>,timeout:notanint'
    accepted, total, _ = _consume(grammar, lltok, tok, bad)
    assert accepted < total, "off-schema (non-integer) timeout was NOT rejected"


@_requires_llguidance
def test_gemma4_missing_required_field_is_rejected(tok, lltok):
    # Dropping the required ``lang`` field must not reach a terminal state.
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    wire = '<|tool_call>call:run{code:<|"|>x<|"|>}<tool_call|>'
    accepted, total, accepting = _consume(grammar, lltok, tok, wire)
    assert not (accepted == total and accepting), "missing required field accepted"


@_requires_llguidance
def test_gemma4_forced_rejects_prose_before_the_call(tok, lltok):
    grammar = _gemma4_grammar(GEMMA4_TOOLS, "required", tok)
    assert grammar is not None
    prose_then_call = "Sure, let me run that. " + _wire("print(1)")
    accepted, _total, _ = _consume(grammar, lltok, tok, prose_then_call)
    assert accepted == 0, (
        f"forced gemma4 grammar accepted {accepted} prose token(s) before the "
        "trigger — the unbounded leading prefix is back (#558 forced-leak)"
    )


@_requires_llguidance
def test_gemma4_auto_opts_out_required_builds(tok):
    # AUTO stays free-form (TOOL_GRAMMAR_AUTO_SAFE=False); required/named build.
    assert _gemma4_grammar(GEMMA4_TOOLS, "auto", tok) is None
    assert _gemma4_grammar(GEMMA4_TOOLS, "required", tok) is not None


@_requires_llguidance
@pytest.mark.parametrize(
    "present",
    [
        "<|tool_call>call:cfg{}<tool_call|>",
        '<|tool_call>call:cfg{a:<|"|>x<|"|>}<tool_call|>',
        "<|tool_call>call:cfg{b:5}<tool_call|>",
        '<|tool_call>call:cfg{a:<|"|>x<|"|>,b:5}<tool_call|>',
    ],
)
def test_gemma4_all_optional_comma_subsets_accepted(present, tok, lltok):
    # The first-present comma construction admits EVERY valid optional subset of an
    # all-optional tool — empty, a-only, b-only, both — each accepted + terminal.
    grammar = _gemma4_grammar(GEMMA4_OPT_TOOL, "required", tok)
    accepted, total, accepting = _consume(grammar, lltok, tok, present)
    assert accepted == total and accepting, f"optional subset rejected: {present!r}"


@_requires_llguidance
def test_gemma4_all_optional_rejects_leading_comma(tok, lltok):
    # A misplaced LEADING comma (the bug a naive per-field ``( )?`` would allow when
    # the first optional is absent) must be rejected.
    grammar = _gemma4_grammar(GEMMA4_OPT_TOOL, "required", tok)
    bad = "<|tool_call>call:cfg{,b:5}<tool_call|>"
    accepted, total, accepting = _consume(grammar, lltok, tok, bad)
    assert not (accepted == total and accepting), "leading-comma body was accepted"


# --------------------------------------------------------------------------
# ROUND-TRIP: the gemma4 parser parses the constrained wire back to
# {name, arguments} with correct types.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "code", ["a < b && c > d", "vector<int> v", "obj = {x:1}", "print('ok')"]
)
def test_gemma4_roundtrip_string_value_with_special_chars(code):
    name, args = _parse(_wire(code), GEMMA4_TOOLS)
    assert name == "run"
    assert args["code"] == code
    assert args["lang"] == "python"


def test_gemma4_roundtrip_scalar_types_int_bool():
    name, args = _parse(
        _wire("x", lang="cpp", timeout=30, verbose="true"), GEMMA4_TOOLS
    )
    assert args["timeout"] == 30 and isinstance(args["timeout"], int)
    assert args["verbose"] is True
    assert args["lang"] == "cpp"


def test_gemma4_roundtrip_nested_object_value():
    # A nested-object ($ref) value round-trips into a JSON object with typed fields
    # — the ``%json`` (standard-JSON) surface the grammar emits is what the lenient
    # gemma4 parser json-decodes.
    wire = '<|tool_call>call:place{origin:{"x": 3, "y": 4}}<tool_call|>'
    name, args = _parse(wire, GEMMA4_OBJ_TOOL)
    assert name == "place"
    assert args["origin"] == {"x": 3, "y": 4}


@_requires_llguidance
def test_gemma4_object_value_via_json_enforced_and_roundtrips(tok, lltok):
    grammar = _gemma4_grammar(GEMMA4_OBJ_TOOL, "required", tok)
    assert grammar is not None
    wire = '<|tool_call>call:place{origin:{"x": 3, "y": 4}}<tool_call|>'
    accepted, total, accepting = _consume(grammar, lltok, tok, wire)
    assert accepted == total and accepting, (
        f"object %json value rejected ({accepted}/{total}, accepting={accepting})"
    )
    name, args = _parse(wire, GEMMA4_OBJ_TOOL)
    assert name == "place" and args["origin"] == {"x": 3, "y": 4}


# ==========================================================================
# FAITHFUL-OR-OPT-OUT: the gemma4 representability guard reuses the SHARED
# strict-allowlist core (``_xml_schema_representable`` + gemma4 wire policy) and
# differs ONLY in the two wire-specific leaf checks.
# ==========================================================================
def test_gemma4_representable_common_and_noarg():
    from vllm_mlx.api.tool_grammar import _gemma4_schema_representable as rep

    assert rep(GEMMA4_TOOLS[0]["parameters"]) is True
    assert rep(GEMMA4_OPT_TOOL[0]["parameters"]) is True
    assert rep(GEMMA4_OBJ_TOOL[0]["parameters"]) is True  # $ref -> object
    assert rep({"type": "object", "properties": {}, "additionalProperties": False})
    assert rep({}) is True  # allow-any / no-arg


def test_gemma4_representable_shares_structural_allowlist_with_xml():
    # The structural allowlist (object-level keywords, string facets, required
    # totality, $ref) is SHARED, so gemma4 opts out on exactly the same shapes.
    from vllm_mlx.api.tool_grammar import _gemma4_schema_representable as rep

    assert rep(False) is False
    assert rep({"type": "object", "required": ["a"]}) is False  # property-less
    assert rep({"type": "array"}) is False  # non-object top-level
    assert rep({"properties": {"s": {"type": "string", "pattern": "^x$"}}}) is False
    props = {"a": {"type": "string"}, "b": {"type": "string"}}
    assert (
        rep({"type": "object", "properties": props, "dependentRequired": {"a": ["b"]}})
        is False
    )
    assert rep({"type": "object", "properties": props, "minProperties": 1}) is False
    assert rep({"properties": {"c": {"$ref": "#/$defs/missing"}, "$defs": {}}}) is False
    # Total ``required`` guard (no crash on unhashable member).
    base = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert rep({**base, "required": [["a"]]}) is False


def test_gemma4_key_safety_is_word_only_stricter_than_xml():
    # gemma4 emits a BARE ``KEY:``; the parser reads ``\w+``, so only ``\w+`` keys
    # round-trip — STRICTER than XML (which allows ``-``/``.``).
    from vllm_mlx.api.tool_grammar import (
        _gemma4_schema_representable as g_rep,
    )
    from vllm_mlx.api.tool_grammar import (
        _xml_schema_representable as x_rep,
    )

    assert g_rep({"properties": {"my_key1": {"type": "string"}}}) is True
    for bad in ("my-key", "a.b", "a b", "a:b", "a,b", "x<y", "a{b"):
        assert g_rep({"properties": {bad: {"type": "string"}}}) is False, bad
    # XML allows ``-``/``.`` (its key wire tolerates them) — proves the divergence
    # is intentional and the shared structural guard is otherwise identical.
    assert x_rep({"properties": {"my-key": {"type": "string"}}}) is True
    assert g_rep({"properties": {"my-key": {"type": "string"}}}) is False


def test_gemma4_enum_delimiter_safety_differs_from_xml():
    # A gemma4 string value is bounded ONLY by ``<|"|>``, so an enum value
    # containing ``<``/``>``/newlines is SAFE (unlike XML). A value containing the
    # ``<|"|>`` marker opts out.
    from vllm_mlx.api.tool_grammar import (
        _gemma4_schema_representable as g_rep,
    )
    from vllm_mlx.api.tool_grammar import (
        _xml_schema_representable as x_rep,
    )

    safe_lt = {"properties": {"s": {"type": "string", "enum": ["a<b", "c>d"]}}}
    assert g_rep(safe_lt) is True  # gemma4: safe inside <|"|>
    assert x_rep(safe_lt) is False  # XML: ``<``/``>`` desync the tag
    # Newline inside a gemma4 string value is fine.
    assert g_rep({"properties": {"s": {"type": "string", "enum": ["a\nb"]}}}) is True
    # But the <|"|> marker literally inside a value opts out (would close early).
    assert (
        g_rep({"properties": {"s": {"type": "string", "enum": ['a<|"|>b']}}}) is False
    )


def test_gemma4_enum_all_structural_markers_opt_out():
    # FIX #1 (codex r2): the enum-wire guard must reject EVERY gemma4 structural
    # special token, not only ``<|"|>``. Each — the string delimiter ``<|"|>`` AND
    # the ``<|tool_call>``/``<tool_call|>`` frame tokens AND the reasoning-channel
    # ``<|channel>``/``<channel|>`` tokens — is a SINGLE special token, so a byte
    # rendering of it inside an enum value can never be produced at runtime and would
    # compile to a DEAD alternation branch. Faithful-or-opt-out: opt the whole
    # request out instead of emitting an unreachable branch.
    from vllm_mlx.api.tool_grammar import (
        _GEMMA4_STRUCTURAL_MARKERS,
    )
    from vllm_mlx.api.tool_grammar import (
        _gemma4_schema_representable as g_rep,
    )

    # The guard's marker set is exactly the five documented structural tokens.
    assert set(_GEMMA4_STRUCTURAL_MARKERS) == {
        "<|tool_call>",
        "<tool_call|>",
        '<|"|>',
        "<|channel>",
        "<channel|>",
    }
    for marker in _GEMMA4_STRUCTURAL_MARKERS:
        schema = {"properties": {"s": {"type": "string", "enum": [f"a{marker}b"]}}}
        assert g_rep(schema) is False, marker
        # A marker at the very start / end of the value opts out too.
        assert (
            g_rep({"properties": {"s": {"enum": [marker], "type": "string"}}}) is False
        )
    # Control: values containing the markers' INDIVIDUAL safe bytes (``<``/``>``/
    # ``|``) but NOT a full marker substring stay representable (gemma4 permits raw
    # ``<``/``>`` inside the ``<|"|>`` pair — the whole point vs XML).
    assert (
        g_rep(
            {
                "properties": {
                    "s": {"type": "string", "enum": ["a<b>c", "x|y", "|>", "<|"]}
                }
            }
        )
        is True
    )


def test_gemma4_enum_type_consistency_shared():
    # The shared enum guard still rejects value/declared-type mismatch and
    # unsupported siblings.
    from vllm_mlx.api.tool_grammar import _gemma4_schema_representable as rep

    assert rep({"properties": {"n": {"type": "integer", "enum": ["x"]}}}) is False
    assert rep({"properties": {"s": {"enum": ["a", "bb"], "minLength": 2}}}) is False
    # Control: clean string / numeric enums stay representable.
    assert rep({"properties": {"s": {"type": "string", "enum": ["a", "b"]}}}) is True
    assert rep({"properties": {"n": {"type": "number", "enum": [1.5, 2]}}}) is True


# ---- OPT-OUT through build_tool_grammar (real parser) --------------------
@_requires_llguidance
def test_gemma4_build_tool_grammar_opts_out_bad_key(tok):
    # Control: a representable gemma4 tool DOES build a grammar.
    assert _gemma4_grammar(GEMMA4_TOOLS, "required", tok) is not None
    bad_key_tool = [
        {
            "name": "run",
            "parameters": {
                "type": "object",
                "properties": {"a-b": {"type": "string"}},
                "required": ["a-b"],
            },
        }
    ]
    assert _gemma4_grammar(bad_key_tool, "required", tok) is None


@_requires_llguidance
def test_gemma4_build_tool_grammar_opts_out_string_pattern(tok):
    pat_tool = [
        {
            "name": "run",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "pattern": "^[a-z]+$"}},
                "required": ["code"],
            },
        }
    ]
    assert _gemma4_grammar(pat_tool, "required", tok) is None


@_requires_llguidance
@pytest.mark.parametrize(
    "marker", ["<|tool_call>", "<tool_call|>", '<|"|>', "<|channel>", "<channel|>"]
)
def test_gemma4_build_tool_grammar_opts_out_enum_with_structural_marker(marker, tok):
    # FIX #1 (codex r2), end-to-end on the REAL tokenizer: an enum value whose wire
    # form embeds ANY gemma4 structural special token would compile to a DEAD
    # byte-literal alternation branch (the token is emitted atomically, never as its
    # bytes). ``build_tool_grammar`` must OPT OUT (return None -> free-form), never
    # ship an unreachable branch. A clean enum (``lang`` in GEMMA4_TOOLS) still builds
    # (control below), so the opt-out is specific to the marker-bearing value.
    assert (
        _gemma4_grammar(GEMMA4_TOOLS, "required", tok) is not None
    )  # clean enum builds
    bad_enum_tool = [
        {
            "name": "run",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "lang": {"type": "string", "enum": ["python", f"c{marker}pp"]},
                },
                "required": ["code", "lang"],
                "additionalProperties": False,
            },
        }
    ]
    assert _gemma4_grammar(bad_enum_tool, "required", tok) is None, marker
