import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "deepseek_context_replay.py"
SPEC = importlib.util.spec_from_file_location("deepseek_context_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_corpus_is_deterministic_and_bounded(tmp_path):
    (tmp_path / "vllm_mlx").mkdir()
    (tmp_path / "vllm_mlx" / "large.py").write_text("a" * 80)
    (tmp_path / "vllm_mlx" / "small.py").write_text("b" * 40)
    first = MODULE.build_corpus(tmp_path, 75)
    second = MODULE.build_corpus(tmp_path, 75)
    assert first == second
    assert len(first) == 75
    assert "large.py" in first


def test_build_corpus_rejects_undersized_source(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tiny.py").write_text("pass")
    try:
        MODULE.build_corpus(tmp_path, 10_000)
    except ValueError as exc:
        assert "source corpus" in str(exc)
    else:
        raise AssertionError("expected undersized corpus to fail")


def test_build_corpus_does_not_follow_source_symlinks(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("TOP_SECRET")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "linked.py").symlink_to(outside)
    (tmp_path / "scripts" / "safe.py").write_text("s" * 100)
    corpus = MODULE.build_corpus(tmp_path, 60)
    assert "TOP_SECRET" not in corpus
    assert "safe.py" in corpus


def test_build_corpus_places_score_evidence_first(tmp_path):
    (tmp_path / "vllm_mlx").mkdir()
    (tmp_path / "vllm_mlx" / "large.py").write_text("x" * 200)
    (tmp_path / "vllm_mlx" / "model_profile.py").write_text("class ModelProfile: pass")
    corpus = MODULE.build_corpus(tmp_path, 80)
    assert corpus.startswith("\n--- FILE: vllm_mlx/model_profile.py ---")
    assert "class ModelProfile" in corpus


def test_build_corpus_rejects_truncated_score_evidence(tmp_path):
    (tmp_path / "vllm_mlx").mkdir()
    (tmp_path / "vllm_mlx" / "model_profile.py").write_text(
        "class ModelProfile:\n" + "    pass\n" * 20
    )
    try:
        MODULE.build_corpus(tmp_path, 40)
    except ValueError as exc:
        assert "complete scoring evidence" in str(exc)
    else:
        raise AssertionError("expected truncated evidence target to fail")


def test_detect_repetition_finds_exact_repeated_suffix():
    prefix = "unique context "
    cycle = "one two three four five six seven eight "
    repeated, period = MODULE.detect_repetition(prefix + cycle * 3)
    assert repeated
    assert period == 8


def test_detect_repetition_finds_short_period_degradation():
    assert MODULE.detect_repetition("prefix " + "error " * 20) == (True, 1)


def test_detect_repetition_ignores_normal_short_answer():
    assert MODULE.detect_repetition("vllm_mlx/model_profile.py") == (False, None)


def test_build_conversation_preserves_complete_prior_turns():
    items = MODULE.build_conversation("a" * 12, turn_chars=5)
    assert [item["role"] for item in items] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert items[1]["content"][0]["text"] == MODULE.TURN_ACKNOWLEDGEMENT
    assert MODULE.TURN_ACKNOWLEDGEMENT in items[0]["content"][0]["text"]
    assert (
        "Which file defines class ModelProfile?" not in items[0]["content"][0]["text"]
    )
    assert "Which file defines class ModelProfile?" in items[-1]["content"][0]["text"]
    assert "a" * 5 in items[0]["content"][0]["text"]
    assert "a" * 2 in items[-1]["content"][0]["text"]


def test_build_conversation_rejects_non_positive_turn_size():
    try:
        MODULE.build_conversation("source", turn_chars=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected invalid turn size to fail")


def test_positive_int_rejects_non_positive_targets():
    assert MODULE.positive_int("1") == 1
    for value in ("0", "-1"):
        try:
            MODULE.positive_int(value)
        except argparse.ArgumentTypeError as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("expected invalid target size to fail")


def test_run_replay_records_transport_failure():
    class BrokenClient:
        def post(self, *args, **kwargs):
            raise RuntimeError("endpoint unavailable")

    result = MODULE.run_replay(
        BrokenClient(),
        endpoint="local",
        url="http://test/v1",
        model="model",
        corpus="source",
        target_chars=6,
    )
    assert result.status == "error"
    assert result.failure == "RuntimeError: endpoint unavailable"
    assert result.answer == ""


def test_replay_endpoint_validation_is_credential_safe_and_unique():
    assert MODULE.parse_endpoint("cloud=https://example.test/v1,CLOUD_KEY") == (
        "cloud",
        "https://example.test/v1",
        "CLOUD_KEY",
    )
    try:
        MODULE.parse_endpoint("cloud=https://example.test/v1,not-a-valid-env-name")
    except argparse.ArgumentTypeError as exc:
        assert "environment-variable name" in str(exc)
        assert "not-a-valid" not in str(exc)
    else:
        raise AssertionError("expected literal credential suffix to fail")
    try:
        MODULE.parse_endpoint("cloud=http://example.test/v1,CLOUD_KEY")
    except argparse.ArgumentTypeError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("expected plaintext credential endpoint to fail")
    try:
        MODULE.reject_duplicate_endpoint_names(
            [("cloud", "http://one", None), ("cloud", "http://two", None)]
        )
    except ValueError as exc:
        assert "duplicate endpoint" in str(exc)
    else:
        raise AssertionError("expected duplicate endpoint names to fail")
