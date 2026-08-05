import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "deepseek_engine_ab.py"
SPEC = importlib.util.spec_from_file_location("deepseek_engine_ab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_extract_response_collects_text_and_calls():
    answer, calls = MODULE.extract_response(
        {
            "output": [
                {"type": "reasoning", "content": []},
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"rg foo a.py b.py"}',
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        }
    )
    assert answer == "done"
    assert len(calls) == 1


def test_score_bounded_inspection_requires_one_call_with_both_paths():
    case = MODULE.CASES[0]
    call = {
        "name": "exec_command",
        "arguments": json.dumps(
            {
                "cmd": "sed -n '1,200p' scripts/release_check_m3_random.py "
                "tests/test_release_check_random.py"
            }
        ),
    }
    assert MODULE.score(case, "", [call]) == (True, None)
    passed, failure = MODULE.score(case, "", [call, call])
    assert not passed
    assert "one exec_command" in failure


def test_score_review_is_exact():
    case = MODULE.CASES[1]
    assert MODULE.score(case, "NO_BLOCKER", []) == (True, None)
    assert MODULE.score(case, "There is NO_BLOCKER", [])[0] is False


def test_score_recovery_rejects_removal_advice():
    case = MODULE.CASES[2]
    assert MODULE.score(case, "Add g12_eligible to the allowed profile schema.", [])[0]
    assert not MODULE.score(case, "Remove g12_eligible from the schema.", [])[0]


def test_parse_endpoint_uses_environment_variable_name():
    assert MODULE.parse_endpoint("cloud=https://example.test/v1,CLOUD_KEY") == (
        "cloud",
        "https://example.test/v1",
        "CLOUD_KEY",
    )
