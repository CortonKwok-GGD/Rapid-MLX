#!/usr/bin/env python3
"""Compare DeepSeek-compatible Responses endpoints on engineering tasks.

The suite deliberately uses small, independently scoreable cases before an
expensive Codex soak.  Endpoint credentials are read from environment
variables and are never included in the JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

EXEC_TOOL = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a shell command in the repository.",
    "parameters": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected: str
    tools: tuple[dict[str, Any], ...] = ()


CASES = (
    Case(
        name="bounded_inspection",
        prompt=(
            "Inspect scripts/release_check_m3_random.py and "
            "tests/test_release_check_random.py. Make exactly one "
            "exec_command tool call that reads both files; do not edit files."
        ),
        expected="one_exec_call_reads_both",
        tools=(EXEC_TOOL,),
    ),
    Case(
        name="evidence_bounded_review",
        prompt=(
            "You are reviewing a narrowly scoped change. Facts: G12 is only "
            "an agent-quality sampling gate. Parser and wire-format tests "
            "remain separate and unchanged for every model. Existing profiles "
            "without g12_eligible must remain eligible for backward "
            "compatibility; only an explicit false opts out. The patch "
            "implements exactly that. Is there a material correctness blocker? "
            "Reply exactly NO_BLOCKER if none. Do not reconsider your answer."
        ),
        expected="exact_no_blocker",
    ),
    Case(
        name="schema_recovery",
        prompt=(
            "AliasProfile is a deprecated alias of ModelProfile. A regression "
            "says: g12_eligible is rejected because it is absent from the "
            "allowed profile-key schema. State the next code change in at most "
            "25 words. Do not propose removing g12_eligible."
        ),
        expected="add_schema_key",
    ),
)


@dataclass
class Trial:
    endpoint: str
    case: str
    repetition: int
    passed: bool
    status: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    tool_calls: int
    answer: str
    failure: str | None = None


def extract_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text: list[str] = []
    calls: list[dict[str, Any]] = []
    for item in data.get("output", []):
        if item.get("type") == "function_call":
            calls.append(item)
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                text.append(content["text"])
    return "\n".join(text).strip(), calls


def score(
    case: Case, answer: str, calls: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    if case.expected == "exact_no_blocker":
        return (
            answer == "NO_BLOCKER",
            None if answer == "NO_BLOCKER" else "not exact NO_BLOCKER",
        )
    if case.expected == "add_schema_key":
        lowered = answer.lower()
        passed = (
            "g12_eligible" in lowered
            and "schema" in lowered
            and any(word in lowered for word in ("add", "allow", "include"))
            and not any(
                phrase in lowered
                for phrase in ("remove `g12_eligible`", "remove g12_eligible")
            )
        )
        return (
            passed,
            None if passed else "did not add g12_eligible to the allowed schema",
        )
    if case.expected == "one_exec_call_reads_both":
        if len(calls) != 1 or calls[0].get("name") != "exec_command":
            return False, f"expected one exec_command, got {len(calls)} tool calls"
        try:
            command = json.loads(calls[0].get("arguments") or "{}").get("cmd", "")
        except json.JSONDecodeError:
            return False, "invalid tool arguments JSON"
        passed = all(
            path in command
            for path in (
                "scripts/release_check_m3_random.py",
                "tests/test_release_check_random.py",
            )
        )
        return (
            passed,
            None if passed else "tool command did not read both requested files",
        )
    raise ValueError(f"unknown expectation: {case.expected}")


def run_trial(
    client: httpx.Client,
    endpoint_name: str,
    url: str,
    model: str,
    case: Case,
    repetition: int,
) -> Trial:
    payload: dict[str, Any] = {
        "model": model,
        "input": case.prompt,
        "max_output_tokens": 1024,
        "reasoning": {"effort": "medium"},
    }
    if case.tools:
        payload["tools"] = list(case.tools)
    started = time.perf_counter()
    response = client.post(f"{url.rstrip('/')}/responses", json=payload)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    answer, calls = extract_response(data)
    passed, failure = score(case, answer, calls)
    usage = data.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    return Trial(
        endpoint=endpoint_name,
        case=case.name,
        repetition=repetition,
        passed=passed and data.get("status") == "completed",
        status=str(data.get("status")),
        latency_seconds=round(elapsed, 3),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        tool_calls=len(calls),
        answer=answer,
        failure=failure,
    )


def parse_endpoint(value: str) -> tuple[str, str, str | None]:
    """Parse NAME=URL[,ENV_VAR] without ever accepting a literal secret."""
    try:
        name, target = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("endpoint must be NAME=URL[,ENV_VAR]") from exc
    url, separator, env_var = target.partition(",")
    if not name or not url:
        raise argparse.ArgumentTypeError("endpoint name and URL are required")
    return name, url, env_var if separator else None


def summarize(trials: list[Trial]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for endpoint in sorted({trial.endpoint for trial in trials}):
        selected = [trial for trial in trials if trial.endpoint == endpoint]
        summary[endpoint] = {
            "passed": sum(trial.passed for trial in selected),
            "trials": len(selected),
            "pass_rate": sum(trial.passed for trial in selected) / len(selected),
            "median_latency_seconds": statistics.median(
                trial.latency_seconds for trial in selected
            ),
            "median_output_tokens": statistics.median(
                trial.output_tokens for trial in selected
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        action="append",
        required=True,
        type=parse_endpoint,
        metavar="NAME=URL[,ENV_VAR]",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    trials: list[Trial] = []
    for name, url, env_var in args.endpoint:
        headers = {}
        if env_var:
            token = os.environ.get(env_var)
            if not token:
                parser.error(f"environment variable {env_var!r} is not set")
            headers["Authorization"] = f"Bearer {token}"
        with httpx.Client(headers=headers, timeout=args.timeout) as client:
            for case in CASES:
                for repetition in range(1, args.repetitions + 1):
                    trial = run_trial(client, name, url, args.model, case, repetition)
                    trials.append(trial)
                    print(json.dumps(asdict(trial), ensure_ascii=False), flush=True)

    report = {
        "trials": [asdict(trial) for trial in trials],
        "summary": summarize(trials),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=False))
    return 0 if all(trial.passed for trial in trials) else 1


if __name__ == "__main__":
    raise SystemExit(main())
