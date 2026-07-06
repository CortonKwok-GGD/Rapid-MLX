# SPDX-License-Identifier: Apache-2.0
"""Tier-1 frameworks × 4 families integration matrix (0.10.2 PR-2).

Three Tier-1 frameworks × 4 Tier-1 families = 12 real cells. Each
framework drives the running rapid-mlx server through its own SDK, so
a regression in the OpenAI-compat chat completions path — or the
tool-binding path — surfaces via the framework the operator would
actually use in prod.

Frameworks:

* LangChain (+ LangGraph — same profile) via ``langchain-openai``.
* PydanticAI via ``pydantic_ai.models.openai.OpenAIChatModel``.
* smolagents via ``OpenAIServerModel`` + ``ToolCallingAgent``.

Deep flows live in the dedicated files (``test_langchain.py``,
``test_pydantic_ai_full.py``, ``test_smolagents_full.py``); the matrix
cell here proves the framework plumbs onto each Tier-1 family without
having to re-run the deep file per family.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.integrations.conftest import (
    DEFAULT_TIMEOUT_S,
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    assert_tool_call_shape,
    strict_skip_or_fail,
)

# --------------------------------------------------------------------------- #
# LangChain (+ LangGraph)
# --------------------------------------------------------------------------- #


class TestLangChain:
    """LangChain / LangGraph — plain invoke + bind_tools tool call."""

    def test_smoke(
        self,
        rapid_mlx_matrix_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai not installed — cell deferred")

        llm = ChatOpenAI(
            model=rapid_mlx_matrix_server["model_id"],
            base_url=rapid_mlx_matrix_server["base_url"],
            api_key="not-needed",
            temperature=0.0,
            max_tokens=256,
            timeout=DEFAULT_TIMEOUT_S,
        )

        # Plain invoke — confirm the model answers over the wire.
        try:
            r = llm.invoke([HumanMessage(content="Reply with just OK.")])
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(
                f"langchain/{family_alias.family}: plain invoke failed: {exc}"
            )
            return
        content = getattr(r, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        assert_content_nonempty(content, ctx=f"langchain/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)

        # Tool call — confirm bind_tools path plumbs onto rapid-mlx.
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"sunny in {city}"

        llm_with_tools = llm.bind_tools([get_weather])
        try:
            r = llm_with_tools.invoke(
                [HumanMessage(content="What's the weather in Tokyo? Use the tool.")]
            )
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(
                f"langchain/{family_alias.family}: tool invoke failed: {exc}"
            )
            return
        tool_calls = getattr(r, "tool_calls", None) or []
        if not tool_calls:
            strict_skip_or_fail(
                f"langchain/{family_alias.family}: bind_tools returned no "
                f"tool_calls — wire regression on the LangChain tool route."
            )
            return
        tc = tool_calls[0]
        tc_dict = {
            "id": tc.get("id") or "call_lc_smoke",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            },
        }
        assert_tool_call_shape(tc_dict)
        assert tc["name"] == "get_weather", tc
        assert "city" in tc["args"], tc
        assert "tokyo" in tc["args"]["city"].lower(), tc


# --------------------------------------------------------------------------- #
# PydanticAI
# --------------------------------------------------------------------------- #


class TestPydanticAI:
    """PydanticAI — plain run + tool call via ``@agent.tool_plain``."""

    def test_smoke(
        self,
        rapid_mlx_matrix_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError:
            pytest.skip("pydantic-ai not installed — cell deferred")

        model = OpenAIChatModel(
            model_name=rapid_mlx_matrix_server["model_id"],
            provider=OpenAIProvider(
                base_url=rapid_mlx_matrix_server["base_url"],
                api_key="not-needed",
            ),
        )
        agent = Agent(model)

        try:
            result = agent.run_sync("Reply with just OK.")
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(
                f"pydantic-ai/{family_alias.family}: run_sync failed: {exc}"
            )
            return
        content = (result.output or "").strip()
        assert_content_nonempty(content, ctx=f"pydantic-ai/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)

        # Tool call — pydantic_ai routes tool calls via @agent.tool.
        # Codex #1033 round-2 BLOCKING #1: assert the tool was actually
        # invoked (not just that the model returned non-empty text). A
        # flag closure captures whether ``get_weather`` ran; strict mode
        # fails if the tool never fired even though the prompt asked for
        # it. Also check the answer references the tool's return
        # (``sunny`` in ``tokyo``) so a broken tool loop where the model
        # ignores the tool result but improvises can't slip through.
        tool_agent = Agent(model)
        tool_invocations: dict[str, int] = {"count": 0}

        @tool_agent.tool_plain
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            tool_invocations["count"] += 1
            return f"sunny in {city}"

        try:
            result = tool_agent.run_sync(
                "What's the weather in Tokyo? Use the get_weather tool."
            )
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(
                f"pydantic-ai/{family_alias.family}: tool run failed: {exc}"
            )
            return
        answer = (result.output or "").lower()
        assert_content_nonempty(answer, ctx=f"pydantic-ai/{family_alias.family}/tool")
        assert_no_think_tag_leak(answer)
        assert_no_analysis_channel_leak(answer)
        if tool_invocations["count"] == 0:
            # The whole point of this cell is the tool-loop regression
            # gate — a model that answers inline without calling the
            # tool is a real signal in strict CI.
            strict_skip_or_fail(
                f"pydantic-ai/{family_alias.family}: get_weather was never "
                f"invoked despite the tool prompt. Answer was: {answer[:120]!r}"
            )
            return
        # Codex #1033 round-4 BLOCKING #3: require BOTH the tool-return
        # marker (``sunny``) AND the requested city (``tokyo``). Using
        # OR let an answer that merely mentioned Tokyo pass even though
        # it ignored the tool's ``sunny in <city>`` payload — a broken
        # tool-loop failure mode this cell is exactly here to catch.
        assert "sunny" in answer and "tokyo" in answer, (
            f"pydantic-ai/{family_alias.family}: tool was called "
            f"({tool_invocations['count']}x) but answer doesn't reference the "
            f"tool's return (need both 'sunny' AND 'tokyo'): {answer[:200]!r}"
        )


# --------------------------------------------------------------------------- #
# smolagents
# --------------------------------------------------------------------------- #


class TestSmolagents:
    """smolagents — ToolCallingAgent with a real Tool.

    Codex #1033 round-4 BLOCKING #4: assert ``GetWeatherTool.forward``
    actually ran (invocation counter on the tool instance) and that
    the final answer references the tool's return (``sunny`` in
    ``tokyo``). Without both, an inline model reply with broken tool
    routing would still pass — the whole point of a framework tool
    cell is the tool-loop regression gate.
    """

    def test_smoke(
        self,
        rapid_mlx_matrix_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from smolagents import OpenAIServerModel, Tool, ToolCallingAgent
        except ImportError:
            pytest.skip("smolagents not installed — cell deferred")

        # Invocation counter kept on the outer scope so the smolagents
        # ``Tool`` subclass (which the SDK wraps) can bump it inside
        # forward() without needing a mutable class attribute.
        invocations: dict[str, int] = {"count": 0}

        class GetWeatherTool(Tool):
            name = "get_weather"
            description = "Get the weather for a city."
            inputs = {
                "city": {
                    "type": "string",
                    "description": "City name.",
                }
            }
            output_type = "string"

            def forward(self, city: str) -> str:  # type: ignore[override]
                invocations["count"] += 1
                return f"sunny in {city}"

        model = OpenAIServerModel(
            model_id=rapid_mlx_matrix_server["model_id"],
            api_base=rapid_mlx_matrix_server["base_url"],
            api_key="not-needed",
        )
        agent = ToolCallingAgent(tools=[GetWeatherTool()], model=model, max_steps=3)
        try:
            answer = agent.run("What's the weather in Tokyo? Use the get_weather tool.")
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(f"smolagents/{family_alias.family}: run failed: {exc}")
            return
        content = str(answer).lower()
        assert_content_nonempty(content, ctx=f"smolagents/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
        if invocations["count"] == 0:
            strict_skip_or_fail(
                f"smolagents/{family_alias.family}: get_weather.forward was "
                f"never called despite the tool prompt. Answer was: "
                f"{content[:120]!r}"
            )
            return
        # Answer must reference the tool's return — both markers so an
        # answer that merely echoes ``tokyo`` without touching the tool
        # result can't slip through.
        assert "sunny" in content and "tokyo" in content, (
            f"smolagents/{family_alias.family}: tool was called "
            f"({invocations['count']}x) but final answer doesn't reference "
            f"the tool's return (need both 'sunny' AND 'tokyo'): "
            f"{content[:200]!r}"
        )
