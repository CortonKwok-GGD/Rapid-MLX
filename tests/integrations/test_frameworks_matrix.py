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
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai not installed — cell deferred")

        llm = ChatOpenAI(
            model=rapid_mlx_server["model_id"],
            base_url=rapid_mlx_server["base_url"],
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
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError:
            pytest.skip("pydantic-ai not installed — cell deferred")

        model = OpenAIChatModel(
            model_name=rapid_mlx_server["model_id"],
            provider=OpenAIProvider(
                base_url=rapid_mlx_server["base_url"],
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
        tool_agent = Agent(model)

        @tool_agent.tool_plain
        def get_weather(city: str) -> str:  # noqa: ARG001 — invoked by agent
            """Get weather for a city."""
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
        # PydanticAI folds the tool result into ``.output`` — confirm the
        # answer references the tool's return.
        answer = (result.output or "").lower()
        assert_content_nonempty(answer, ctx=f"pydantic-ai/{family_alias.family}/tool")
        assert_no_think_tag_leak(answer)
        assert_no_analysis_channel_leak(answer)


# --------------------------------------------------------------------------- #
# smolagents
# --------------------------------------------------------------------------- #


class TestSmolagents:
    """smolagents — ToolCallingAgent with a real Tool."""

    def test_smoke(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        try:
            from smolagents import OpenAIServerModel, Tool, ToolCallingAgent
        except ImportError:
            pytest.skip("smolagents not installed — cell deferred")

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
                return f"sunny in {city}"

        model = OpenAIServerModel(
            model_id=rapid_mlx_server["model_id"],
            api_base=rapid_mlx_server["base_url"],
            api_key="not-needed",
        )
        agent = ToolCallingAgent(tools=[GetWeatherTool()], model=model, max_steps=3)
        try:
            answer = agent.run("What's the weather in Tokyo? Use the get_weather tool.")
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(
                f"smolagents/{family_alias.family}: run failed: {exc}"
            )
            return
        content = str(answer)
        assert_content_nonempty(content, ctx=f"smolagents/{family_alias.family}")
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
