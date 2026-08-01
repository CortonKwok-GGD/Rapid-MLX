# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 reasoning parser.

The state transitions mirror the dedicated DeepSeek V4 parsers in vLLM and
SGLang: a bare ``</think>`` is a control token even when thinking is disabled,
and a DSML tool-call opener implicitly closes an unfinished reasoning block.
"""

from __future__ import annotations

from .base import DeltaMessage, ReasoningParser

_THINK_START = "<think>"
_THINK_END = "</think>"
_DSML_TOOL_START = "<｜DSML｜tool_calls>"
_CONTROL_TOKENS = (_THINK_START, _THINK_END, _DSML_TOOL_START)


class DeepSeekV4ReasoningParser(ReasoningParser):
    """Incremental DeepSeek V4 ``<think>``/DSML state machine."""

    # Unlike generic reasoning parsers, this parser must remain active in chat
    # mode to absorb the protocol's bare ``</think>`` control token.
    sanitize_when_thinking_disabled = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset_state()

    def reset_state(self, *, start_in_thinking: bool = False):
        super().reset_state()
        self._in_reasoning = start_in_thinking
        self._buffer = ""

    def configure_request(self, *, enable_thinking: bool | None = None) -> None:
        # DeepSeek V4 emits implicit scratch reasoning without an opening
        # marker.  Treat an unspecified request like the model's default
        # thinking mode; only an explicit False starts in visible content.
        self.reset_state(start_in_thinking=enable_thinking is not False)

    @staticmethod
    def _partial_control_suffix(text: str) -> int:
        best = 0
        for token in _CONTROL_TOKENS:
            for size in range(1, min(len(text), len(token) - 1) + 1):
                if text.endswith(token[:size]):
                    best = max(best, size)
        return best

    def _consume(self, text: str, *, final: bool = False) -> DeltaMessage | None:
        pending = self._buffer + text
        self._buffer = ""
        content: list[str] = []
        reasoning: list[str] = []

        def emit(value: str) -> None:
            if not value:
                return
            (reasoning if self._in_reasoning else content).append(value)

        while pending:
            matches = [
                (idx, token)
                for token in _CONTROL_TOKENS
                if (idx := pending.find(token)) >= 0
            ]
            if not matches:
                held = 0 if final else self._partial_control_suffix(pending)
                emit(pending[:-held] if held else pending)
                if held:
                    self._buffer = pending[-held:]
                break

            index, token = min(matches, key=lambda match: match[0])
            emit(pending[:index])
            pending = pending[index + len(token) :]
            if token == _THINK_START:
                # vLLM absorbs duplicate openers while already reasoning.
                self._in_reasoning = True
            elif token == _THINK_END:
                # vLLM also absorbs a bare closer in CONTENT state.
                self._in_reasoning = False
            else:
                # A DSML tool call is always outside reasoning. Preserve its
                # opener for the downstream DeepSeek V4 tool parser.
                self._in_reasoning = False
                content.append(token)

        normal = "".join(content) or None
        thought = "".join(reasoning) or None
        if normal is None and thought is None:
            return None
        return DeltaMessage(content=normal, reasoning=thought)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        del previous_text, current_text
        return self._consume(delta_text)

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        self.reset_state(start_in_thinking=enable_thinking is not False)
        parsed = self._consume(model_output, final=True)
        if parsed is None:
            return None, None
        return parsed.reasoning, parsed.content

    def finalize_streaming(
        self,
        accumulated_text: str,
        **_kwargs,
    ) -> DeltaMessage | None:
        del accumulated_text
        return self._consume("", final=True)

    def is_open_in_think(self, _accumulated_text: str) -> bool:
        return self._in_reasoning
