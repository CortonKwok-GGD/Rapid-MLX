# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for MiniCPM5's documented native XML format."""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from typing import Any

from ..api.tool_calling import _serialize_tool_arguments
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

_OPEN = "<function"
_CLOSE = "</function>"
_CDATA_OPEN = "<![CDATA["
_CDATA_CLOSE = "]]>"


def _tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module("minicpm")
class MiniCPMToolParser(ToolParser):
    """Parse ``<function name=...><param name=...>...</param></function>``."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("minicpm_native",)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._streamed_call_count = 0

    @staticmethod
    def _is_function_opener(text: str, start: int) -> bool:
        """Avoid treating ordinary words such as ``<functionality>`` as XML."""
        end = start + len(_OPEN)
        return end == len(text) or text[end].isspace() or text[end] in ">/"

    @staticmethod
    def _function_end(text: str, start: int) -> int | None:
        """Find the native close tag without mistaking CDATA data for markup."""
        cursor = start + len(_OPEN)
        opener_end = text.find(">", cursor)
        if opener_end != -1 and text[opener_end - 1] == "/":
            return opener_end + 1
        while True:
            close = text.find(_CLOSE, cursor)
            if close == -1:
                return None
            cdata = text.find(_CDATA_OPEN, cursor)
            if cdata == -1 or close < cdata:
                return close + len(_CLOSE)
            cdata_end = text.find(_CDATA_CLOSE, cdata + len(_CDATA_OPEN))
            if cdata_end == -1:
                return None
            cursor = cdata_end + len(_CDATA_CLOSE)

    @classmethod
    def _blocks(cls, text: str) -> Iterator[tuple[int, int, ET.Element]]:
        """Yield complete, well-formed native call elements in wire order."""
        cursor = 0
        while (start := text.find(_OPEN, cursor)) != -1:
            if not cls._is_function_opener(text, start):
                cursor = start + len(_OPEN)
                continue

            end = cls._function_end(text, start)
            if end is None:
                return
            try:
                element = ET.fromstring(text[start:end])
            except ET.ParseError:
                cursor = end
                continue
            yield start, end, element
            cursor = end

    @staticmethod
    def _call_from_element(
        element: ET.Element, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Validate one documented element and normalize it to OpenAI JSON."""
        name = element.get("name")
        if (
            element.tag != "function"
            or set(element.attrib) != {"name"}
            or not isinstance(name, str)
            or not name.strip()
            or (element.text or "").strip()
        ):
            return None

        arguments: dict[str, str] = {}
        for param in element:
            param_name = param.get("name")
            if (
                param.tag != "param"
                or set(param.attrib) != {"name"}
                or not isinstance(param_name, str)
                or not param_name.strip()
                or param_name in arguments
                or len(param)
                or (param.tail or "").strip()
            ):
                return None
            arguments[param_name] = param.text or ""

        return {
            "id": _tool_id(),
            "name": name.strip(),
            "arguments": _serialize_tool_arguments(arguments, name.strip(), request),
        }

    @classmethod
    def _parsed(
        cls, text: str, request: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], str]:
        calls: list[dict[str, Any]] = []
        spans: list[tuple[int, int]] = []
        for start, end, element in cls._blocks(text):
            call = cls._call_from_element(element, request)
            if call is not None:
                calls.append(call)
                spans.append((start, end))

        if not spans:
            return calls, text
        content: list[str] = []
        cursor = 0
        for start, end in spans:
            content.append(text[cursor:start])
            cursor = end
        content.append(text[cursor:])
        return calls, "".join(content)

    @classmethod
    def _incomplete_start(cls, text: str) -> int | None:
        """Return a partial native-call opener that must stay out of SSE text."""
        for length in range(min(len(_OPEN) - 1, len(text)), 0, -1):
            if text.endswith(_OPEN[:length]):
                return len(text) - length
        start = text.rfind(_OPEN)
        if start != -1 and cls._is_function_opener(text, start):
            return start if cls._function_end(text, start) is None else None
        return None

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Return the part of an SSE prefix known not to be native markup."""
        start = cls._incomplete_start(text)
        return text if start is None else text[:start]

    def reset(self) -> None:
        super().reset()
        self._streamed_call_count = 0

    def has_pending_tool_call(self, text: str) -> bool:
        return self._incomplete_start(text) is not None

    def flush_held_content(self, full_text: str) -> str:
        start = self._incomplete_start(full_text)
        return full_text[start:] if start is not None else ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        calls, content = self._parsed(model_output, request)
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=(content or None) if calls else model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not previous_text or not current_text.startswith(previous_text):
            self.reset()

        calls, current_content = self._parsed(current_text, request)
        _, previous_content = self._parsed(previous_text, request)
        safe_current = self._safe_content_prefix(current_content)
        safe_previous = self._safe_content_prefix(previous_content)
        new_calls = calls[self._streamed_call_count :]
        if new_calls:
            self._streamed_call_count = len(calls)
            event: dict[str, Any] = {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for index, call in enumerate(
                        new_calls, self._streamed_call_count - len(new_calls)
                    )
                ]
            }
            if safe_current.startswith(safe_previous):
                content = safe_current[len(safe_previous) :]
            else:
                content = safe_current
            if content:
                event["content"] = content
            return event

        if safe_current.startswith(safe_previous):
            delta = safe_current[len(safe_previous) :]
        else:
            delta = safe_current
        return {"content": delta} if delta else None
