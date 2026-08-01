from vllm_mlx.config import ServerConfig
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.reasoning.deepseek_v4_parser import DeepSeekV4ReasoningParser
from vllm_mlx.service.postprocessor import StreamingPostProcessor


def test_chat_mode_absorbs_bare_think_close_after_content() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=False)

    first = parser.extract_reasoning_streaming("", "ruff passed", "ruff passed")
    close = parser.extract_reasoning_streaming(
        "ruff passed", "ruff passed\n</think>", "\n</think>"
    )

    assert first is not None and first.content == "ruff passed"
    assert close is not None and close.content == "\n"
    assert "</think>" not in close.content


def test_split_bare_close_never_leaks_to_content() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=False)

    chunks = ["done<", "/thi", "nk>next"]
    emitted = []
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        result = parser.extract_reasoning_streaming(previous, current, chunk)
        if result and result.content:
            emitted.append(result.content)
        previous = current

    assert "".join(emitted) == "donenext"


def test_thinking_mode_routes_reasoning_then_content() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=True)

    thought = parser.extract_reasoning_streaming("", "inspect", "inspect")
    answer = parser.extract_reasoning_streaming(
        "inspect", "inspect</think>fixed", "</think>fixed"
    )

    assert thought is not None and thought.reasoning == "inspect"
    assert answer is not None and answer.content == "fixed"
    assert answer.reasoning is None


def test_dsml_tool_start_implicitly_closes_reasoning() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=True)

    parsed = parser.extract_reasoning_streaming(
        "",
        "plan<｜DSML｜tool_calls><｜DSML｜invoke",
        "plan<｜DSML｜tool_calls><｜DSML｜invoke",
    )

    assert parsed is not None
    assert parsed.reasoning == "plan"
    assert parsed.content == "<｜DSML｜tool_calls><｜DSML｜invoke"


def test_nonstream_chat_mode_absorbs_control_tokens() -> None:
    parser = DeepSeekV4ReasoningParser()
    reasoning, content = parser.extract_reasoning(
        "first</think>second", enable_thinking=False
    )

    assert reasoning is None
    assert content == "firstsecond"


def test_chat_postprocessor_keeps_sanitizer_active_when_thinking_disabled() -> None:
    cfg = ServerConfig()
    cfg.reasoning_parser_name = "deepseek_v4"
    processor = StreamingPostProcessor(cfg, enable_thinking=False)
    processor.reset()

    visible = []
    for text in ("ruff passed\n<", "/think>", "next"):
        events = processor.process_chunk(
            GenerationOutput(text=text, new_text=text, tokens=[1], finished=False)
        )
        visible.extend(event.content for event in events if event.type == "content")

    assert "".join(visible) == "ruff passed\nnext"
    assert "</think>" not in "".join(visible)
