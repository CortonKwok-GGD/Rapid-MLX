# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the 0.10.2 agent + framework integration matrices.

Two matrices share this harness:

* ``test_agents_matrix.py`` — 8 Tier-1 agents × 4 families (Qwen 3.6,
  Gemma 4, DeepSeek V4 Flash, gpt-oss 120B) = 32 cells.
* ``test_frameworks_matrix.py`` — 3 Tier-1 frameworks × 4 families = 12 cells.

Both matrices reuse the same server fixture (which auto-boots
``rapid-mlx serve <alias>`` on port 8802 for the family currently under
test), the same tool schemas, and the same channel-leak assertions.

Strong-pick policy (raullen sign-off 2026-07-06, 0.10.2 PR-2)
------------------------------------------------------------

Each family's alias is the **strong pick** — the largest available
public MLX quant that fits a 512 GB M3 Ultra and still leaves headroom
for operator services on ports 8801 / 8772:

* Qwen 3.6: ``qwen3.6-35b-8bit`` — 35 GB, Qwen3.6-35B-A3B-8bit MoE.
* Gemma 4: ``gemma-4-31b-4bit`` — 18 GB, gemma-4-31b-it-4bit.
* DeepSeek V4 Flash: ``deepseek-v4-flash-8bit`` — 50 GB.
* gpt-oss 120B: ``gpt-oss-120b-mxfp4-q8`` — 65 GB (Harmony wire).

Small variants (4B / 12B) fail tool-calling for reasons unrelated to
the wire path (model 降智 — capability ceiling of the quant/size, not
a rapid-mlx bug). Testing against strong picks isolates the integration
signal from model-capability noise. The trade-off is longer boot time
and per-cell wall-clock (~2-5 min boot, ~60 s per cell) — acceptable
because this matrix runs per-integration-PR, not per-commit.

Environment overrides
---------------------

* ``RAPID_MLX_BASE_URL`` — point at an already-running server instead of
  auto-booting one. When set, the ``rapid_mlx_server`` fixture skips the
  boot dance and just probes /v1/models. Handy for local dev when a
  large model is already loaded and re-loading would waste 3-5 min.
* ``RAPID_MLX_AGENT_MATRIX_FAMILY`` — restrict matrix to one family
  (``qwen36`` / ``gemma4`` / ``deepseek`` / ``gptoss``). Handy for CI
  shards. If ``RAPID_MLX_BASE_URL`` is also set, this must match the
  family the running server is serving (guard checks /v1/models).
* ``RAPID_MLX_MATRIX_STRICT`` — if ``1``, missing-server / server-error /
  degraded-cell → fail instead of skip. Off by default so a naive
  ``pytest`` run stays green without hardware.
* ``RAPID_MLX_MATRIX_PORT`` — port to boot on (default 8802). NEVER
  overlap operator services on 8801 (qwen3-vl) or 8772 (holo3) — G1.
* ``RAPID_MLX_SERVE_BIN`` — how to invoke rapid-mlx (default:
  ``python3.12 -m vllm_mlx.cli``). Set to ``rapid-mlx`` to use the
  installed wrapper when the wrapper points at the intended venv.

Post-#1030 codex-review fold
----------------------------

Prior scaffold made every matrix cell skip unless the operator hand-
booted a server. Codex #1030 flagged that as regression-hiding — a
green run with 44/44 skipped is indistinguishable from a green run
with 44/44 passing. The 0.10.2 PR-2 rewrite fixes this by (a) auto-
booting a server per family and (b) elevating ``RAPID_MLX_MATRIX_STRICT``
to the CI enforcement lever.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


DEFAULT_MATRIX_PORT = 8802
FORBIDDEN_PORTS = frozenset({8801, 8772})  # G1: operator services
SERVER_BOOT_TIMEOUT_S = 600  # 65B gpt-oss cold boot fits in ~5 min on M3 Ultra
SERVER_READY_POLL_S = 3.0
DEFAULT_TIMEOUT_S = 180  # per-cell HTTP client timeout — 35B/120B decodes are slow


@dataclass(frozen=True)
class FamilyAlias:
    """A strong-per-family alias used across the matrices."""

    family: str  # matrix column key: qwen36 / gemma4 / deepseek / gptoss
    alias: str  # rapid-mlx alias string (positional model arg)
    hf_path: str  # HuggingFace repo id (for cache probing)
    tool_call_parser: str  # documented parser (for skip-inference)
    reasoning_parser: str  # documented reasoning parser
    reason: str  # why this strong pick


# Strong per-family aliases (raullen 2026-07-06 sign-off). Keep in sync
# with the ``hf_path`` fields declared in ``vllm_mlx/aliases.json``.
_FAMILY_ALIASES: dict[str, FamilyAlias] = {
    "qwen36": FamilyAlias(
        family="qwen36",
        alias="qwen3.6-35b-8bit",
        hf_path="mlx-community/Qwen3.6-35B-A3B-8bit",
        tool_call_parser="qwen3_coder_xml",
        reasoning_parser="qwen3",
        reason="Qwen 3.6 35B MoE strong pick (raullen sign-off 2026-07-06)",
    ),
    "gemma4": FamilyAlias(
        family="gemma4",
        alias="gemma-4-31b-4bit",
        hf_path="mlx-community/gemma-4-31b-it-4bit",
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        reason="Gemma 4 31B strong pick (12B fails tool-calling — model 降智)",
    ),
    "deepseek": FamilyAlias(
        family="deepseek",
        alias="deepseek-v4-flash-8bit",
        hf_path="mlx-community/DeepSeek-V4-Flash-8bit",
        tool_call_parser="deepseek",
        reasoning_parser="deepseek_r1",
        reason="DeepSeek V4 Flash 8bit — only Tier-1 DeepSeek MLX quant on-shelf",
    ),
    "gptoss": FamilyAlias(
        family="gptoss",
        alias="gpt-oss-120b-mxfp4-q8",
        hf_path="mlx-community/gpt-oss-120b-MXFP4-Q8",
        tool_call_parser="harmony",
        reasoning_parser="harmony",
        reason="gpt-oss 120B Harmony strong pick (20B skips reasoning channel)",
    ),
}


def _families_in_scope() -> tuple[str, ...]:
    """Return the families to parametrize over.

    Honours ``RAPID_MLX_AGENT_MATRIX_FAMILY`` for CI sharding — set that
    env to one family key to restrict the run to a single column.
    """
    only = os.environ.get("RAPID_MLX_AGENT_MATRIX_FAMILY", "").strip()
    if only:
        if only not in _FAMILY_ALIASES:
            raise ValueError(
                f"RAPID_MLX_AGENT_MATRIX_FAMILY={only!r} unknown; "
                f"valid: {sorted(_FAMILY_ALIASES)}"
            )
        return (only,)
    return tuple(_FAMILY_ALIASES.keys())


def _strict() -> bool:
    return os.environ.get("RAPID_MLX_MATRIX_STRICT", "").strip() == "1"


def matrix_strict_mode() -> bool:
    """Public accessor for ``RAPID_MLX_MATRIX_STRICT``."""
    return _strict()


def strict_skip_or_fail(reason: str) -> None:
    """Skip in non-strict mode; fail in strict mode.

    Consolidates the "cell degraded, not red" pattern so a broken
    server-side route or a regressed SDK doesn't quietly hide behind a
    green skipped cell when the operator asked for enforcement via
    ``RAPID_MLX_MATRIX_STRICT=1``. Codex #1030 flagged the earlier all-skip
    pattern as regression-hiding.
    """
    if _strict():
        pytest.fail(reason)
    pytest.skip(reason)


# --------------------------------------------------------------------------- #
# Server auto-boot
# --------------------------------------------------------------------------- #


def _matrix_port() -> int:
    raw = os.environ.get("RAPID_MLX_MATRIX_PORT", str(DEFAULT_MATRIX_PORT))
    # Codex #1033 round-1 NIT #1: catch malformed port so a typo like
    # ``RAPID_MLX_MATRIX_PORT=eight-thousand`` yields the harness's own
    # skip/fail message instead of a raw ``ValueError`` traceback.
    try:
        port = int(raw)
    except ValueError:
        pytest.exit(
            f"RAPID_MLX_MATRIX_PORT={raw!r} is not an integer. "
            f"Pick a numeric port (e.g. 8802)."
        )
    # G1: never overlap operator services.
    if port in FORBIDDEN_PORTS:
        pytest.exit(
            f"RAPID_MLX_MATRIX_PORT={port} collides with operator service "
            f"(forbidden: {sorted(FORBIDDEN_PORTS)}). Pick another port."
        )
    return port


def _serve_command(alias: str, port: int) -> list[str]:
    """Return argv for ``rapid-mlx serve <alias> --port <port>``.

    Defaults to ``python3.12 -m vllm_mlx.cli`` (worktree-safe: editable
    install without wrapper indirection). Override via ``RAPID_MLX_SERVE_BIN``
    when the ``rapid-mlx`` wrapper points at the correct venv.

    Codex #1033 round-1 NIT #2: use ``shlex.split`` so a quoted path like
    ``RAPID_MLX_SERVE_BIN='/opt/homebrew/opt/python@3.12/bin/python3.12 -m vllm_mlx.cli'``
    or an argv element containing a space parses correctly. Plain
    ``str.split`` would tokenise on the space inside a quoted path.
    """
    bin_override = os.environ.get("RAPID_MLX_SERVE_BIN", "").strip()
    if bin_override:
        argv = shlex.split(bin_override)
    else:
        argv = [sys.executable, "-m", "vllm_mlx.cli"]
    argv += ["serve", alias, "--port", str(port)]
    return argv


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _wait_for_ready(port: int, timeout_s: int) -> bool:
    """Poll /v1/models (200) until ready or timeout."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(SERVER_READY_POLL_S)
    return False


@dataclass
class _ServerHandle:
    proc: subprocess.Popen | None
    port: int
    base_url: str
    model_id: str
    log_path: Path | None


def _boot_server(alias: FamilyAlias, port: int) -> _ServerHandle:
    """Boot ``rapid-mlx serve <alias>`` on ``port`` and return handle.

    Uses a log file under /tmp so operators can tail progress. Skips
    (or fails, in strict mode) on boot failure — the matrix cell is
    the audience, not the pytest harness.

    Codex #1033 round-1 BLOCKING #2: the parent's ``log_f`` file
    descriptor is only used to hand stdout/stderr to ``Popen``. Once
    the child has inherited the fd, close the parent-side handle so
    every successful family boot doesn't leak an fd. The child keeps
    writing through its own inherited fd until it exits and shutdown
    cleans up.
    """
    if _port_in_use(port):
        strict_skip_or_fail(
            f"matrix port {port} already in use — refusing to clobber "
            f"(check `lsof -i :{port}`)"
        )

    log_path = Path("/tmp") / f"rapid-mlx-matrix-{alias.family}-{port}.log"
    cmd = _serve_command(alias.alias, port)
    proc = None
    with open(log_path, "w") as log_f:
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except FileNotFoundError as exc:
            strict_skip_or_fail(
                f"could not exec {cmd!r}: {exc}. Set RAPID_MLX_SERVE_BIN or "
                f"ensure `{sys.executable} -m vllm_mlx.cli` is importable."
            )
            return _ServerHandle(None, port, "", "", None)  # unreachable in strict
    # ``with`` block closed ``log_f`` — the child now owns its own dup'd
    # fd on the log file, so parent-side fd is safely released.

    if not _wait_for_ready(port, SERVER_BOOT_TIMEOUT_S):
        # Best-effort teardown.
        try:
            proc.send_signal(2)
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        strict_skip_or_fail(
            f"{alias.alias} did not become ready on port {port} within "
            f"{SERVER_BOOT_TIMEOUT_S}s — see {log_path}"
        )
        return _ServerHandle(None, port, "", "", None)  # unreachable in strict

    # Probe once more to grab the model_id the server actually reported.
    import urllib.request

    try:
        with urllib.request.urlopen(  # noqa: S310
            f"http://127.0.0.1:{port}/v1/models", timeout=3
        ) as resp:
            model_id = (json.loads(resp.read()).get("data") or [{}])[0].get("id", "")
    except Exception:  # noqa: BLE001
        model_id = alias.alias

    return _ServerHandle(
        proc=proc,
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1",
        model_id=model_id,
        log_path=log_path,
    )


def _shutdown_server(handle: _ServerHandle) -> None:
    """Best-effort SIGINT → SIGKILL teardown."""
    proc = handle.proc
    if proc is None:
        return
    try:
        proc.send_signal(2)  # SIGINT — triggers FastAPI lifespan
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Fixtures — the matrix booter
# --------------------------------------------------------------------------- #


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Auto-parametrize any test that requests a ``family_alias`` argument."""
    if "family_alias" in metafunc.fixturenames:
        families = _families_in_scope()
        aliases = [_FAMILY_ALIASES[f] for f in families]
        metafunc.parametrize(
            "family_alias",
            aliases,
            ids=[a.family for a in aliases],
            indirect=False,
            scope="session",
        )


_MODEL_ID_TO_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "qwen36": ("qwen3.6",),
    "gemma4": ("gemma-4", "gemma4"),
    "deepseek": ("deepseek-v4", "deepseek-r1", "deepseek"),
    "gptoss": ("gpt-oss", "openai/gpt-oss"),
}


def _infer_family_from_model_id(model_id: str) -> str | None:
    """Best-effort mapping from the server's reported ``/v1/models`` id
    to a matrix family key. Returns ``None`` if no prefix matches so a
    stranger model can't silently satisfy any family cell.
    """
    if not model_id:
        return None
    lower = model_id.lower()
    for family, prefixes in _MODEL_ID_TO_FAMILY_PREFIXES.items():
        for prefix in prefixes:
            if prefix in lower:
                return family
    return None


@pytest.fixture(scope="session")
def rapid_mlx_server(family_alias: FamilyAlias) -> Iterator[dict[str, Any]]:
    """Yield metadata for a rapid-mlx server serving ``family_alias``.

    Two modes:

    * **External** (env ``RAPID_MLX_BASE_URL`` set): probe /v1/models,
      assert the reported model_id maps to ``family_alias.family``, and
      yield without touching any subprocess. Local-dev shortcut so a
      large model already loaded elsewhere doesn't get re-booted. If
      the external server serves a different family, the cell skips
      (or fails in strict mode) — matching the guard the auto-boot
      path enforces by definition.
    * **Auto-boot** (default): boot ``rapid-mlx serve <alias>`` on
      ``RAPID_MLX_MATRIX_PORT`` (default 8802), yield, teardown at
      session end.

    Session-scoped keyed on ``family_alias`` — a new server boots for
    each parametrized family (Qwen 3.6 → shutdown → Gemma 4 → shutdown
    → ...). Sequential is intentional: two 30-65 GB models in memory
    would OOM even the 512 GB M3 Ultra with operator services running.
    """
    port = _matrix_port()

    external = os.environ.get("RAPID_MLX_BASE_URL", "").strip().rstrip("/")
    if external:
        import urllib.request

        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{external}/models", timeout=3
            ) as resp:
                data = json.loads(resp.read()).get("data") or []
            model_id = data[0]["id"] if data else ""
        except Exception as exc:  # noqa: BLE001
            strict_skip_or_fail(
                f"external RAPID_MLX_BASE_URL={external} unreachable: {exc!r}"
            )
            return
        # Codex #1033 round-1 BLOCKING #1: verify the server's model_id
        # maps to the parametrized family — otherwise a single Qwen server
        # would silently "cover" every Gemma/DeepSeek/gpt-oss cell.
        active_family = _infer_family_from_model_id(model_id)
        if active_family != family_alias.family:
            strict_skip_or_fail(
                f"cell {family_alias.family}: external RAPID_MLX_BASE_URL "
                f"serves {model_id!r} which maps to family "
                f"{active_family or 'unknown'!r}. Restart with the correct "
                f"model or set RAPID_MLX_AGENT_MATRIX_FAMILY={active_family or ''} "
                f"to shard on the family the external server serves."
            )
            return
        yield {
            "base_url": external,
            "model_id": model_id,
            "family": family_alias.family,
            "alias": family_alias.alias,
        }
        return

    # Auto-boot mode.
    if not _hf_cache_present(family_alias.hf_path):
        strict_skip_or_fail(
            f"HF weight cache miss for {family_alias.hf_path!r} "
            f"(need at least one .safetensors / .npz / .bin file locally). "
            f'Pre-download with: python3.12 -c "from huggingface_hub import '
            f"snapshot_download; snapshot_download('{family_alias.hf_path}')\""
        )
        return

    handle = _boot_server(family_alias, port)
    try:
        yield {
            "base_url": handle.base_url,
            "model_id": handle.model_id,
            "family": family_alias.family,
            "alias": family_alias.alias,
            "server_log": str(handle.log_path) if handle.log_path else None,
        }
    finally:
        _shutdown_server(handle)


# Weight-file suffixes that count as "real weights are on disk". Config
# / tokenizer files alone don't — codex #1033 round-1 BLOCKING #3.
_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin", ".gguf")


def _hf_cache_present(hf_path: str) -> bool:
    """Return True iff the HF snapshot cache holds at least one weight file.

    Codex #1033 round-1 BLOCKING #3: the earlier "any file counts"
    heuristic passed on a partial snapshot (config.json + tokenizer.json
    only) and auto-boot would then trigger a 50-65 GB weight download
    inside the pytest lifespan — turning a per-PR gate into an hours-
    long fetch. Require at least one weight-format file (.safetensors,
    .npz, .bin, .gguf) before declaring cache present.
    """
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hub = home / "hub"
    safe = "models--" + hf_path.replace("/", "--")
    snapshots = hub / safe / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        for entry in snap.iterdir():
            name = entry.name.lower()
            if name.endswith(_WEIGHT_SUFFIXES):
                # Symlinks are typical (HF layout); resolve and check
                # the pointed-to blob really exists and is non-empty
                # (a broken symlink or a zero-byte blob is not a hit).
                try:
                    resolved = entry.resolve(strict=True)
                    if resolved.stat().st_size > 0:
                        return True
                except (FileNotFoundError, OSError):
                    continue
    return False


# --------------------------------------------------------------------------- #
# Assertion helpers — shared across both matrices
# --------------------------------------------------------------------------- #


def assert_content_nonempty(text: str, ctx: str = "") -> None:
    """Assert the model produced non-empty visible content."""
    assert isinstance(text, str), f"{ctx}: expected str, got {type(text)!r}"
    assert text.strip(), f"{ctx}: empty content"


def assert_tool_call_shape(tool_call: dict[str, Any]) -> None:
    """Assert an OpenAI-shape tool call dict is valid."""
    assert isinstance(tool_call, dict), f"tool_call not a dict: {tool_call!r}"
    assert tool_call.get("id"), f"tool_call missing id: {tool_call!r}"
    assert tool_call.get("type") == "function", tool_call
    fn = tool_call.get("function")
    assert isinstance(fn, dict), f"tool_call.function not a dict: {fn!r}"
    assert fn.get("name"), f"tool_call.function missing name: {fn!r}"
    args = fn.get("arguments")
    assert isinstance(args, str), (
        f"tool_call.function.arguments must be JSON string: {args!r}"
    )
    try:
        json.loads(args)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"tool_call.function.arguments not JSON-parseable: {args!r} ({exc})"
        ) from exc


def assert_stream_deltas_valid(events: list[dict[str, Any]]) -> None:
    """Assert an OpenAI streaming response yielded well-formed deltas."""
    assert events, "no streaming events collected"
    assert any(
        ev.get("choices", [{}])[0].get("delta", {}).get("content") for ev in events
    ), f"no content deltas in {len(events)} events"


def assert_no_analysis_channel_leak(text: str) -> None:
    """Assert the openai-harmony ``analysis`` channel didn't leak into content.

    gpt-oss models emit ``<|channel|>analysis`` / ``<|channel|>final`` markers
    around chain-of-thought; the server must strip / route the analysis
    channel to ``reasoning_content``, not the visible answer. Regression
    fixture referenced by Continue #8990.
    """
    for marker in ("<|channel|>analysis", "analysis<|message|>", "<|start|>analysis"):
        assert marker not in text, (
            f"gpt-oss analysis-channel leak into content: found {marker!r} in {text!r}"
        )


def assert_no_think_tag_leak(text: str) -> None:
    """Assert ``<think>...</think>`` traces don't leak into visible content."""
    for marker in ("<think>", "</think>", "<|thinking|>", "<|/thinking|>"):
        assert marker not in text, (
            f"think-tag leak into content: found {marker!r} in {text!r}"
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "FamilyAlias",
    "assert_content_nonempty",
    "assert_no_analysis_channel_leak",
    "assert_no_think_tag_leak",
    "assert_stream_deltas_valid",
    "assert_tool_call_shape",
    "matrix_strict_mode",
    "strict_skip_or_fail",
]
