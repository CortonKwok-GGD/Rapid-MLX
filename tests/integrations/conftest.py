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
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
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
    """A strong-per-family alias used across the matrices.

    ``serve_args`` are the extra CLI flags appended to
    ``rapid-mlx serve <alias>`` at auto-boot time. **Empty by default** —
    the matrix exercises the same ``rapid-mlx serve <alias>`` invocation
    that PyPI / Homebrew users run, so the alias-auto-detect table in
    ``vllm_mlx/aliases.json`` is under test. A regression in
    auto-detect surfaces as a matrix cell red.

    Populate ``serve_args`` (via ``RAPID_MLX_MATRIX_EXPLICIT_PARSER=1``
    env override, see ``_serve_command``) to belt-and-braces the flags
    at the CLI layer — useful for a second matrix shard that isolates
    engine bugs from auto-detect bugs. Not the default.

    Codex #1033 round-3 → round-4 evolution: round-3 wanted the explicit
    parser flags (defense-in-depth); round-4 pushed back that the
    matrix must also cover the naive-user auto-detect path. The
    resolution is: default = auto-detect (the naive-user path);
    ``RAPID_MLX_MATRIX_EXPLICIT_PARSER=1`` env flips to belt-and-braces
    for a separate shard.
    """

    family: str  # matrix column key: qwen36 / gemma4 / deepseek / gptoss
    alias: str  # rapid-mlx alias string (positional model arg)
    hf_path: str  # HuggingFace repo id (for cache probing)
    tool_call_parser: str  # documented parser (for skip-inference + explicit flags)
    reasoning_parser: str  # documented reasoning parser
    reason: str  # why this strong pick
    explicit_serve_args: tuple[str, ...] = ()  # opt-in override — see _serve_command


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
        explicit_serve_args=(
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder_xml",
        ),
    ),
    "gemma4": FamilyAlias(
        family="gemma4",
        alias="gemma-4-31b-4bit",
        hf_path="mlx-community/gemma-4-31b-it-4bit",
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        reason="Gemma 4 31B strong pick (12B fails tool-calling — model 降智)",
        explicit_serve_args=(
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "gemma4",
        ),
    ),
    "deepseek": FamilyAlias(
        family="deepseek",
        alias="deepseek-v4-flash-8bit",
        hf_path="mlx-community/DeepSeek-V4-Flash-8bit",
        tool_call_parser="deepseek",
        reasoning_parser="deepseek_r1",
        reason="DeepSeek V4 Flash 8bit — only Tier-1 DeepSeek MLX quant on-shelf",
        explicit_serve_args=(
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "deepseek",
        ),
    ),
    "gptoss": FamilyAlias(
        family="gptoss",
        alias="gpt-oss-120b-mxfp4-q8",
        hf_path="mlx-community/gpt-oss-120b-MXFP4-Q8",
        tool_call_parser="harmony",
        reasoning_parser="harmony",
        reason="gpt-oss 120B Harmony strong pick (20B skips reasoning channel)",
        explicit_serve_args=(
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "harmony",
        ),
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


def _serve_command(alias: FamilyAlias, port: int) -> list[str]:
    """Return argv for ``rapid-mlx serve <alias> --port <port> [extra]``.

    Defaults to ``python3.12 -m vllm_mlx.cli`` (worktree-safe: editable
    install without wrapper indirection). Override via ``RAPID_MLX_SERVE_BIN``
    when the ``rapid-mlx`` wrapper points at the correct venv.

    Codex #1033 round-1 NIT #2: use ``shlex.split`` so a quoted path like
    ``RAPID_MLX_SERVE_BIN='/opt/homebrew/opt/python@3.12/bin/python3.12 -m vllm_mlx.cli'``
    or an argv element containing a space parses correctly.

    Codex #1033 round-3 → round-4 evolution: the default is now the
    **naive-user command** — plain ``serve <alias> --port <port>``, no
    extra flags. This exercises the alias auto-detect table that PyPI /
    Homebrew users hit. Set ``RAPID_MLX_MATRIX_EXPLICIT_PARSER=1`` to
    append the per-family ``explicit_serve_args`` (``--enable-auto-tool-
    choice`` + ``--tool-call-parser <parser>``) — useful for a CI shard
    that isolates engine bugs from auto-detect bugs.
    """
    bin_override = os.environ.get("RAPID_MLX_SERVE_BIN", "").strip()
    if bin_override:
        argv = shlex.split(bin_override)
    else:
        argv = [sys.executable, "-m", "vllm_mlx.cli"]
    argv += ["serve", alias.alias, "--port", str(port)]
    if os.environ.get("RAPID_MLX_MATRIX_EXPLICIT_PARSER", "").strip() == "1":
        argv += list(alias.explicit_serve_args)
    return argv


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _wait_for_ready(
    port: int,
    timeout_s: int,
    proc: subprocess.Popen | None = None,
) -> bool:
    """Poll /v1/models (200) until ready or timeout.

    Codex #1033 round-5 NIT #1: check ``proc.poll()`` on each iteration
    so a child that died at import time (e.g. missing dependency,
    stale editable install) fails fast instead of appearing to hang
    for the full boot budget. Returns False if the process is gone.
    """
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            # Child exited before /v1/models came up — no point polling.
            return False
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

    # Codex #1033 round-5 NIT #2: use ``NamedTemporaryFile(delete=False)``
    # so a symlink at ``/tmp/rapid-mlx-matrix-...`` (planted by a shared-
    # machine attacker) can't be followed to clobber an arbitrary file.
    # ``delete=False`` keeps the log around after the fixture exits so
    # operators can post-mortem.
    log_f = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"rapid-mlx-matrix-{alias.family}-{port}-",
        suffix=".log",
        delete=False,
    )
    log_path = Path(log_f.name)
    cmd = _serve_command(alias, port)
    proc = None
    try:
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        # Codex #1033 round-7 BLOCKING #2: catch OSError broadly (covers
        # ``PermissionError`` from a chmod-blocked ``RAPID_MLX_SERVE_BIN``
        # as well as ``FileNotFoundError``). Anything else escapes — the
        # ``finally`` still closes the parent-side fd.
        except OSError as exc:
            strict_skip_or_fail(
                f"could not exec {cmd!r}: {exc}. Set RAPID_MLX_SERVE_BIN or "
                f"ensure `{sys.executable} -m vllm_mlx.cli` is importable. "
                f"(log path: {log_path})"
            )
            return _ServerHandle(None, port, "", "", None)  # unreachable in strict
    finally:
        # Close the parent-side fd — child has dup'd its own copy for
        # writing. Codex #1033 round-1 BLOCKING #2 fold.
        log_f.close()

    if not _wait_for_ready(port, SERVER_BOOT_TIMEOUT_S, proc=proc):
        # Best-effort teardown — codex #1033 round-4 BLOCKING #5: after
        # ``proc.kill()`` reap the child with ``proc.wait()`` so a boot-
        # timeout doesn't leave a zombie process; mirrors the normal
        # ``_shutdown_server`` path.
        try:
            proc.send_signal(2)
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
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
            # Session scope on the parametrization means each family_alias
            # gets its own instance across the session; combined with the
            # module-level ``_ACTIVE_SERVER`` cache below, this gives us
            # "one boot per family, torn down before the next family
            # starts" — matches the sequential OOM budget the M3 Ultra
            # requires with operator services running on 8801 / 8772.
            scope="session",
        )


# Module-level cache: (family_key) → _ServerHandle. Codex #1033 round-7
# BLOCKING #1 fold — a function-scoped fixture around this cache gives
# each test cell its own fixture call, but a fresh boot only when the
# family flips. When the family flips, the previous handle is torn down
# BEFORE the new one boots; that guarantees the "Qwen 3.6 → shutdown →
# Gemma 4 → shutdown" sequencing the OOM budget requires and can't be
# violated by pytest's session-scope reuse rules.
_ACTIVE_SERVER: dict[str, Any] = {"family": None, "handle": None, "meta": None}


def _teardown_active_server() -> None:
    handle = _ACTIVE_SERVER.get("handle")
    if handle is not None:
        _shutdown_server(handle)
    _ACTIVE_SERVER["family"] = None
    _ACTIVE_SERVER["handle"] = None
    _ACTIVE_SERVER["meta"] = None


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


@pytest.fixture
def rapid_mlx_server(
    family_alias: FamilyAlias, request: pytest.FixtureRequest
) -> dict[str, Any]:
    """Return metadata for a rapid-mlx server serving ``family_alias``.

    Two modes:

    * **External** (env ``RAPID_MLX_BASE_URL`` set): probe /v1/models,
      assert the reported model_id maps to ``family_alias.family``, and
      return without touching any subprocess. Local-dev shortcut so a
      large model already loaded elsewhere doesn't get re-booted. If
      the external server serves a different family, the cell skips
      (or fails in strict mode) — matching the guard the auto-boot
      path enforces by definition.
    * **Auto-boot** (default): boot ``rapid-mlx serve <alias>`` on
      ``RAPID_MLX_MATRIX_PORT`` (default 8802), return, teardown when
      the family flips (or at session end via ``_teardown_active_server``).

    Fixture is **function-scoped** but backed by the module-level
    ``_ACTIVE_SERVER`` cache — codex #1033 round-7 BLOCKING #1 fold:
    a session-scoped fixture parametrized across families relies on
    pytest's finalizer ordering to sequence "boot Qwen → teardown Qwen
    → boot Gemma → ..." and that ordering is not guaranteed when
    parametrization is done via ``pytest_generate_tests``. Function
    scope + explicit cache flip is deterministic: when we see a family
    different from the currently-active one, we tear down the old
    handle BEFORE booting the new one, which is exactly the OOM
    sequencing budget we need.

    Teardown of the *last* family's server is registered as a session
    finalizer so we don't leak a running process at pytest exit.
    """
    external = os.environ.get("RAPID_MLX_BASE_URL", "").strip().rstrip("/")
    if external:
        # Codex #1033 round-5 BLOCKING #1: normalize so the probe hits
        # /v1/models regardless of whether the operator set the /v1 base
        # URL (documented) or the host root (natural miss). Downstream
        # SDK clients receive the /v1 base URL either way.
        if not external.endswith("/v1"):
            external = external + "/v1"
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
            return {}
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
            return {}
        return {
            "base_url": external,
            "model_id": model_id,
            "family": family_alias.family,
            "alias": family_alias.alias,
        }

    # Auto-boot mode. Check module-level cache first: if the currently-
    # loaded family matches, reuse the running handle — one boot per
    # family, N cells share it. When the family flips, tear the current
    # one down BEFORE booting the new one (OOM budget), then cache.
    if _ACTIVE_SERVER["family"] == family_alias.family and _ACTIVE_SERVER["handle"]:
        meta: dict[str, Any] = _ACTIVE_SERVER["meta"]
        return meta

    # Family flipped (or first cell of the session) — flush any active
    # server before booting the new one.
    if _ACTIVE_SERVER["family"] is not None:
        _teardown_active_server()

    # Register session finalizer once (subsequent registrations are
    # harmless idempotent adds on the same callable, but we gate on the
    # cache pointer to avoid stacking N teardowns).
    if _ACTIVE_SERVER.get("finalizer_registered") is not True:
        request.session.addfinalizer(_teardown_active_server)
        _ACTIVE_SERVER["finalizer_registered"] = True

    port = _matrix_port()

    if not _hf_cache_present(family_alias.hf_path):
        strict_skip_or_fail(
            f"HF weight cache miss for {family_alias.hf_path!r} "
            f"(need at least one .safetensors / .npz / .bin file locally). "
            f'Pre-download with: python3.12 -c "from huggingface_hub import '
            f"snapshot_download; snapshot_download('{family_alias.hf_path}')\""
        )
        return {}

    handle = _boot_server(family_alias, port)
    meta = {
        "base_url": handle.base_url,
        "model_id": handle.model_id,
        "family": family_alias.family,
        "alias": family_alias.alias,
        "server_log": str(handle.log_path) if handle.log_path else None,
    }
    _ACTIVE_SERVER["family"] = family_alias.family
    _ACTIVE_SERVER["handle"] = handle
    _ACTIVE_SERVER["meta"] = meta
    return meta


# Weight-file suffixes that count as "real weights are on disk". Config
# / tokenizer files alone don't — codex #1033 round-1 BLOCKING #3.
_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin", ".gguf")

# HF sharded-weight filename pattern: ``model-00007-of-00033.safetensors``,
# ``pytorch_model-00007-of-00033.bin``, etc. Any single shard we find
# tells us there are ``N`` total shards, and we can verify all are present.
# Codex #1033 round-6 BLOCKING #2 fold.
_SHARD_FILENAME_RE = re.compile(
    r"^(?P<stem>[\w.\-]+?)-(?P<idx>\d+)-of-(?P<total>\d+)"
    r"(?P<ext>\.safetensors|\.npz|\.bin|\.gguf)$"
)


def _resolve_hf_hub_cache() -> Path:
    """Return the HF hub cache root, honoring the standard env vars.

    Prefers ``huggingface_hub.constants.HF_HUB_CACHE`` (which folds in
    the standard priority order: ``HF_HUB_CACHE`` > ``HUGGINGFACE_HUB_CACHE``
    > ``HF_HOME`` + ``/hub`` > ``~/.cache/huggingface/hub``). Falls back
    to the same priority manually if ``huggingface_hub`` isn't
    importable at fixture-collection time (unlikely — it's a rapid-mlx
    core dep — but the fallback keeps the conftest importable in
    weird environments).

    Codex #1033 round-5 BLOCKING #2 fold.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            val = os.environ.get(env, "").strip()
            if val:
                return Path(val)
        home = os.environ.get("HF_HOME", "").strip()
        if home:
            return Path(home) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_cache_present(hf_path: str) -> bool:
    """Return True iff a COMPLETE snapshot is on disk.

    Codex #1033 round-1 → round-6 evolution:
    * round-1 BLOCKING #3: reject snapshots that only carry
      config/tokenizer files — require at least one weight file.
    * round-3 BLOCKING #2: use ``rglob`` so subdir weight layouts don't
      falsely miss.
    * round-5 BLOCKING #2: honor ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE``.
    * round-6 BLOCKING #2 (this fold): a *partial* sharded snapshot
      (say 8 of 33 safetensors complete) previously passed because the
      loop returned True on the first present weight. Now we read
      ``model.safetensors.index.json`` when present and require every
      referenced shard to exist on disk with a non-zero resolved size.
      Snapshots without an index (single-file models) still pass on
      the "at least one weight" rule.
    """
    hub = _resolve_hf_hub_cache()
    safe = "models--" + hf_path.replace("/", "--")
    snapshots = hub / safe / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        if not snap.is_dir():
            continue
        # Prefer the sharded-index path: if ``model.safetensors.index.json``
        # exists, verify every referenced shard is on disk. This is the
        # only way to distinguish "8 of 33 shards" from "the model was
        # shipped as a single safetensors file". Codex #1033 round-7
        # NIT #1 fold: seed with the top-level index only (rglob below
        # already covers it), then extend with any sub-dir instances.
        index_paths: list[Path] = [snap / "model.safetensors.index.json"]
        # Also scan for indices in sub-dirs (weights/, shards/ layouts).
        for extra in snap.rglob("model.safetensors.index.json"):
            if extra not in index_paths:
                index_paths.append(extra)
        found_index = False
        for index_path in index_paths:
            if not index_path.exists():
                continue
            found_index = True
            try:
                index = json.loads(index_path.read_text())
                weight_map: dict[str, str] = index.get("weight_map") or {}
                shards = set(weight_map.values())
            except (OSError, json.JSONDecodeError):
                continue
            if not shards:
                continue
            root = index_path.parent
            complete = True
            for shard in shards:
                shard_path = root / shard
                try:
                    resolved = shard_path.resolve(strict=True)
                    if resolved.stat().st_size <= 0:
                        complete = False
                        break
                except (FileNotFoundError, OSError):
                    complete = False
                    break
            if complete:
                return True
        if found_index:
            # A sharded model whose index is present but shards are
            # missing → partial cache. Don't fall through to the
            # "any weight" heuristic, which would falsely report ready.
            continue
        # No safetensors index — walk the snapshot for weight files and
        # check the shard-name pattern. If any file is named
        # ``model-XX-of-YY.<ext>``, YY tells us the shard count and we
        # require all YY shards to be present with non-zero size.
        weight_files: dict[Path, Path] = {}  # dir → sample weight file
        for entry in snap.rglob("*"):
            if not entry.is_file() and not entry.is_symlink():
                continue
            if not entry.name.lower().endswith(_WEIGHT_SUFFIXES):
                continue
            weight_files.setdefault(entry.parent, entry)

        for wdir, sample in weight_files.items():
            m = _SHARD_FILENAME_RE.match(sample.name)
            if m:
                total = int(m.group("total"))
                stem = m.group("stem")
                ext = m.group("ext")
                complete = True
                for i in range(1, total + 1):
                    shard_name = f"{stem}-{i:05d}-of-{total:05d}{ext}"
                    shard_path = wdir / shard_name
                    try:
                        resolved = shard_path.resolve(strict=True)
                        if resolved.stat().st_size <= 0:
                            complete = False
                            break
                    except (FileNotFoundError, OSError):
                        complete = False
                        break
                if complete:
                    return True
            else:
                # Non-sharded single weight file — check it's non-empty.
                try:
                    resolved = sample.resolve(strict=True)
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
