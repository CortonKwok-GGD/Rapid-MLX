#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-stage default-on validation gate for #558 constrained tool-calling.

Purpose (raullen's acceptance bar)
----------------------------------
After the llguidance migration (#1131) + constrained tool-calling (#558) land
with **default-on**, EVERY metric must be at PARITY with the pre-migration
baseline and SLIGHTLY BETTER on perf + correctness. This gate produces the
per-cell **baseline-ref vs post-change-ref** comparison that proves it, plus a
tool-calling perf/correctness signal.

What it runs (per git ref, per family)
--------------------------------------
* **Stage 1 — smoke sweep**: the existing per-cell agent×family + framework
  wire smokes (``tests/integrations/test_agents_matrix.py`` +
  ``test_frameworks_matrix.py``), run once against a server booted with the
  tool-calling constraint OFF (``RAPID_MLX_CONSTRAIN_TOOLS=0`` — free-form
  base) and once against a server booted default-ON (env unset — #558 PR-5).
  Requests keep ``tool_choice="auto"`` in both arms so the on arm exercises the
  real PR-5 auto-path (see ``test_default_on_deep._apply_constraint_mode``).
  Per cell we record pass/fail/skip/xfail + latency + any channel-marker leak
  (the matrix's ``assert_no_*_leak`` helpers fail the cell on a leak).
* **Stage 2 — deep cells**: ``tests/integrations/test_default_on_deep.py`` —
  a multi-turn tool loop, four varied schemas (enum / nested / required /
  additionalProperties:false), and a **negative control** that proves the
  llguidance constraint actually masks off-schema tokens. Run in both modes.

How to run it
-------------
The gate boots a FRESH server per ``(ref, family, mode)`` so the off/on arms
are a real SERVER-env toggle. It:
  1. materialises the given git ref in a sibling worktree (unless the ref is
     ``WORKTREE`` = "use the current worktree as-is"),
  2. builds an isolated venv there and installs ``.[guided,dev]``,
  3. for each mode boots ``rapid-mlx serve <family-alias>`` on ``--port``
     (non-operator) with the mode's ``RAPID_MLX_CONSTRAIN_TOOLS`` — off=``0``
     (free-form), on=unset (default-ON),
  4. runs Stage-1 + Stage-2 against that server via pytest ``--junit-xml``,
  5. emits ``<out>/<ref-label>/<family>/{off,on}.xml`` + a parsed
     ``results.json`` (per-cell status + latency + mode).

``compare`` diffs two refs' ``results.json`` (same mode) into a per-cell
PARITY / BETTER / REGRESSED table; ``compare-modes`` diffs off-vs-on WITHIN a
single ref's ``results.json`` (the free-form-vs-constrained parity table).

Typical invocations
-------------------
Validate ONE cheap family end-to-end on the CURRENT worktree (Task-4 shape,
no checkout, no contention with heavy families)::

    python scripts/default_on_gate.py run \
        --ref WORKTREE --families qwen36 --port 8123 \
        --out /tmp/gate-out --reuse-server-url http://127.0.0.1:8123/v1

Full 5-family baseline-vs-head comparison (run later when resources are free)::

    # 1) baseline ref (pre-#1131 = parent of 46223363):
    python scripts/default_on_gate.py run --ref 75b1fe3b \
        --families qwen36,gemma4,deepseek,gptoss,hy3 --port 8123 --out /tmp/gate-out
    # 2) post-change head:
    python scripts/default_on_gate.py run --ref HEAD \
        --families qwen36,gemma4,deepseek,gptoss,hy3 --port 8123 --out /tmp/gate-out
    # 3) compare:
    python scripts/default_on_gate.py compare \
        --baseline /tmp/gate-out/75b1fe3b --head /tmp/gate-out/HEAD

Notes
-----
* Ports: NEVER binds 8801 / 8772 / 8451 (operator services). Default 8123.
* Downloads: this script NEVER downloads a model — it only ``rapid-mlx serve``
  s aliases whose weights are already local / symlinked into the HF cache.
* HY3 is Ultra-only + strict-xfail; its cells XFAIL/skip without a 166 GB boot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_OPERATOR_PORTS = {8801, 8772, 8451}

# Pre-migration baseline = the parent of the #1131 llguidance migration commit
# (46223363). Documented here so the "full baseline" invocation is copy-paste.
BASELINE_REF = "75b1fe3b"  # docs(reference): opt-in cache flags (#1129), pre-#1131

_FAMILY_ALIASES = {
    "qwen36": "qwen3.5-4b-4bit",
    "gemma4": "gemma-4-12b-4bit",
    "deepseek": "deepseek-r1-32b-4bit",
    "gptoss": "gpt-oss-20b-mxfp4-q8",
    "hy3": "hy3-preview-4bit",
}

_MATRIX_FILES = [
    "tests/integrations/test_agents_matrix.py",
    "tests/integrations/test_frameworks_matrix.py",
]
_DEEP_FILE = "tests/integrations/test_default_on_deep.py"

_MODES = ("off", "on")


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    """One (cell, mode) outcome parsed from a junit report."""

    nodeid: str  # e.g. "test_agents_matrix.py::TestOpenCode::test_smoke[qwen36]"
    family: str
    mode: str  # "off" | "on"
    status: str  # "pass" | "fail" | "skip" | "xfail" | "xpass" | "error"
    duration_s: float
    leak: bool  # channel-marker leak surfaced in the failure text
    message: str = ""


@dataclass
class RunReport:
    ref_label: str
    families: list[str]
    port: int
    cells: list[CellResult] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# junit parsing
# --------------------------------------------------------------------------- #


def _classify(testcase: ET.Element) -> tuple[str, str, bool]:
    """Return (status, message, leak) for a junit ``<testcase>`` element."""
    failure = testcase.find("failure")
    error = testcase.find("error")
    skipped = testcase.find("skipped")

    def _leak(text: str) -> bool:
        low = text.lower()
        return "leak" in low or "<|channel|>" in text or "<think>" in text

    if error is not None:
        msg = error.get("message", "") or (error.text or "")
        return "error", msg, _leak(msg)
    if failure is not None:
        msg = failure.get("message", "") or (failure.text or "")
        # pytest encodes an xpass(strict) as a failure whose message says so.
        if "xpass" in msg.lower() or "XPASS" in msg:
            return "xpass", msg, _leak(msg)
        return "fail", msg, _leak(msg)
    if skipped is not None:
        msg = skipped.get("message", "") or (skipped.text or "")
        # pytest marks an xfail'd cell as skipped with type="pytest.xfail".
        if skipped.get("type", "").endswith("xfail") or "xfail" in msg.lower():
            return "xfail", msg, False
        return "skip", msg, False
    return "pass", "", False


def parse_junit(xml_path: Path, family: str, mode: str) -> list[CellResult]:
    """Parse a pytest junit-xml file into per-cell CellResults."""
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    cells: list[CellResult] = []
    for tc in tree.iter("testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        # Reconstruct a stable, file-relative nodeid.
        # classname is like "tests.integrations.test_agents_matrix.TestOpenCode"
        parts = classname.split(".")
        file_part = next(
            (p for p in parts if p.startswith("test_")), parts[-1] if parts else ""
        )
        cls_part = parts[-1] if parts and parts[-1].startswith("Test") else ""
        nodeid = f"{file_part}.py::{cls_part}::{name}" if cls_part else f"{file_part}.py::{name}"
        status, message, leak = _classify(tc)
        cells.append(
            CellResult(
                nodeid=nodeid,
                family=family,
                mode=mode,
                status=status,
                duration_s=float(tc.get("time", "0") or 0.0),
                leak=leak,
                message=message[:400],
            )
        )
    return cells


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def _wait_for_server(base_url: str, timeout_s: float = 240.0) -> str | None:
    """Poll /v1/models until the server answers; return served model_id."""
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=4) as r:  # noqa: S310
                data = json.loads(r.read())
                mids = [m["id"] for m in data.get("data", [])]
                if mids:
                    return mids[0]
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(2)
    return None


def _boot_server(
    repo_dir: Path,
    python_bin: Path,
    alias: str,
    port: int,
    log_path: Path,
    env_overrides: dict[str, str | None] | None = None,
) -> subprocess.Popen:
    """Launch ``rapid-mlx serve <alias>`` on the given port; return the Popen.

    ``env_overrides`` sets (or, when a value is ``None``, UNSETS) environment
    variables in the server child process. The gate uses this to make the
    off/on arms a real SERVER-env toggle of ``RAPID_MLX_CONSTRAIN_TOOLS``
    (#558 PR-5 default-on) rather than a per-request ``tool_choice`` swap:
    off boots with ``RAPID_MLX_CONSTRAIN_TOOLS=0`` (free-form), on unsets it so
    the server default (ON) applies. Unsetting is explicit so an opt-out value
    inherited from the gate's own environment can never leak into the on arm.
    """
    log_fh = open(log_path, "w")  # noqa: SIM115 — handed to Popen, closed on stop
    cmd = [
        str(python_bin),
        "-m",
        "vllm_mlx.cli",
        "serve",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "INFO",
    ]
    env = os.environ.copy()
    for key, val in (env_overrides or {}).items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    return subprocess.Popen(
        cmd, cwd=str(repo_dir), stdout=log_fh, stderr=subprocess.STDOUT, env=env
    )


# --------------------------------------------------------------------------- #
# Worktree / venv materialisation
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _materialise_ref(ref: str, workdir: Path) -> tuple[Path, Path]:
    """Return (repo_dir, python_bin) for the given ref.

    ``ref == "WORKTREE"`` uses the current worktree in-place (no checkout, no
    fresh venv) — this is the Task-4 cheap-family validation path. Any other
    ref is checked out into a sibling git worktree with its own venv.
    """
    if ref == "WORKTREE":
        root = _repo_root()
        venv_py = root / ".venv" / "bin" / "python"
        if not venv_py.exists():
            raise SystemExit(
                f"WORKTREE mode needs an existing venv at {venv_py}; create it "
                "with: python3.12 -m venv .venv && "
                "./.venv/bin/pip install -e '.[guided,dev]'"
            )
        return root, venv_py

    root = _repo_root()
    wt_dir = workdir / f"ref-{ref.replace('/', '_')}"
    if not wt_dir.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_dir), ref],
            cwd=str(root),
            check=True,
        )
    venv_dir = wt_dir / ".venv"
    venv_py = venv_dir / "bin" / "python"
    if not venv_py.exists():
        subprocess.run(["python3.12", "-m", "venv", str(venv_dir)], check=True)
        subprocess.run(
            [str(venv_dir / "bin" / "pip"), "install", "-e", ".[guided,dev]", "--quiet"],
            cwd=str(wt_dir),
            check=True,
        )
    return wt_dir, venv_py


# --------------------------------------------------------------------------- #
# One (family, mode) pytest invocation
# --------------------------------------------------------------------------- #


def _run_stage(
    repo_dir: Path,
    python_bin: Path,
    family: str,
    mode: str,
    base_url: str,
    out_dir: Path,
) -> Path:
    """Run Stage-1 (matrix) + Stage-2 (deep) for one (family, mode).

    Returns the junit xml path. Uses ``RAPID_MLX_MATRIX_STRICT=1`` so a
    server/route/SDK failure is a hard red (release-gate semantics), and
    scopes the run to one family via ``RAPID_MLX_AGENT_MATRIX_FAMILY``.
    """
    xml_path = out_dir / f"{mode}.xml"
    env = os.environ.copy()
    env["RAPID_MLX_BASE_URL"] = base_url
    env["RAPID_MLX_AGENT_MATRIX_FAMILY"] = family
    env["RAPID_MLX_MATRIX_STRICT"] = "1"
    env["RAPID_MLX_TOOL_CONSTRAINT"] = mode  # off | on
    targets = _MATRIX_FILES + [_DEEP_FILE]
    cmd = [
        str(python_bin),
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junit-xml={xml_path}",
        "-o",
        "junit_logging=all",  # capture stdout (latency/negctrl breadcrumbs)
    ]
    subprocess.run(cmd, cwd=str(repo_dir), env=env)  # non-zero on red is fine
    return xml_path


def _server_env_for_mode(mode: str) -> dict[str, str | None]:
    """Return the server-process env override implementing the off/on arm.

    #558 PR-5 default-on: constrained tool-calling is ON unless
    ``RAPID_MLX_CONSTRAIN_TOOLS`` is ``0``/``off``/``false``.

    * **off** — free-form base: set ``RAPID_MLX_CONSTRAIN_TOOLS=0`` (opt OUT).
    * **on**  — default-on auto-path: UNSET the var (``None``) so the server's
      own default (ON) applies, even if the gate's own environment happens to
      carry an opt-out value.

    Requests keep ``tool_choice="auto"`` in both arms (see
    ``test_default_on_deep._apply_constraint_mode``), so the server env is the
    only independent variable in the off-vs-on parity comparison.
    """
    if mode == "off":
        return {"RAPID_MLX_CONSTRAIN_TOOLS": "0"}
    return {"RAPID_MLX_CONSTRAIN_TOOLS": None}  # unset => server default (ON)


# --------------------------------------------------------------------------- #
# run subcommand
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    if args.port in _OPERATOR_PORTS:
        raise SystemExit(
            f"refusing to bind operator port {args.port} "
            f"(reserved: {sorted(_OPERATOR_PORTS)}) — pick another --port"
        )
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for f in families:
        if f not in _FAMILY_ALIASES:
            raise SystemExit(f"unknown family {f!r}; valid: {sorted(_FAMILY_ALIASES)}")

    out_root = Path(args.out)
    ref_label = args.ref if args.ref != "WORKTREE" else "WORKTREE"
    workdir = out_root / "_worktrees"
    workdir.mkdir(parents=True, exist_ok=True)

    repo_dir, python_bin = _materialise_ref(args.ref, workdir)
    print(f"[gate] ref={ref_label} repo_dir={repo_dir} python={python_bin}")

    report = RunReport(ref_label=ref_label, families=families, port=args.port)

    for family in families:
        alias = _FAMILY_ALIASES[family]
        fam_out = out_root / ref_label / family
        fam_out.mkdir(parents=True, exist_ok=True)
        base_url = args.reuse_server_url or f"http://127.0.0.1:{args.port}/v1"

        if args.reuse_server_url:
            # A reused server has ONE fixed RAPID_MLX_CONSTRAIN_TOOLS config, so
            # the off/on arms cannot be a real server-env toggle here — both
            # arms hit the same server. Kept only for quick single-arm smokes;
            # the parity matrix uses the boot-per-(family,mode) path below.
            print(
                f"[gate] {family}: reusing server at {base_url} — WARNING: "
                "off/on SERVER-env toggle inactive under --reuse-server-url; "
                "both arms exercise the reused server's own constraint config"
            )
            model_id = _wait_for_server(base_url, timeout_s=20)
            if model_id is None:
                print(f"[gate] {family}: WARN no server at {base_url} — cells will skip/fail")
            for mode in _MODES:
                print(f"[gate] {family}/{mode}: running stages (reused server)...")
                xml_path = _run_stage(
                    repo_dir, python_bin, family, mode, base_url, fam_out
                )
                report.cells.extend(parse_junit(xml_path, family, mode))
            continue

        # Boot a FRESH server per (family, mode) so the off/on arm is a real
        # SERVER-env toggle of RAPID_MLX_CONSTRAIN_TOOLS (#558 PR-5 default-on),
        # NOT a per-request tool_choice swap. Requests keep tool_choice="auto"
        # in both arms.
        for mode in _MODES:
            env_overrides = _server_env_for_mode(mode)
            constrain = env_overrides["RAPID_MLX_CONSTRAIN_TOOLS"]
            log_path = fam_out / f"serve-{mode}.log"
            print(
                f"[gate] {family}/{mode}: booting rapid-mlx serve {alias} on "
                f":{args.port} (RAPID_MLX_CONSTRAIN_TOOLS="
                f"{constrain if constrain is not None else '<unset:default-ON>'})"
            )
            proc = _boot_server(
                repo_dir, python_bin, alias, args.port, log_path, env_overrides
            )
            try:
                model_id = _wait_for_server(base_url, timeout_s=args.boot_timeout)
                if model_id is None:
                    print(
                        f"[gate] {family}/{mode}: server did not come up within "
                        f"{args.boot_timeout}s — see {log_path}; skipping this arm"
                    )
                    continue
                print(f"[gate] {family}/{mode}: server up, model_id={model_id}")
                print(f"[gate] {family}/{mode}: running stages...")
                xml_path = _run_stage(
                    repo_dir, python_bin, family, mode, base_url, fam_out
                )
                report.cells.extend(parse_junit(xml_path, family, mode))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()

    results_path = out_root / ref_label / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "ref_label": report.ref_label,
                "families": report.families,
                "port": report.port,
                "cells": [asdict(c) for c in report.cells],
            },
            indent=2,
        )
    )
    print(f"[gate] wrote {results_path} ({len(report.cells)} cell-results)")
    _print_run_summary(report)
    return 0


def _print_run_summary(report: RunReport) -> None:
    print(f"\n=== RUN SUMMARY ref={report.ref_label} ===")
    print(f"{'nodeid':<62} {'mode':<4} {'status':<7} {'lat(s)':>7} leak")
    print("-" * 92)
    for c in sorted(report.cells, key=lambda x: (x.nodeid, x.mode)):
        node = c.nodeid if len(c.nodeid) <= 62 else "…" + c.nodeid[-61:]
        print(
            f"{node:<62} {c.mode:<4} {c.status:<7} {c.duration_s:>7.2f} "
            f"{'YES' if c.leak else '-'}"
        )


# --------------------------------------------------------------------------- #
# compare subcommand
# --------------------------------------------------------------------------- #

_OK_STATUSES = {"pass", "xfail"}  # xfail is an APPROVED non-red outcome


def _verdict(base: CellResult | None, head: CellResult | None) -> str:
    """Classify a per-(cell,mode) pair as PARITY / BETTER / REGRESSED / NEW / GONE."""
    if base is None and head is None:
        return "MISSING"
    if base is None:
        return "NEW"
    if head is None:
        return "GONE"
    b_ok = base.status in _OK_STATUSES
    h_ok = head.status in _OK_STATUSES
    if b_ok and not h_ok:
        return "REGRESSED"
    if not b_ok and h_ok:
        return "BETTER-correctness"
    if b_ok and h_ok:
        # both green — judge on latency + leak.
        if head.leak and not base.leak:
            return "REGRESSED"  # new channel leak
        if base.duration_s > 0 and head.duration_s > 0:
            ratio = head.duration_s / base.duration_s
            if ratio <= 0.90:
                return "BETTER-perf"
            if ratio >= 1.15:
                return "REGRESSED-perf"
        return "PARITY"
    # both non-green
    if base.status == head.status:
        return "PARITY"
    return "CHANGED"


def _load(path: Path) -> dict[tuple[str, str], CellResult]:
    data = json.loads((path / "results.json").read_text())
    out: dict[tuple[str, str], CellResult] = {}
    for c in data["cells"]:
        cr = CellResult(**c)
        out[(cr.nodeid, cr.mode)] = cr
    return out


def cmd_compare(args: argparse.Namespace) -> int:
    base = _load(Path(args.baseline))
    head = _load(Path(args.head))
    keys = sorted(set(base) | set(head))

    print(f"\n=== COMPARISON  baseline={args.baseline}  head={args.head} ===")
    print(f"{'nodeid':<58} {'mode':<4} {'base':<6} {'head':<6} {'blat':>6} {'hlat':>6} verdict")
    print("-" * 108)
    counts: dict[str, int] = {}
    regressions: list[str] = []
    for nodeid, mode in keys:
        b = base.get((nodeid, mode))
        h = head.get((nodeid, mode))
        v = _verdict(b, h)
        counts[v] = counts.get(v, 0) + 1
        if v.startswith("REGRESSED"):
            regressions.append(f"{nodeid} [{mode}]: {v}")
        node = nodeid if len(nodeid) <= 58 else "…" + nodeid[-57:]
        print(
            f"{node:<58} {mode:<4} "
            f"{(b.status if b else '-'):<6} {(h.status if h else '-'):<6} "
            f"{(b.duration_s if b else 0):>6.2f} {(h.duration_s if h else 0):>6.2f} {v}"
        )

    print("\n=== VERDICT TALLY ===")
    for v in sorted(counts):
        print(f"  {v:<20} {counts[v]}")

    if regressions:
        print("\n*** REGRESSIONS (block default-on) ***")
        for r in regressions:
            print(f"  - {r}")
        print(f"\nGATE: RED — {len(regressions)} regressed cell(s)")
        return 1
    print("\nGATE: GREEN — every cell at parity or better")
    return 0


def cmd_compare_modes(args: argparse.Namespace) -> int:
    """Diff the off-arm vs on-arm cells WITHIN a single run's results.json.

    This is the Stage-1b parity comparison the #558 PR-5 gate exists to emit:
    free-form base (server ``RAPID_MLX_CONSTRAIN_TOOLS=0`` -> mode=off cells)
    vs default-on auto-path (server default ON -> mode=on cells), on the SAME
    worktree ref. It reuses ``_verdict`` (base=off, head=on) so PARITY /
    BETTER / REGRESSED semantics match the cross-ref ``compare``: an on-arm
    cell that newly fails vs its off-arm base, or is materially slower, is a
    REGRESSED that blocks default-on.
    """
    cells = _load(Path(args.run))
    by_node: dict[str, dict[str, CellResult]] = {}
    for (nodeid, mode), cr in cells.items():
        by_node.setdefault(nodeid, {})[mode] = cr

    print(f"\n=== OFF-vs-ON PARITY  run={args.run} ===")
    print("(base=off free-form  head=on default-on auto-path; same worktree ref)")
    print(f"{'nodeid':<58} {'off':<6} {'on':<6} {'olat':>6} {'nlat':>6} verdict")
    print("-" * 106)
    counts: dict[str, int] = {}
    regressions: list[str] = []
    for nodeid in sorted(by_node):
        off = by_node[nodeid].get("off")
        on = by_node[nodeid].get("on")
        v = _verdict(off, on)
        counts[v] = counts.get(v, 0) + 1
        if v.startswith("REGRESSED"):
            regressions.append(f"{nodeid}: {v}")
        node = nodeid if len(nodeid) <= 58 else "…" + nodeid[-57:]
        print(
            f"{node:<58} "
            f"{(off.status if off else '-'):<6} {(on.status if on else '-'):<6} "
            f"{(off.duration_s if off else 0):>6.2f} "
            f"{(on.duration_s if on else 0):>6.2f} {v}"
        )

    print("\n=== VERDICT TALLY ===")
    for v in sorted(counts):
        print(f"  {v:<20} {counts[v]}")

    if regressions:
        print("\n*** REGRESSIONS (block default-on) ***")
        for r in regressions:
            print(f"  - {r}")
        print(f"\nGATE: RED — {len(regressions)} regressed cell(s)")
        return 1
    print("\nGATE: GREEN — every on-arm cell at parity or better vs off-arm base")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the two-stage gate for a git ref")
    r.add_argument(
        "--ref",
        required=True,
        help="git ref to test, or 'WORKTREE' to use the current worktree as-is",
    )
    r.add_argument(
        "--families",
        default="qwen36",
        help="comma-separated families (qwen36,gemma4,deepseek,gptoss,hy3)",
    )
    r.add_argument("--port", type=int, default=8123, help="serve port (never operator ports)")
    r.add_argument("--out", default="/tmp/gate-out", help="output root dir")
    r.add_argument("--boot-timeout", type=float, default=300.0, help="server boot wait (s)")
    r.add_argument(
        "--reuse-server-url",
        default=None,
        help="if set, skip booting and run against this already-running server "
        "(e.g. http://127.0.0.1:8123/v1) — single family only",
    )
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="diff two run outputs into a parity table")
    c.add_argument("--baseline", required=True, help="baseline ref out dir (…/<ref-label>)")
    c.add_argument("--head", required=True, help="head ref out dir (…/<ref-label>)")
    c.set_defaults(func=cmd_compare)

    cm = sub.add_parser(
        "compare-modes",
        help="diff off-arm vs on-arm cells within one run's results.json "
        "(the #558 PR-5 free-form-vs-constrained parity table)",
    )
    cm.add_argument(
        "--run",
        required=True,
        help="a run output dir containing results.json (…/<ref-label>)",
    )
    cm.set_defaults(func=cmd_compare_modes)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
