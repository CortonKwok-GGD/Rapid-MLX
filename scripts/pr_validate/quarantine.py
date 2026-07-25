# SPDX-License-Identifier: Apache-2.0
"""Known-flaky test registry — loader + node-id matcher.

Backs dev-flow proposal item ③ (flake tracking). The registry lives in
``quarantine.yaml`` next to this module; a test listed there that fails
does NOT block a PR (the ``full_unit`` gate consults ``partition_failures``
to split a failure set into blocking vs quarantined). See the YAML header
for the discipline on what may be added.

Design notes:
  * Loading is FAIL-SAFE at the call site, not here. ``load_quarantine``
    returns ``[]`` for a missing file (no quarantine configured is a
    legitimate state) but RAISES ``QuarantineError`` for a present-but-
    malformed file — that's an author mistake worth surfacing. The
    gating step catches the error and falls back to an EMPTY quarantine,
    i.e. a *stricter* gate: a broken registry can never make the gate
    pass something it otherwise wouldn't.
  * Matching is by exact pytest node id, with one deliberate
    convenience: an entry id carrying no ``[...]`` also matches every
    parametrization of that test (``foo::test`` quarantines
    ``foo::test[a]`` and ``foo::test[b]``). An entry that DOES name a
    specific parametrization matches only that one.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_QUARANTINE_PATH = Path(__file__).with_name("quarantine.yaml")

# Registry location relative to the repo root — used to read it out of a
# git revision (the protected base) rather than the candidate checkout.
QUARANTINE_REL_PATH = "scripts/pr_validate/quarantine.yaml"
_GIT_SHOW_TIMEOUT_S = 30


class QuarantineError(Exception):
    """Raised when a present quarantine file is malformed."""


@dataclass(frozen=True)
class QuarantineEntry:
    """One known-flaky test. ``id`` is a pytest node id; the rest is
    human bookkeeping the scorecard can echo back."""

    id: str
    reason: str = ""
    added: str = ""
    issue: str = ""


def load_quarantine(path: Path | None = None) -> list[QuarantineEntry]:
    """Parse the quarantine registry from a file on disk.

    Missing file → ``[]`` (legitimate: no quarantine configured).
    Present but malformed → ``QuarantineError`` (author mistake).

    NOTE for gating callers: prefer ``load_quarantine_from_ref`` so the
    registry is read from the protected base, not the candidate checkout
    (otherwise a PR could quarantine its own failing tests). This
    file-path form is for tests and for reading the shipped default.
    """
    path = path or DEFAULT_QUARANTINE_PATH
    if not path.exists():
        return []
    return _parse_quarantine_text(path.read_text(), source=str(path))


def load_quarantine_from_ref(
    ref: str,
    repo_root: Path,
    rel_path: str = QUARANTINE_REL_PATH,
) -> list[QuarantineEntry]:
    """Read the registry from a git revision (the PROTECTED base) via
    ``git show <ref>:<rel_path>``, NOT from the working tree — so the
    same PR being validated cannot add its own failing tests to the
    allowlist and make them non-blocking (codex #1222).

    Absent at ``ref`` (the revision predates the registry, or any other
    git non-zero) → ``[]``: a base without the file has no quarantine,
    and falling back to empty keeps the gate *stricter*, never looser.
    A malformed file that IS present at ``ref`` → ``QuarantineError``
    (the caller fails safe to empty).
    """
    try:
        proc = subprocess.run(  # noqa: S603
            ["git", "show", f"{ref}:{rel_path}"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=_GIT_SHOW_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # git missing / not a repo / timeout → treat as no quarantine
        # (stricter gate). Never let an infra hiccup loosen the gate.
        return []
    if proc.returncode != 0:
        return []
    return _parse_quarantine_text(proc.stdout, source=f"{ref}:{rel_path}")


def _parse_quarantine_text(text: str, source: str) -> list[QuarantineEntry]:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise QuarantineError(f"{source}: not valid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise QuarantineError(
            f"{source}: must be a mapping with a 'tests' key, got {type(raw).__name__}"
        )
    tests = raw.get("tests", [])
    if tests is None:
        tests = []
    if not isinstance(tests, list):
        raise QuarantineError(
            f"{source}: 'tests' must be a list, got {type(tests).__name__}"
        )

    entries: list[QuarantineEntry] = []
    for i, item in enumerate(tests):
        if not isinstance(item, dict):
            raise QuarantineError(
                f"{source}: test #{i} must be a mapping, got {type(item).__name__}"
            )
        node_id = item.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise QuarantineError(f"{source}: test #{i} needs a non-empty string 'id'")
        entries.append(
            QuarantineEntry(
                id=node_id.strip(),
                reason=str(item.get("reason", "") or "").strip(),
                added=str(item.get("added", "") or "").strip(),
                issue=str(item.get("issue", "") or "").strip(),
            )
        )
    return entries


def node_id_matches(failed_id: str, entry_id: str) -> bool:
    """True iff a failed node id is covered by a quarantine entry.

    Exact match, OR the entry names a whole test and the failure is one
    of its parametrizations (``entry_id`` + ``[...]``).
    """
    if failed_id == entry_id:
        return True
    # A base entry (no bracket of its own) covers every parametrization.
    if "[" not in entry_id and failed_id.startswith(entry_id + "["):
        return True
    return False


def is_quarantined(failed_id: str, entries: Iterable[QuarantineEntry]) -> bool:
    """True iff any entry covers ``failed_id``."""
    return any(node_id_matches(failed_id, e.id) for e in entries)


def partition_failures(
    failed_ids: Iterable[str], entries: Iterable[QuarantineEntry]
) -> tuple[list[str], list[str]]:
    """Split a failure set into ``(blocking, quarantined)``.

    Order is preserved from ``failed_ids``. ``blocking`` is everything
    NOT covered by a quarantine entry — those still fail the gate.
    ``quarantined`` is reported but non-blocking.
    """
    entries = list(entries)
    blocking: list[str] = []
    quarantined: list[str] = []
    for fid in failed_ids:
        if is_quarantined(fid, entries):
            quarantined.append(fid)
        else:
            blocking.append(fid)
    return blocking, quarantined
