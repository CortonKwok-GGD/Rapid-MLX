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
  * Matching is by EXACT pytest node id by default (codex #1222 r14).
    An entry may opt into parameter-family matching with ``family: true``,
    which makes a base (no-bracket) id also cover every parametrization
    (``foo::test`` then quarantines ``foo::test[a]`` and ``foo::test[b]``).
    Exact-by-default is deliberate: a PR that adds a NEW, deterministically
    failing parametrization to a listed test must not have it silently
    waived — family coverage is a reviewed decision, made once at the base.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
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
    human bookkeeping the scorecard can echo back.

    ``family`` opts a base (no-bracket) entry into matching EVERY
    parametrization of the test, not just the exact id. It defaults False
    so that adding a new, deterministically-failing parametrization to a
    quarantined test is never silently waived without an explicit,
    base-reviewed decision (codex #1222 r14)."""

    id: str
    reason: str = ""
    added: str = ""
    issue: str = ""
    family: bool = False


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
    # Attempt the read DIRECTLY — never gate on ``path.exists()`` first.
    # ``Path.exists()`` re-raises a ``PermissionError`` when a parent
    # directory is inaccessible (it only swallows ENOENT/ENOTDIR/etc.), so
    # a pre-check outside the handler could escape the documented
    # QuarantineError fail-safe with a raw OSError (codex #1222 r24). Only a
    # CONFIRMED-missing file (``FileNotFoundError``) is the legitimate
    # "absent → empty" case; every other OSError is a present-but-unusable
    # registry.
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # No registry configured — legitimate empty quarantine.
        return []
    except UnicodeDecodeError as e:
        # A non-UTF-8 registry is malformed, not "absent" — surface it as
        # QuarantineError so the gating caller fails safe to empty rather
        # than crashing on an unhandled ValueError (codex #1222 r5).
        raise QuarantineError(f"{path}: not valid UTF-8: {e}") from e
    except OSError as e:
        # A present-but-unreadable file (permission denied, an inaccessible
        # parent, an I/O error) is a read failure, not "absent" — the
        # docstring promises QuarantineError for a present-but-unusable
        # registry, so a caller catching only that exception doesn't crash on
        # a raw OSError (codex #1222 r18/r24).
        raise QuarantineError(f"{path}: could not be read: {e}") from e
    return _parse_quarantine_text(text, source=str(path))


def load_quarantine_from_ref(
    ref: str,
    repo_root: Path,
    rel_path: str = QUARANTINE_REL_PATH,
) -> list[QuarantineEntry]:
    """Read the registry from a git revision (the PROTECTED base) via
    ``git show <ref>:<rel_path>``, NOT from the working tree — so the
    same PR being validated cannot add its own failing tests to the
    allowlist and make them non-blocking (codex #1222).

    Note the outcomes carefully (codex #1222 r10): only a CONFIRMED
    missing path → ``[]`` (the base predates the registry; empty keeps the
    gate stricter, never looser). Every other failure — git absent, a
    timeout, a bad ref, a non-UTF-8 or malformed blob — raises
    ``QuarantineError`` so the gating caller falls back to empty *and
    reports it loudly*, instead of a silent empty that's indistinguishable
    from "nothing is quarantined". Never let an infra hiccup pass unseen.

    ``--no-replace-objects`` is REQUIRED, not cosmetic (codex #1222 r12):
    an immutable base SHA pins the *object*, but ``git show`` otherwise
    honors local replacement refs, so a candidate test that ran earlier in
    the pipeline could ``git replace <base_sha> <attacker-commit>`` and make
    ``git show <base_sha>:…`` serve its own allowlist despite the pinned
    SHA. With replacement disabled, ``<40-hex-sha>:<path>`` resolves the
    true content-addressed blob — the only git indirection that could remap
    it is closed (grafts/alternates can't override an existing object's
    content; a mutable branch ref is no longer used per r10).
    """
    try:
        proc = subprocess.run(  # noqa: S603
            # --no-replace-objects: ignore any refs/replace/* a candidate
            # test may have planted; read the true base blob (codex r12).
            ["git", "--no-replace-objects", "show", f"{ref}:{rel_path}"],  # noqa: S607
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(repo_root),
            timeout=_GIT_SHOW_TIMEOUT_S,
            # Pin the locale so the "missing path" stderr fragments matched
            # below are git's stable C-locale English, not a translated
            # message under a non-English LANG — otherwise a legitimately
            # absent registry would misclassify as a hard error (codex #1222
            # r25). LC_ALL wins over LANG/LC_MESSAGES; set both for belt-and-
            # braces.
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except FileNotFoundError as e:
        # git not installed / not on PATH — infra failure, not "no registry".
        raise QuarantineError(f"git unavailable to read {ref}:{rel_path}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise QuarantineError(
            f"git show timed out after {_GIT_SHOW_TIMEOUT_S}s reading {ref}:{rel_path}"
        ) from e
    except (OSError, subprocess.SubprocessError) as e:
        raise QuarantineError(f"git show failed for {ref}:{rel_path}: {e}") from e
    except UnicodeDecodeError as e:
        # A non-UTF-8 blob at ref is malformed → QuarantineError so the
        # gating caller fails safe to empty, not an unhandled ValueError
        # (codex #1222 r5).
        raise QuarantineError(f"{ref}:{rel_path}: not valid UTF-8: {e}") from e
    if proc.returncode != 0:
        # git exits 128 for BOTH "path absent at this ref" (legitimate) and
        # real errors (bad ref, not a repo). Only a confirmed missing PATH
        # is an empty registry; anything else is surfaced.
        stderr = (proc.stderr or "").lower()
        if "does not exist in" in stderr or "exists on disk, but not in" in stderr:
            return []
        raise QuarantineError(
            f"git show failed for {ref}:{rel_path} (exit {proc.returncode}): "
            f"{(proc.stderr or '').strip()}"
        )
    return _parse_quarantine_text(proc.stdout, source=f"{ref}:{rel_path}")


def _audit_string(
    value: object, *, source: str, index: int, node_id: str, field: str
) -> str:
    """Validate a free-text audit field (``reason`` / ``issue``).

    A MISSING field (``None``) is the empty string. A present but non-string
    value — e.g. ``reason: [foo, bar]`` — is a schema violation, NOT
    something to silently ``str(...)``-coerce into a bogus ``"['foo', …]"``
    rationale that then satisfies the mandatory-field check (codex #1222
    r15). Returns the stripped string; the caller enforces non-emptiness for
    a mandatory field."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise QuarantineError(
            f"{source}: test #{index} ('{node_id}') '{field}' must be a "
            f"string, got {type(value).__name__}"
        )
    return value.strip()


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that REJECTS duplicate mapping keys (codex #1222 r21).

    PyYAML's default silently keeps the LAST value for a duplicated key, so a
    quarantine entry like ``{id: real, id: attacker}`` — or a duplicated
    top-level ``tests:`` / a duplicated ``family:`` — would let a reviewer
    approve one value while another invisibly overrides it, exactly the kind
    of audit-trail swap the mandatory ``reason``/``added`` fields exist to
    prevent. Rejecting duplicates keeps the YAML honest BEFORE schema
    validation runs. Raises ``ConstructorError`` (a ``yaml.YAMLError``), so the
    caller's existing YAML-error handling wraps it as a ``QuarantineError``.
    Checks every mapping level (root + each test dict) because the loader
    routes every mapping node through ``construct_mapping``.
    """

    def construct_mapping(self, node, deep=False):  # noqa: ANN001, ANN201, FBT002
        seen: set = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            # A complex (list / mapping) YAML key is unhashable, so ``key in
            # seen`` would raise a raw ``TypeError`` instead of the documented
            # ``QuarantineError`` (codex #1222 r22). Convert it to a
            # ``ConstructorError`` (a ``yaml.YAMLError``) so the caller's
            # error handling wraps it — the same class of complaint pyyaml
            # itself raises for an unhashable key, just surfaced here first.
            try:
                is_dup = key in seen
            except TypeError as e:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"unacceptable (unhashable) key: {e}",
                    key_node.start_mark,
                ) from e
            if is_dup:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _parse_quarantine_text(text: str, source: str) -> list[QuarantineEntry]:
    try:
        # Strict loader: duplicate keys raise instead of silently last-wins
        # (codex #1222 r21). Loader is a SafeLoader subclass, so this is not
        # an unsafe ``yaml.load``.
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)  # noqa: S506
    except yaml.YAMLError as e:
        raise QuarantineError(f"{source}: not valid YAML: {e}") from e

    # Only an ABSENT document (empty file / all comments → None) is an
    # empty registry. A falsey-but-present root such as ``[]`` / ``false``
    # / ``0`` is a schema violation and must surface, not be swallowed by
    # ``or {}`` (codex #1222 r2).
    if raw is None:
        raw = {}
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
        node_id = node_id.strip()
        # ``reason`` and ``added`` are MANDATORY audit data — every entry is
        # a hole in the gate, and an undocumented one (no rationale, no
        # date) is how a permanent silent waiver creeps in. Enforce them so
        # a bare ``{id: …}`` can't slip a test onto the allowlist without a
        # paper trail (codex #1222 r10). ``issue`` stays optional (not every
        # flake has a tracker link yet). ``reason``/``issue`` must be actual
        # strings, not ``str(...)``-coerced from a list/mapping (codex #1222
        # r15); ``added`` is checked below by strict date parsing.
        reason = _audit_string(
            item.get("reason"), source=source, index=i, node_id=node_id, field="reason"
        )
        added = str(item.get("added", "") or "").strip()
        if not reason:
            raise QuarantineError(
                f"{source}: test #{i} ('{node_id}') needs a non-empty 'reason' "
                f"— document WHY it is flaky"
            )
        if not added:
            raise QuarantineError(
                f"{source}: test #{i} ('{node_id}') needs a non-empty 'added' "
                f"date (YYYY-MM-DD)"
            )
        # ``added`` claims to be a YYYY-MM-DD date, so ENFORCE it — a
        # non-empty-but-garbage value ("soon", "2026-13-40") is exactly the
        # silent audit rot the mandatory field exists to prevent. Require a
        # value that round-trips through ``date.fromisoformat`` to canonical
        # YYYY-MM-DD (rejects "2026-7-5", timestamps, and out-of-range
        # dates) (codex #1222 r13).
        try:
            parsed = date.fromisoformat(added)
        except ValueError as e:
            raise QuarantineError(
                f"{source}: test #{i} ('{node_id}') 'added' must be a "
                f"YYYY-MM-DD date, got {added!r}: {e}"
            ) from e
        if parsed.isoformat() != added:
            raise QuarantineError(
                f"{source}: test #{i} ('{node_id}') 'added' must be canonical "
                f"YYYY-MM-DD, got {added!r} (expected {parsed.isoformat()})"
            )
        # ``family`` is an explicit, reviewed opt-in — a strict boolean so a
        # stray string can't be truthy-by-accident and silently widen the
        # allowlist to every parametrization (codex #1222 r14).
        family = item.get("family", False)
        if not isinstance(family, bool):
            raise QuarantineError(
                f"{source}: test #{i} ('{node_id}') 'family' must be a boolean, "
                f"got {type(family).__name__}"
            )
        entries.append(
            QuarantineEntry(
                id=node_id,
                reason=reason,
                added=added,
                issue=_audit_string(
                    item.get("issue"),
                    source=source,
                    index=i,
                    node_id=node_id,
                    field="issue",
                ),
                family=family,
            )
        )
    return entries


def _has_param_suffix(node_id: str) -> bool:
    """True iff the node id's FINAL component carries a ``[...]`` parameter
    suffix (``tests/x.py::TestC::test_y[p]``).

    Only the last ``::`` component can hold a parametrization bracket; a ``[``
    earlier in the path is a directory or class name (e.g. a test living under
    ``tests/models/[legacy]/test_x.py::test_y``). Searching the WHOLE id for
    ``[`` wrongly disabled family matching for such tests (codex #1222 r27)."""
    return "[" in node_id.rsplit("::", 1)[-1]


def node_id_matches(failed_id: str, entry_id: str, *, family: bool = False) -> bool:
    """True iff a failed node id is covered by a quarantine entry.

    EXACT match by default. Only when ``family`` is set does a base
    (unparametrized) entry also cover every parametrization (``entry_id`` +
    ``[...]``) — exact-by-default keeps a newly-added, deterministically
    failing parametrization from being silently waived (codex #1222 r14).
    """
    if failed_id == entry_id:
        return True
    # Family coverage is OPT-IN and only meaningful for a base (unparametrized)
    # entry — gated on the FINAL component's suffix, not any ``[`` in the path,
    # so a test under a bracket-named directory can still use it (codex r27).
    if (
        family
        and not _has_param_suffix(entry_id)
        and failed_id.startswith(entry_id + "[")
    ):
        return True
    return False


def is_quarantined(failed_id: str, entries: Iterable[QuarantineEntry]) -> bool:
    """True iff any entry covers ``failed_id``."""
    return any(node_id_matches(failed_id, e.id, family=e.family) for e in entries)


def effective_quarantine(
    base_entries: Iterable[QuarantineEntry],
    candidate_entries: Iterable[QuarantineEntry],
) -> list[QuarantineEntry]:
    """The registry the gate enforces: base ∩ candidate (codex #1222 r21).

    The gate reads quarantine coverage from the PROTECTED base so a PR cannot
    quarantine its OWN failing tests. But base-only silently ignored the other
    direction — a PR that REMOVES an entry (de-quarantine): the removed test's
    failure would still be waived off the stale base list, so a de-quarantine
    PR could merge while that test is still red, reintroducing a regression
    alongside the removal.

    Intersecting base with the candidate registry fixes that while keeping the
    self-quarantine protection intact:
      * an id is effective ONLY if BOTH base and candidate list it — a
        candidate REMOVAL drops it, so a still-red de-quarantined test blocks;
      * a candidate ADDITION (id absent from base) is ignored — a PR still
        cannot waive its own failure;
      * breadth is the NARROWER of the two (``family`` only when BOTH set it),
        so a candidate may TIGHTEN (family→exact, or drop a param) but can
        never WIDEN (exact→family) to waive more than the base approved.

    The result is ALWAYS a subset of ``base_entries`` with breadth no wider,
    so even a maximally-hostile candidate registry can only make the gate
    STRICTER — never waive anything the protected base didn't already approve.
    Base audit metadata (reason / added / issue) is preserved; only ``family``
    is narrowed.
    """
    # Narrowest candidate breadth per id — a duplicate candidate id can only
    # tighten (``family`` stays True only if EVERY candidate copy is family).
    cand_family: dict[str, bool] = {}
    for c in candidate_entries:
        cand_family[c.id] = (
            c.family if c.id not in cand_family else (cand_family[c.id] and c.family)
        )
    effective: list[QuarantineEntry] = []
    for b in base_entries:
        if b.id not in cand_family:
            continue  # removed / never listed by the candidate → drop
        effective.append(replace(b, family=b.family and cand_family[b.id]))
    return effective


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
