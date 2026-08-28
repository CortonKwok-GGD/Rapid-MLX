# SPDX-License-Identifier: Apache-2.0
"""Tests for ``CLDescriptionQualityStep`` — the PR description-quality gate.

Issue #2510: an UNEDITED ``.github/PULL_REQUEST_TEMPLATE.md`` (with the
``[x]`` checklist boxes ticked) was a FALSE green — the template's
``## Why`` / ``## Scope`` headings matched the rationale-signal regex even
though every word of real prose sat inside ``<!-- … -->`` HTML comments.
The fix: strip HTML comments before scoring and require substantive prose
under the ``## Why`` / ``## Scope`` contract headings.

These are pure-CPU tests: no MLX, no network. The raw template file on disk
is the fixture for the false-green regression test.
"""

from __future__ import annotations

from pathlib import Path

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.cl_description_quality import (
    CLDescriptionQualityStep,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE = _REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"


def _ctx(body: str, title: str = "feat: describe a scoped real change") -> Context:
    """Context shell with a good title and the given body — everything the
    step reads apart from ``ctx.pr_title`` / ``ctx.pr_body`` is defaulted."""
    ctx = Context(pr_number=999, repo="x/y")
    ctx.pr_title = title
    ctx.pr_body = body
    return ctx


def _run(body: str, title: str = "feat: describe a scoped real change"):
    return CLDescriptionQualityStep().run(_ctx(body, title=title))


def test_raw_template_fails():
    """The core #2510 regression: the RAW, UNEDITED template (prose all
    inside ``<!-- -->`` comments, boxes ticked) must FAIL — its ``## Why`` /
    ``## Scope`` sections are empty once comments are stripped."""
    body = _TEMPLATE.read_text()
    assert "## Why" in body, "test fixture sanity: template has a ## Why section"
    result = _run(body)
    assert result.status == "fail"
    assert "empty" in result.summary


def test_why_section_empty_fails_even_with_box_ticked():
    """A body that left ``## Why`` as an empty comment-fence fails even when
    a later ``## Checklist`` box is ``[x]`` — the template skeleton alone is
    not a rationale."""
    body = (
        "## Why\n\n"
        "<!-- fill me in -->\n\n"
        "## Scope\n"
        "rewrote the cache invalidation path.\n\n"
        "## Checklist\n"
        "- [x] lint\n"
    )
    result = _run(body)
    assert result.status == "fail"
    assert "## Why" in result.summary or "Why" in result.summary


def test_scope_section_empty_fails():
    """The ``## Scope`` heading is also a contract heading — an empty (or
    comment-only) Scope fails alongside a filled Why."""
    body = (
        "## Why\n"
        "fixes #123 (parser drops tool_call deltas)\n\n"
        "## Scope\n"
        "<!-- what changed goes here -->\n"
    )
    result = _run(body)
    assert result.status == "fail"
    assert "## Scope" in result.summary


def test_real_pr_with_why_and_scope_passes():
    """A genuine PR with real prose under both contract headings passes."""
    body = (
        "## Why\n"
        "fixes #123 (parser drops tool_call deltas)\n\n"
        "## Scope\n"
        "- rewrote the grammar edge for nested tool calls.\n\n"
        "## Non-goals\n- no server change.\n\n"
        "## Verification\n- [x] pytest tests/\n"
    )
    result = _run(body)
    assert result.status == "pass"


def test_comment_only_prose_does_not_satisfy_rationale():
    """A body whose ONLY content is an HTML comment carries no rationale —
    it must fail as effectively empty (issue #2510)."""
    body = "<!-- This reverts a bad commit because it caused a crash -->"
    result = _run(body, title="fix: revert a bad commit to stop a crash")
    assert result.status == "fail"
    assert "empty" in result.summary


def test_why_prose_without_scope_heading_passes():
    """A partial-template PR (Why filled, no Scope heading) still passes —
    we only gate headings that are actually present."""
    body = "## Why\nfixes #123 (parser drops tool_call deltas)\n"
    result = _run(body)
    assert result.status == "pass"


def test_author_section_empty_does_not_false_fail():
    """Compatibility with the OPTIONAL ``## Author`` section (PR #2532): a
    legitimate PR with real Why/Scope prose but an EMPTY Author field still
    passes — only Why/Scope prose is gated."""
    body = (
        "## Why\n"
        "restores N% TPS regression on model M\n\n"
        "## Author\n"
        "(empty)\n\n"
        "## Scope\n"
        "- rebuilt the attention kernel.\n\n"
        "## Checklist\n- [x] lint\n"
    )
    result = _run(body)
    assert result.status == "pass"


def test_reason_why_line_without_heading_passes():
    """The lenient non-template path is preserved: an inline ``**Why:**``
    line (no ``## Why`` heading) is still a rationale signal."""
    body = "**Why:** the flake only shows under residency load.\n"
    result = _run(body)
    assert result.status == "pass"


def test_no_rationale_fails():
    """A body with real words but no rationale signal (no why heading, no
    issue link, no because, no Why: line) fails."""
    body = "This change touches the scheduler module and is small.\n"
    result = _run(body)
    assert result.status == "fail"
    assert "no rationale" in result.summary
