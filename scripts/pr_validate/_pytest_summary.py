# SPDX-License-Identifier: Apache-2.0
"""Shared parser for pytest's short-summary section.

Two steps (``full_unit`` — quarantine partition — and ``flake_tracking``
— rerun classification) need to pull node ids out of pytest's short test
summary. Keeping one parser here means the FAILED/ERROR extraction can't
drift between them (a divergence would silently mis-gate).
"""

from __future__ import annotations

import re

# pytest writes the section header as a full-width separator banner:
# ``==================== short test summary info ====================``.
# Anchoring on that *padded* banner — not a bare ``"short test summary"``
# substring — stops a test from injecting a forged summary block: a test
# that prints ``short test summary`` (or even a fully padded fake banner)
# in its own captured stdout can't be mistaken for the real section,
# because we take the LAST banner and captured output always renders in
# the FAILURES section *above* the genuine summary (codex #1222 r2).
_SUMMARY_BANNER = re.compile(r"^=+\s*short test summary info\s*=+$")


def summary_node_ids(text: str, *labels: str) -> list[str]:
    """Node ids pytest lists under the given short-summary labels.

    A summary line is ``<LABEL> <nodeid>[ - <message>]``; the node id is
    everything between the label and the ``" - "`` message separator.
    Parametrized ids can hold spaces inside ``[...]``, so we split on
    ``" - "`` rather than on whitespace. (A param value literally
    containing ``" - "`` is inherently ambiguous in pytest's own text
    output and is not handled — that is a pytest-format limitation, not
    one we can resolve here.)

    Only lines inside the genuine ``short test summary info`` banner block
    count — a ``FAILED`` token in a traceback (or a forged banner) above
    it is ignored. Defaults to the ``FAILED`` label when none is given.
    """
    wanted = labels or ("FAILED",)
    lines = (text or "").splitlines()

    # Find the LAST real banner. Anything before it — captured stdout,
    # tracebacks, a test-forged banner — is not the authoritative summary.
    start = None
    for i, line in enumerate(lines):
        if _SUMMARY_BANNER.match(line.strip()):
            start = i
    if start is None:
        return []

    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("="):
            break
        for label in wanted:
            if line.startswith(label + " "):
                node_id = line[len(label) + 1 :].split(" - ", 1)[0].strip()
                if node_id:
                    out.append(node_id)
                break
    return out
