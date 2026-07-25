# SPDX-License-Identifier: Apache-2.0
"""Shared parser for pytest's short-summary section.

Two steps (``full_unit`` — quarantine partition — and ``flake_tracking``
— rerun classification) need to pull node ids out of pytest's short test
summary. Keeping one parser here means the FAILED/ERROR extraction can't
drift between them (a divergence would silently mis-gate).
"""

from __future__ import annotations


def summary_node_ids(text: str, *labels: str) -> list[str]:
    """Node ids pytest lists under the given short-summary labels.

    A summary line is ``<LABEL> <nodeid>[ - <message>]``; the node id is
    everything between the label and the ``" - "`` message separator.
    Parametrized ids can hold spaces inside ``[...]``, so we split on
    ``" - "`` rather than on whitespace. (A param value literally
    containing ``" - "`` is inherently ambiguous in pytest's own text
    output and is not handled — that is a pytest-format limitation, not
    one we can resolve here.)

    Only lines inside the ``short test summary info`` block count — a
    ``FAILED`` token in a traceback above it is ignored. Defaults to the
    ``FAILED`` label when none is given.
    """
    wanted = labels or ("FAILED",)
    out: list[str] = []
    in_summary = False
    for line in (text or "").splitlines():
        if "short test summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.startswith("="):
            break
        for label in wanted:
            if line.startswith(label + " "):
                node_id = line[len(label) + 1 :].split(" - ", 1)[0].strip()
                if node_id:
                    out.append(node_id)
                break
    return out
