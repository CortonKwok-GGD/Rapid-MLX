# SPDX-License-Identifier: Apache-2.0
"""Silence one advisory HuggingFace log line on our download paths.

huggingface_hub surfaces server-sent ``X-HF-Warning`` response headers by
logging them at WARNING level from ``huggingface_hub.utils._http``
(the ``_warn_on_warning_headers`` helper). On an *unauthenticated* request
— the state every naive first-run user is in, since pulling a public MLX
model needs no token — the Hub sends back:

    Warning: You are sending unauthenticated requests to the HF Hub.
    Please set a HF_TOKEN to enable higher rate limits and faster
    downloads.

For rapid-mlx this is pure noise: we front downloads with the R2 mirror,
public repos download fine anonymously, and the lone stray ``Warning:``
line (emitted warn-once per process) reads like an error to a first-time
user — right where the cold-download output was otherwise just cleaned up
(#1259). This is the third first-run papercut from the 0.11 dogfood.

We drop **only** that one advisory, with a narrow, fail-open filter:

* **Fail-open by construction.** It matches on message content, so if the
  Hub rewords or stops sending the header the (reworded) line simply
  reappears — we never hide a real problem. Genuine auth failures (401),
  rate-limit errors (429) and integrity errors are raised as exceptions
  or logged elsewhere; none flow through this one advisory, and none
  match the filter's predicate.
* **Attached to the exact emitting logger.** A filter on a *parent* logger
  is not consulted for a child logger's records during propagation
  (verified empirically), so filtering ``huggingface_hub`` would not work
  — we target ``huggingface_hub.utils._http`` directly. If a future
  huggingface_hub renames that module the filter silently stops matching
  (the advisory returns) rather than crashing — again, fail-open.

Stdlib-only (``logging``) so it stays cheap to import from the otherwise
self-contained download gate. Idempotent: installing more than once is a
no-op.
"""

from __future__ import annotations

import logging

# The logger huggingface_hub's ``_warn_on_warning_headers`` emits from:
# ``logging.get_logger(__name__)`` where ``__name__`` is this module.
_HF_HTTP_LOGGER = "huggingface_hub.utils._http"

# Marker attribute so a re-install can spot our own filter and not add a
# duplicate (the filter class name alone isn't a reliable identity across
# reloads).
_FILTER_TAG = "_rapid_mlx_hf_unauth_filter"


class _DropUnauthenticatedAdvisory(logging.Filter):
    """Drop HF's "unauthenticated requests … set a HF_TOKEN" advisory.

    Returns ``False`` (drop) only for that specific server-sent advisory;
    every other record — including real 401/429 errors, which never carry
    this exact wording — passes untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage().lower()
        except Exception:
            # A record we can't even render is not ours to suppress.
            return True
        return not (
            "unauthenticated" in message
            and ("hf_token" in message or "rate limit" in message)
        )


def silence_hf_unauthenticated_warning() -> None:
    """Install the fail-open advisory filter on the HF http logger.

    Idempotent and cheap — safe to call at the top of every download
    entry point. It is called from :mod:`vllm_mlx._download_gate` (the
    cold-pull size probe) and :mod:`vllm_mlx._mirror` (the R2/HF
    downloader) right before their first Hub request. Because the advisory
    is emitted warn-once per process, installing at those two chokepoints
    covers every flow that acquires a model through the gate or the
    mirror — which, on a cold first run, is all of them.
    """
    logger = logging.getLogger(_HF_HTTP_LOGGER)
    if any(getattr(f, _FILTER_TAG, False) for f in logger.filters):
        return
    filt = _DropUnauthenticatedAdvisory()
    setattr(filt, _FILTER_TAG, True)
    logger.addFilter(filt)
