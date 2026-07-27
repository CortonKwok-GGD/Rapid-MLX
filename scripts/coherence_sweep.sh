#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Output-coherence sweep (#1247) — run the deterministic golden gate against a
# SET of representative aliases, one at a time. A model-specific regression (the
# 35B RMSNorm incident, #1234) does NOT surface if the release check only runs
# the default 4B/9B alias, so a release / model-path change must sweep the
# aliases it affects — one representative per family, plus any alias the change
# touches.
#
# Each alias is booted on a dedicated port with --no-thinking, run through
# evals/coherence_gate.py (blocking golden answers), then torn down before the
# next. Any alias that fails its blocking golden gate fails the whole sweep.
#
# Usage:
#   bash scripts/coherence_sweep.sh qwen3.5-4b-4bit qwen3.6-35b
#   MODELS="qwen3.5-4b-4bit qwen3.6-35b" bash scripts/coherence_sweep.sh
#
# Exit codes:
#   0 — every alias passed its blocking golden gate
#   1 — at least one alias failed
#   2 — pre-flight refusal (port busy) or a server that never came up

set -euo pipefail

PY="${PY:-python3.12}"
PORT="${PORT:-8402}"
MODELS="${*:-${MODELS:-qwen3.5-4b-4bit}}"
LOG=/tmp/coherence-sweep.log
PIDFILE=/tmp/coherence-sweep.pid

line() { printf '%s\n' "============================================================"; }

if lsof -i ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already in use — pick another with PORT=... or free it." >&2
  exit 2
fi

CURRENT_PID=""
cleanup() {
  if [ -n "$CURRENT_PID" ]; then kill "$CURRENT_PID" 2>/dev/null || true; fi
  if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi
}
trap cleanup EXIT INT TERM

export RAPID_MLX_BASE_URL="http://127.0.0.1:${PORT}/v1"

line
echo "  output-coherence sweep"
echo "  models: $MODELS"
echo "  port:   $PORT"
line

failed=""
for MODEL in $MODELS; do
  line
  echo "  → $MODEL"
  line

  "$PY" -m vllm_mlx.cli serve "$MODEL" --port "$PORT" --no-thinking > "$LOG" 2>&1 &
  CURRENT_PID=$!
  echo "$CURRENT_PID" > "$PIDFILE"

  up=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
    # Bail early if the server process died during load.
    if ! kill -0 "$CURRENT_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if [ "$up" != 1 ]; then
    echo "ERROR: $MODEL server did not come up. Last log lines:" >&2
    tail -20 "$LOG" >&2
    failed="$failed $MODEL(boot)"
  else
    if "$PY" evals/coherence_gate.py; then
      echo "  ✓ $MODEL coherent"
    else
      echo "  ✗ $MODEL FAILED coherence gate" >&2
      failed="$failed $MODEL"
    fi
  fi

  kill "$CURRENT_PID" 2>/dev/null || true
  wait "$CURRENT_PID" 2>/dev/null || true
  CURRENT_PID=""
  rm -f "$PIDFILE"
done

line
if [ -n "$failed" ]; then
  echo "  SWEEP FAILED —$failed"
  line
  exit 1
fi
echo "  SWEEP PASSED — all aliases coherent"
line
