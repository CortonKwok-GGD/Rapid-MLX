#!/bin/bash
# OpenHands Docker E2E integration harness against a running `rapid-mlx serve`.
#
# What it proves:
#   1. OpenHands' CodeActAgent can connect to rapid-mlx's OpenAI-compatible
#      endpoint via LiteLLM's ``openai/<alias>`` provider prefix.
#   2. OpenHands' text-action edit format (``<execute_bash>...``,
#      ``<execute_ipython>...``, and its file-write markdown blocks —
#      parsed by OpenHands itself, NOT via OpenAI tool_calls) actually
#      rewrites a local file the way we asked ("fix the bug in add.py —
#      add, not subtract").
#
# Why the correctness signal is NOT OpenAI tool_calls: OpenHands'
# native wire (openhands.yaml capabilities.function_calling: false) is a
# text-action format. The CodeActAgent parses the model's plaintext
# reply, extracts actions, and applies file edits through its sandbox
# runtime. So the pass gate is whether the executed ``add()`` really
# returns ``a + b`` after openhands exits — not a simple grep, which
# would flake if the model reformats the body, and not a tool_call
# assertion, which the wire doesn't emit.
#
# This harness is the sibling of ``test_aider.sh``. Same arg parsing,
# same correctness taxonomy (AST BinOp(op=Add) + runtime add(2,3) == 5),
# same exit-code taxonomy — with docker-daemon + docker-in-docker
# sock-passthrough layered on for OpenHands' sandbox-runtime container.
#
# Usage:
#   test_openhands.sh --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>]
#
# ``--base-url`` takes the full ``http[s]://host:port/v1`` URL and is the
# preferred form — it lets the Python wrapper pass whatever URL the
# ``rapid_mlx_server`` fixture is actually pointed at. Regardless of the
# host in ``--base-url``, the URL passed into the OpenHands container is
# rewritten to ``http://host.docker.internal:PORT/v1`` — from inside the
# container ``localhost`` refers to the container itself, not the host
# where rapid-mlx is listening.
#
# Exit codes (aligned with ``test_aider.sh``):
#   0  — OpenHands completed and add(2, 3) == 5 in the rewritten file
#   1  — arg parse / setup error (also: docker daemon unreachable,
#        ``timeout`` / ``gtimeout`` missing, rapid-mlx serve unreachable)
#   2  — OpenHands runtime exited non-zero (agent crashed, LLM refused,
#        transport blip; runtime container failed to boot)
#   3  — OpenHands ran but the file wasn't corrected (edit didn't apply;
#        agent hit ``-i`` iteration cap without solving; LLM refused;
#        wrong operator, etc.)
#   4  — timeout

# ``set -euo pipefail`` so an unchecked setup step (``mktemp``,
# ``docker info``, ``mkdir``) aborts the harness instead of running
# OpenHands against a half-configured scratch dir. Mirrors the same
# lesson-learned gate baked into ``test_aider.sh`` (round-2 finding #4).
set -euo pipefail

# ---------------------------------------------------------------------------
# Pinned image tags — see PR body for the version-selection rationale.
#
# ``ghcr.io/all-hands-ai/openhands`` moved on from ``docker.all-hands.dev``
# (the old registry's DNS no longer resolves — see the PR body for the
# transition timeline). We pin to a specific ``0.9.0`` tag (multi-arch
# manifest — both linux/arm64 and linux/amd64) so the harness can't
# silently drift when a new OpenHands release lands upstream.
#
# The runtime container tag is coupled to the app tag: OpenHands
# derives its own hash-tagged runtime image from this baseline, so the
# baseline must match the OpenHands major version. Currently:
#   od_v0.9.0_image_nikolaik___python-nodejs_tag_python3.11-nodejs22
# ---------------------------------------------------------------------------
OPENHANDS_IMAGE="ghcr.io/all-hands-ai/openhands:0.9.0"
OPENHANDS_RUNTIME_IMAGE="ghcr.io/all-hands-ai/runtime:od_v0.9.0_image_nikolaik___python-nodejs_tag_python3.11-nodejs22"

TIMEOUT=600
MAX_ITERATIONS=10
MODEL=""
PORT=""
BASE_URL=""
VERBOSE=0

usage() {
    echo "Usage: $0 --model <alias> (--base-url <url> | --port <port>) [--timeout <secs>] [--max-iterations <n>] [-v]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || { [ -z "$BASE_URL" ] && [ -z "$PORT" ]; }; then
    usage
fi

# Derive BASE_URL from --port only if --base-url wasn't given (back-compat
# for the standalone invocation shape kept for local docs/dev). --base-url
# wins so a Python wrapper that always passes the full URL is authoritative.
if [ -z "$BASE_URL" ]; then
    BASE_URL="http://127.0.0.1:${PORT}/v1"
fi

# Extract host + port from --base-url so we can decide whether to rewrite
# the host for the inside-container view. Codex #1048 round-1 finding #1
# (BLOCKING): the previous unconditional rewrite to ``host.docker.internal``
# silently broke a CI shard pointing at a genuine remote-serve node
# (``--base-url http://remote-host:8802/v1``) — the container would have
# hit the M3 Ultra host instead of the intended remote server. Now we
# rewrite ONLY the local-loopback aliases (``localhost``, ``127.0.0.1``,
# ``0.0.0.0``, ``::1``), and preserve any other host so remote fixtures
# still work.
CONTAINER_HOST="$(printf '%s' "$BASE_URL" | sed -E 's#https?://([^:/]+):([0-9]+)/?.*#\1#')"
CONTAINER_PORT="$(printf '%s' "$BASE_URL" | sed -E 's#https?://([^:/]+):([0-9]+)/?.*#\2#')"
if ! [[ "$CONTAINER_PORT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not extract port from --base-url=$BASE_URL" >&2
    exit 1
fi
if [ -z "$CONTAINER_HOST" ]; then
    echo "ERROR: could not extract host from --base-url=$BASE_URL" >&2
    exit 1
fi
case "$CONTAINER_HOST" in
    localhost|127.0.0.1|0.0.0.0|::1)
        CONTAINER_HOST="host.docker.internal"
        ;;
esac
CONTAINER_BASE_URL="http://${CONTAINER_HOST}:${CONTAINER_PORT}/v1"

# ``python3`` powers the correctness check below (AST parse + runtime
# ``add(2, 3) == 5`` assertion). Fail early with a clear message so a
# broken system Python doesn't get diagnosed as an OpenHands failure.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH — required for correctness check" >&2
    exit 1
fi

# Docker daemon must be reachable BEFORE we spend 5-10 minutes staging
# the sandbox runtime. ``docker info`` is the canonical "server reachable"
# probe (``docker version`` returns the client version even when the
# daemon socket is dead).
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable — start Docker Desktop / dockerd first" >&2
    exit 1
fi

# Pick a timeout wrapper — same requirement as ``test_aider.sh``: we need
# the whole exec'd tree killed on timeout, and BSD's built-in no-timeout
# fallback would leak a running ``docker run`` past harness exit.
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(gtimeout --preserve-status --kill-after=15 "$TIMEOUT")
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout --preserve-status --kill-after=15 "$TIMEOUT")
else
    echo "ERROR: neither 'timeout' nor 'gtimeout' available — install " >&2
    echo "       coreutils (macOS: 'brew install coreutils') before running." >&2
    exit 1
fi

# Scratch state — HOME override so the OpenHands agent drops its config /
# cache into a throw-away tree we can nuke on exit. We NEVER touch the
# operator's real ``~/.openhands*`` state. ``mktemp -d`` both for atomic
# creation and to avoid predictable-path rm -rf races.
WORKDIR="$(mktemp -d -t openhands-test-work.XXXXXX)"
OPENHANDS_STATE="$(mktemp -d -t openhands-test-state.XXXXXX)"
mkdir -p "$OPENHANDS_STATE/.openhands"

# Uniqueness for the docker container name — timestamp + PID + $$ nesting
# guard so parallel harness invocations (or a stale prior run's zombie)
# can't collide on the name. We also keep the value in a variable so the
# cleanup trap can force-remove even if the run was aborted before
# ``docker run`` returned.
CONTAINER_NAME="openhands-test-$$-$(date +%s)"

cleanup() {
    local rc=$?
    # Best-effort: ``docker rm -f`` succeeds silently if the container
    # already exited via ``--rm``; suppress stderr so a race with the
    # auto-remove doesn't spam an error into the harness log. We do NOT
    # ``docker system prune`` — that would nuke the operator's other
    # containers (violates G11 + the operator-lane guardrail).
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    if [ "$VERBOSE" -eq 0 ]; then
        rm -rf "$WORKDIR" "$OPENHANDS_STATE" 2>/dev/null || true
    else
        echo "VERBOSE: preserved WORKDIR=$WORKDIR OPENHANDS_STATE=$OPENHANDS_STATE" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

# Toy file with an obvious bug — identical to ``test_aider.sh`` so the
# two harnesses agree on the pass gate and a divergence between agents
# is obviously the agent, not the fixture.
cat > "$WORKDIR/add.py" <<'PYEOF'
def add(a, b):
    return a - b  # BUG
PYEOF

# Sanity: is the server actually up on the HOST-side URL? A quick
# /v1/models probe with a 5 s timeout catches "operator forgot to boot
# serve" instantly instead of eating the full harness timeout. We probe
# BASE_URL (host-visible), not CONTAINER_BASE_URL (only meaningful
# inside the container).
if ! curl -sS -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: rapid-mlx server not reachable at $BASE_URL" >&2
    exit 1
fi

# LiteLLM (which OpenHands uses under the hood) needs the ``openai/``
# prefix to route through the OpenAI-compat chat completions path against
# our custom base URL — without it LiteLLM tries to pick a provider
# from the alias string and fails on non-canonical rapid-mlx aliases.
LITELLM_MODEL="openai/${MODEL}"

# Ensure the runtime base image is present locally — if the pull fails
# we want that surfaced as a setup error (exit 1), not misdiagnosed as
# an OpenHands runtime error (exit 2). ``docker pull`` with ``--quiet``
# is idempotent and prints only the digest on success; with ``2>&1`` we
# fold pull chatter (progress lines) into the harness log for diagnostics.
for img in "$OPENHANDS_IMAGE" "$OPENHANDS_RUNTIME_IMAGE"; do
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "[test_openhands.sh] pulling missing image: $img"
        if ! docker pull "$img" 2>&1 | tail -5 >&2; then
            echo "ERROR: docker pull failed for $img" >&2
            exit 1
        fi
    fi
done

echo "[test_openhands.sh] model=$MODEL host-base-url=$BASE_URL container-base-url=$CONTAINER_BASE_URL"
echo "[test_openhands.sh] litellm-model=$LITELLM_MODEL timeout=${TIMEOUT}s max-iter=${MAX_ITERATIONS}"
echo "[test_openhands.sh] openhands-image=$OPENHANDS_IMAGE"
echo "[test_openhands.sh] runtime-image=$OPENHANDS_RUNTIME_IMAGE"
echo "[test_openhands.sh] scratch workdir=$WORKDIR openhands-state=$OPENHANDS_STATE"
echo "[test_openhands.sh] container-name=$CONTAINER_NAME"
echo "[test_openhands.sh] BEFORE add.py:"
cat "$WORKDIR/add.py"
echo "--------"

# Run OpenHands one-shot with ``python -m openhands.core.main -t "..."``.
# Key env-var wiring:
#   SANDBOX_CONTAINER_IMAGE  — the pre-built runtime; when this image
#                              string contains the ``ghcr.io/all-hands-ai
#                              /runtime`` repo prefix, OpenHands' builder
#                              short-circuits its Dockerfile-from-scratch
#                              path and reuses / derives from this image
#                              instead of pulling ``nikolaik`` from Docker
#                              Hub. Saves ~5 minutes of apt-install on a
#                              cold M3 Ultra.
#   SANDBOX_USER_ID          — matches the host UID so files created by
#                              the sandbox in our bind-mounted WORKDIR
#                              come back owned by the invoking user, not
#                              root. Without this, cleanup rm -rf fails
#                              on macOS when the docker VM's overlay
#                              driver leaves root-owned artifacts.
#   SANDBOX_TIMEOUT          — per-command timeout inside the sandbox;
#                              default 120s is tight for slow local
#                              inference, we bump to 180s.
#   WORKSPACE_MOUNT_PATH     — the ABSOLUTE HOST PATH that OpenHands will
#                              bind-mount into the sandbox as its
#                              workspace. This is critical — without it
#                              OpenHands uses a default path that won't
#                              exist / won't match our WORKDIR.
#   WORKSPACE_BASE           — the in-container path the sandbox sees as
#                              the workspace. Matches the openhands
#                              image default (``/opt/workspace_base``).
#   LLM_BASE_URL/MODEL/API_KEY — LiteLLM wiring for the rapid-mlx endpoint.
# Docker flags:
#   --add-host host.docker.internal:host-gateway  — required on Linux
#                              (macOS Docker Desktop injects this
#                              automatically, but Linux CI needs the
#                              explicit --add-host). Keeping it here
#                              works on both platforms and is idempotent.
#   -v /var/run/docker.sock:/var/run/docker.sock — docker-in-docker sock
#                              passthrough so OpenHands can spawn its
#                              sandbox runtime container.
#   --pull=missing             — only pull if not already cached; the
#                              two images ARE cached (we ensured above),
#                              so this is a no-op belt-and-braces guard.
LOG="$WORKDIR/openhands.log"
STATUS=0

# We disable ``set -e`` for the docker run so a non-zero exit (timeout,
# agent crash, LLM refusal, transport blip) is captured into $STATUS
# instead of aborting the harness before we can print the diagnostic tail.
set +e
"${TIMEOUT_CMD[@]}" \
docker run \
    --rm \
    --name "$CONTAINER_NAME" \
    -e "SANDBOX_CONTAINER_IMAGE=$OPENHANDS_RUNTIME_IMAGE" \
    -e "SANDBOX_USER_ID=$(id -u)" \
    -e "SANDBOX_TIMEOUT=180" \
    -e "WORKSPACE_MOUNT_PATH=$WORKDIR" \
    -e "WORKSPACE_BASE=/opt/workspace_base" \
    -e "LLM_BASE_URL=$CONTAINER_BASE_URL" \
    -e "LLM_MODEL=$LITELLM_MODEL" \
    -e "LLM_API_KEY=rapidmlx" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$WORKDIR:/opt/workspace_base" \
    -v "$OPENHANDS_STATE/.openhands:/home/openhands/.openhands" \
    --add-host host.docker.internal:host-gateway \
    --pull=missing \
    "$OPENHANDS_IMAGE" \
    python -m openhands.core.main \
        -i "$MAX_ITERATIONS" \
        -t "The file add.py in the current workspace has a bug: it returns a - b when it should return a + b. Open add.py, change the '-' operator to '+' in the return statement, and save the file. Do not modify anything else. Once the file is saved, stop." \
    >"$LOG" 2>&1
STATUS=$?
set -e

# Detect timeout: 124 = coreutils timeout (SIGTERM path); 137 = --kill-after
# escalation (SIGKILL); 143 = SIGTERM. All three mean "we killed it, not
# OpenHands exiting cleanly with a non-zero code."
if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 137 ] || [ "$STATUS" -eq 143 ]; then
    echo "[test_openhands.sh] TIMEOUT after ${TIMEOUT}s" >&2
    echo "--- last 60 lines of openhands log ---" >&2
    tail -60 "$LOG" >&2 || true
    exit 4
fi

echo "[test_openhands.sh] openhands exit=$STATUS"
echo "--- last 60 lines of openhands log ---"
tail -60 "$LOG" || true
echo "--------"
echo "[test_openhands.sh] AFTER add.py:"
cat "$WORKDIR/add.py"
echo "--------"

if [ "$STATUS" -ne 0 ]; then
    echo "[test_openhands.sh] FAIL: openhands exited $STATUS" >&2
    exit 2
fi

# Correctness check — Codex #1048 round-1 findings #2 and #3 (BLOCKING):
# the previous single-pair ``add(2, 3) == 5`` gate + "any ast.Add anywhere
# in add()" AST match let ``return a - b + 6`` and ``return (a - b) + 6``
# fake a pass while preserving the original subtraction bug. Fix: sweep a
# family of input pairs that no polynomial masquerade can satisfy
# simultaneously, and drop the weak AST match — no single ``a + b + k``
# / ``a - b + k`` / hard-coded-return cheat can pass ALL five checks,
# because they pin the linear combination to slope-1 on both variables
# with zero intercept. Runtime evaluation is safe because the scratch
# file was written by us and only mutated in place by OpenHands' sandbox
# edits — no arbitrary source enters the tree.
if ! python3 - "$WORKDIR" <<'PYEOF'
import importlib.util
import sys

workdir = sys.argv[1]
target = f"{workdir}/add.py"

# Load the module.
spec = importlib.util.spec_from_file_location("_openhands_test_add", target)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as exc:  # noqa: BLE001 — diagnose whatever it was
    print(f"[correctness] MODULE-LOAD-ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

if not hasattr(mod, "add") or not callable(mod.add):
    print("[correctness] MODULE-SHAPE-ERROR: add() missing or not callable",
          file=sys.stderr)
    sys.exit(1)

# Five checks that jointly force ``add(a, b) == a + b`` — pin slope on
# both variables to 1 and intercept to 0, so no ``a + k``, ``a - b + k``,
# ``a + b + k`` (k ≠ 0), ``return CONST`` cheat can satisfy them all:
#
#     (2, 3)      → 5    — the "obvious" case (guards against no-op)
#     (10, -4)    → 6    — kills ``a - b + 6`` (10 - (-4) + 6 = 20)
#     (0, 0)      → 0    — kills any non-zero intercept
#     (-1, 1)     → 0    — kills ``a + k`` (any b-independence)
#     (100, 200)  → 300  — kills any b-scaling other than 1
CHECKS = [
    ((2, 3), 5),
    ((10, -4), 6),
    ((0, 0), 0),
    ((-1, 1), 0),
    ((100, 200), 300),
]

for (a, b), expected in CHECKS:
    try:
        got = mod.add(a, b)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[correctness] RUNTIME-ERROR: add({a}, {b}) raised {exc!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if got != expected:
        print(
            f"[correctness] VALUE-ERROR: add({a}, {b}) returned {got!r}, "
            f"expected {expected!r}",
            file=sys.stderr,
        )
        sys.exit(1)

print("[correctness] OK: all five (a, b) → a+b checks passed")
sys.exit(0)
PYEOF
then
    echo "[test_openhands.sh] FAIL: add.py correctness check failed" >&2
    echo "--- final add.py ---" >&2
    cat "$WORKDIR/add.py" >&2
    exit 3
fi

echo "[test_openhands.sh] PASS: five-pair add(a, b) == a+b checks satisfied"
exit 0
