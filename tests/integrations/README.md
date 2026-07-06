# Integration tests

End-to-end tests that exercise Rapid-MLX from a real client library.

These are **not** run as part of `pytest tests/` because they need a running
Rapid-MLX server on `http://localhost:8802` and a loaded model. The fixtures
auto-boot `rapid-mlx serve <alias>` for each family under test — no manual
server management is required.

## Two matrices — 8 agents + 3 frameworks × 4 Tier-1 families

0.10.2 PR-2 restructured this directory around **two matrices**, both sharing
the harness in `conftest.py`:

- `test_agents_matrix.py` — **8 Tier-1 agents × 4 families** (Qwen 3.6, Gemma
  4, DeepSeek V4 Flash, gpt-oss 120B) = **32 cells** + 1 shared streaming
  cell × 4 = 36 cells total.
- `test_frameworks_matrix.py` — **3 Tier-1 frameworks × 4 families** = **12
  cells**.

Total: **48 real cells** across the strong picks. Each cell drives the
real client SDK against a booted rapid-mlx server, asserts response
shape, asserts no reasoning-channel leak, and (for tool-call cells)
asserts the emitted tool_call's function name and JSON arguments.

Support ≡ a real integration test that boots the server + real model + real
client flow. See `workflow.md` W3 taxonomy §B.3.

### Tier-1 agents

| Agent | Wire | Matrix cell | Deep flow |
|---|---|---|---|
| codex-cli | `/v1/responses` | `TestCodexCLI` | (matrix only) |
| claude-code | `/v1/messages` | `TestClaudeCode` | `test_anthropic_sdk.py` |
| opencode | `/v1/chat/completions` | `TestOpenCode` | (matrix only) |
| qwen-code | `/v1/chat/completions` | `TestQwenCode` | (matrix only) |
| openhands | `/v1/chat/completions` | `TestOpenHands` (wire smoke) | (Docker E2E deferred to 0.10.6 Phase 4) |
| hermes-agent | `/v1/chat/completions` | `TestHermesAgent` | `test_hermes.py` (62-tool E2E) |
| aider | `/v1/chat/completions` | `TestAider` (wire smoke) | `test_aider.sh` (real CLI edit-and-write) |
| kilo-code | `/v1/chat/completions` | `TestKiloCode` | (matrix only) |

Bonus cell — `TestStreamingDeltas` — asserts SSE deltas parse cleanly and
contain no channel-marker leak. Not a specific agent; every family sees this
gate.

### Tier-1 frameworks

| Framework | Wire | Matrix cell | Deep flow |
|---|---|---|---|
| LangChain (+ LangGraph) | `/v1/chat/completions` | `TestLangChain` | `test_langchain.py` |
| PydanticAI | `/v1/chat/completions` | `TestPydanticAI` | `test_pydantic_ai_full.py` |
| smolagents | `/v1/chat/completions` | `TestSmolagents` | `test_smolagents_full.py` |

## Strong-pick policy (0.10.2 PR-2, 2026-07-06)

Each family runs against the **strong pick** — the largest currently-shipping
public MLX quant that fits a 512 GB M3 Ultra and still leaves headroom for
operator services on ports 8801 / 8772:

| Family | Alias | HF path | Size |
|---|---|---|---|
| Qwen 3.6 | `qwen3.6-35b-8bit` | `mlx-community/Qwen3.6-35B-A3B-8bit` | 35 GB |
| Gemma 4 | `gemma-4-31b-4bit` | `mlx-community/gemma-4-31b-it-4bit` | 18 GB |
| DeepSeek | `deepseek-v4-flash-8bit` | `mlx-community/DeepSeek-V4-Flash-8bit` | 50 GB |
| gpt-oss | `gpt-oss-120b-mxfp4-q8` | `mlx-community/gpt-oss-120b-MXFP4-Q8` | 65 GB |

Small variants (4B/12B) fail tool-calling for reasons unrelated to the wire
path (model 降智 — capability ceiling of the quant/size, not a rapid-mlx
bug), which pollutes the integration signal. Strong picks isolate engine
regressions cleanly.

## Running

Auto-boot mode (default) — the `rapid_mlx_server` fixture in `conftest.py`
boots `rapid-mlx serve <alias> --port 8802` for the family currently under
test, waits for `/v1/models` to return 200, yields the base_url, and tears
down at session end.

```bash
# Full matrix — auto-boots one server per family, sequential.
pytest tests/integrations/test_agents_matrix.py tests/integrations/test_frameworks_matrix.py -v

# Per-family shard (recommended for CI — one job per family, own boot).
RAPID_MLX_AGENT_MATRIX_FAMILY=qwen36 \
    RAPID_MLX_MATRIX_STRICT=1 \
    pytest tests/integrations/ -v

# One agent's cells across all families
pytest tests/integrations/test_agents_matrix.py -k QwenCode

# Deep flows (Python)
python3 tests/integrations/test_pydantic_ai_full.py
python3 tests/integrations/test_smolagents_full.py
python3 tests/integrations/test_langchain.py
python3 tests/integrations/test_anthropic_sdk.py
python3 tests/integrations/test_openwebui.py

# Deep flows (CLI / Docker)
bash tests/integrations/test_aider.sh
python3 tests/integrations/test_librechat_docker.py
```

External-server mode — point at an already-running server (skip the boot
dance):

```bash
RAPID_MLX_BASE_URL=http://127.0.0.1:8802/v1 \
    RAPID_MLX_AGENT_MATRIX_FAMILY=qwen36 \
    pytest tests/integrations/test_agents_matrix.py -v
```

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `RAPID_MLX_BASE_URL` | (unset — auto-boot) | Point at an already-running server instead of auto-booting one |
| `RAPID_MLX_AGENT_MATRIX_FAMILY` | (all) | Restrict to `qwen36` / `gemma4` / `deepseek` / `gptoss` |
| `RAPID_MLX_MATRIX_STRICT` | `0` | If `1`, degraded cells → fail (default: skip) |
| `RAPID_MLX_MATRIX_PORT` | `8802` | Boot port (never `8801` / `8772` — operator services) |
| `RAPID_MLX_SERVE_BIN` | `python3.12 -m vllm_mlx.cli` | How to invoke the server |

## Guardrails

- **G1**: The matrix boots on **port 8802** by default. Ports 8801
  (operator qwen3-vl) and 8772 (operator Holo3) are forbidden — the
  conftest refuses to boot on either.
- **G11**: Disk budget respected via sequential per-family boots — only
  one large model is resident at a time.

## Current cell status (2026-07-06 · 0.10.2 PR-2)

### Agent × Family matrix (8 × 4 = 32) + streaming (1 × 4 = 4)

| Agent | Qwen 3.6 | Gemma 4 | DeepSeek V4 Flash | gpt-oss 120B |
|---|---|---|---|---|
| codex-cli | ✅ | ✅ | (pending) | ✅ |
| claude-code | ✅ | ✅ | (pending) | ✅ |
| opencode | ✅ | ✅ | (pending) | ✅ |
| qwen-code | ✅ | ✅ | (pending) | ✅ |
| openhands | ✅ | ✅ | (pending) | ✅ |
| hermes-agent | ✅ | ✅ | (pending) | ✅ |
| aider | ✅ | ✅ | (pending) | ✅ |
| kilo-code | ✅ | ✅ | (pending) | ✅ |
| streaming (bonus) | ✅ | ✅ | (pending) | ✅ |

### Framework × Family matrix (3 × 4 = 12)

| Framework | Qwen 3.6 | Gemma 4 | DeepSeek V4 Flash | gpt-oss 120B |
|---|---|---|---|---|
| LangChain (+ LangGraph) | ✅ | ✅ | (pending) | ✅ |
| PydanticAI | ✅ | ✅ | (pending) | ✅ |
| smolagents | ✅ | ✅ | (pending) | ✅ |

Legend: ✅ passing · ⚠️ skipped (known cause) · (pending) DeepSeek weights still downloading · ❌ failing

## Historical deep-file coverage (pre-0.10.2)

For reference — this is what the deep flows historically covered on the
2026-06 M3 Ultra baseline before the matrix restructure:

| Test | Plain | Stream | Tool | Multi-tool | Structured | Notes |
|---|---|---|---|---|---|---|
| `test_pydantic_ai_full.py` | x | x | x | x | x | + multi-turn |
| `test_smolagents_full.py` | x | — | x | x | — | CodeAgent + ToolCallingAgent |
| `test_langchain.py` | x | x | x | x | x | + system prompt |
| `test_anthropic_sdk.py` | x | x | x | — | — | `/v1/messages` endpoint |
| `test_openwebui.py` | — | x | — | — | — | Docker: register, login, models, chat |
| `test_aider.sh` | — | — | — | — | — | CLI edit-and-write workflow |
| `test_librechat_docker.py` | — | — | — | — | — | Docker: register, login, endpoints, models |
| `test_hermes.py` | x | x | x | x | — | 62-tool Hermes Agent E2E + API stress test |

Model is auto-detected from the running server (`/v1/models` endpoint).

Run all agent tests automatically via:

```bash
rapid-mlx agents hermes --test
rapid-mlx agents                    # list all supported agents
```
