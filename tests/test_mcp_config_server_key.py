# SPDX-License-Identifier: Apache-2.0
"""The MCP config loader must accept the ecosystem-standard ``mcpServers`` key.

Claude Desktop, ``.mcp.json``, VS Code — and our own
``docs/guides/mcp-tools.md`` — all key the server map under ``mcpServers``.
The loader historically read only ``servers``, so a user who pasted a standard
config (or copied our own documented example) got ZERO servers loaded with no
error: a mysteriously tool-less agent. These tests pin the both-keys contract:
``mcpServers`` is accepted and preferred, ``servers`` still works for
back-compat, and a non-empty config with neither key warns instead of silently
loading nothing.
"""

from __future__ import annotations

import json
import logging

import pytest

from vllm_mlx.mcp.config import create_example_config, validate_config
from vllm_mlx.mcp.types import MCPConfig, MCPTransport

# ``python3`` is on the security allowlist AND always on PATH (it's the
# interpreter running pytest), so config validation never flakes on a missing
# binary — matching the house style in test_chat_mcp.py.
_SERVER = {"command": "python3", "args": ["-m", "some_mcp_server"]}


def test_mcpservers_standard_key_loads_servers():
    cfg = validate_config({"mcpServers": {"filesystem": _SERVER}})
    assert list(cfg.servers) == ["filesystem"]
    assert cfg.servers["filesystem"].command == "python3"


def test_legacy_servers_key_still_loads():
    # Back-compat: existing rapid-mlx configs used the "servers" key.
    cfg = validate_config({"servers": {"filesystem": _SERVER}})
    assert list(cfg.servers) == ["filesystem"]


def test_mcpservers_preferred_when_both_present(caplog):
    # Standard key wins; the legacy key is ignored (and a warning explains it).
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.mcp.config"):
        cfg = validate_config(
            {
                "mcpServers": {"std": _SERVER},
                "servers": {"legacy": _SERVER},
            }
        )
    assert list(cfg.servers) == ["std"]
    assert "legacy" not in cfg.servers
    assert any("both 'mcpServers' and 'servers'" in r.message for r in caplog.records)


def test_neither_key_nonempty_config_warns(caplog):
    # A non-empty config with a wrong/missing server key must NOT silently
    # yield an empty server set — warn so the misconfig is visible.
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.mcp.config"):
        cfg = validate_config({"default_timeout": 45.0})
    assert cfg.servers == {}
    assert any(
        "neither 'mcpServers' nor 'servers'" in r.message for r in caplog.records
    )


def test_empty_mcpservers_does_not_warn(caplog):
    # ``{"mcpServers": {}}`` is a legitimate "no servers yet" config (the
    # discovery-test fixture writes exactly this) — it must not trip the
    # neither-key warning.
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.mcp.config"):
        cfg = validate_config({"mcpServers": {}})
    assert cfg.servers == {}
    assert not any("neither" in r.message for r in caplog.records)


def test_standard_stdio_schema_omits_transport():
    # The exact shape our mcp-tools.md guide documents: command + args, no
    # explicit "transport". It must parse as stdio via the field default.
    cfg = validate_config(
        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                }
            }
        }
    )
    assert cfg.servers["filesystem"].transport is MCPTransport.STDIO


def test_from_dict_accepts_mcpservers():
    # The secondary MCPConfig.from_dict path mirrors validate_config.
    cfg = MCPConfig.from_dict({"mcpServers": {"filesystem": dict(_SERVER)}})
    assert list(cfg.servers) == ["filesystem"]


def test_present_but_null_mcpservers_rejected_consistently():
    # A present-but-invalid standard key (JSON ``null``) must RAISE on BOTH
    # paths — never silently fall back to a legacy ``servers`` value. Keying
    # on presence keeps from_dict and validate_config consistent.
    bad = {"mcpServers": None, "servers": {"legacy": dict(_SERVER)}}
    with pytest.raises(ValueError, match="mcpServers"):
        validate_config(dict(bad))
    with pytest.raises(ValueError, match="mcpServers"):
        MCPConfig.from_dict(dict(bad))


def test_example_config_uses_mcpservers_and_roundtrips():
    data = json.loads(create_example_config())
    # The emitted example must key under the standard name...
    assert "mcpServers" in data
    assert "servers" not in data
    # ...and must itself be a valid config the loader accepts.
    cfg = validate_config(data)
    assert set(cfg.servers) == set(data["mcpServers"])
