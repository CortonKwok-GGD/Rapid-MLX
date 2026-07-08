"""Unit tests for ``scripts/mirror_to_r2.py``.

Three cases, all offline (no network, no boto3 calls):

1. CLI argparse smoke — the parser accepts the documented flag set and
   surfaces defaults.
2. Content-type router shape — ``.json`` / ``.md`` / everything-else.
3. Skip-if-exists head-check logic — with a mocked boto3-shaped client,
   an R2 object whose size matches HF is SKIPPED (no ``upload_file``
   call); a size mismatch or 404 forces an upload.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Load the CLI module from ``scripts/`` — it isn't a package member.
_SCRIPT = Path(__file__).parent.parent / "scripts" / "mirror_to_r2.py"
_SPEC = importlib.util.spec_from_file_location("mirror_to_r2", _SCRIPT)
assert _SPEC and _SPEC.loader
mirror_to_r2 = importlib.util.module_from_spec(_SPEC)
sys.modules["mirror_to_r2"] = mirror_to_r2
_SPEC.loader.exec_module(mirror_to_r2)


# --------- 1. CLI argparse smoke ---------


def test_cli_parser_accepts_repo_id_and_defaults() -> None:
    """The parser accepts a bare repo_id and surfaces all documented defaults."""
    p = mirror_to_r2._build_parser()
    args = p.parse_args(["mlx-community/Qwen3-0.6B-4bit"])
    assert args.repo_id == "mlx-community/Qwen3-0.6B-4bit"
    assert args.bucket == mirror_to_r2.DEFAULT_BUCKET
    assert args.profile == mirror_to_r2.DEFAULT_PROFILE
    assert args.endpoint_url == mirror_to_r2.DEFAULT_ENDPOINT_URL
    assert args.public_base == mirror_to_r2.DEFAULT_PUBLIC_BASE
    assert args.dry_run is False
    assert args.verify_only is False
    assert args.tmp_dir is None


def test_cli_parser_accepts_all_flags() -> None:
    """Every documented flag can be overridden from the CLI."""
    p = mirror_to_r2._build_parser()
    args = p.parse_args(
        [
            "some/repo",
            "--endpoint-url",
            "https://elsewhere.example",
            "--bucket",
            "other-bucket",
            "--profile",
            "other-profile",
            "--public-base",
            "https://elsewhere.example",
            "--dry-run",
            "--tmp-dir",
            "/mnt/scratch",
        ]
    )
    assert args.endpoint_url == "https://elsewhere.example"
    assert args.bucket == "other-bucket"
    assert args.profile == "other-profile"
    assert args.public_base == "https://elsewhere.example"
    assert args.dry_run is True
    assert args.tmp_dir == "/mnt/scratch"


def test_cli_parser_rejects_missing_repo_id() -> None:
    """Bare invocation is an error (repo_id is positional-required)."""
    p = mirror_to_r2._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


# --------- 2. Content-type router shape ---------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("config.json", "application/json"),
        ("tokenizer_config.json", "application/json"),
        ("nested/dir/config.json", "application/json"),
        ("README.md", "text/markdown"),
        ("subdir/NOTES.md", "text/markdown"),
        ("model.safetensors", "application/octet-stream"),
        ("model-00001-of-00013.safetensors", "application/octet-stream"),
        ("tokenizer.model", "application/octet-stream"),
        ("special_tokens_map", "application/octet-stream"),
        ("weights.bin", "application/octet-stream"),
        (".gitattributes", "application/octet-stream"),
    ],
)
def test_content_type_for(filename: str, expected: str) -> None:
    assert mirror_to_r2.content_type_for(filename) == expected


def test_content_type_for_is_case_insensitive_on_extension() -> None:
    """Uppercase extensions map to the same MIME type as lowercase."""
    assert mirror_to_r2.content_type_for("README.MD") == "text/markdown"
    assert mirror_to_r2.content_type_for("Config.JSON") == "application/json"


# --------- 3. Skip-if-exists head-check logic ---------


def test_should_skip_when_r2_size_matches_hf_size() -> None:
    """R2 already has the object at the expected size → SKIP (no upload)."""
    assert mirror_to_r2.should_skip(existing_size=937, expected_size=937) is True


def test_should_skip_false_when_r2_missing() -> None:
    """R2 HEAD → 404 (``None``) means we must upload."""
    assert mirror_to_r2.should_skip(existing_size=None, expected_size=937) is False


def test_should_skip_false_on_size_mismatch() -> None:
    """R2 has an object of the wrong size (partial upload) → re-upload."""
    assert mirror_to_r2.should_skip(existing_size=500, expected_size=937) is False


def test_should_skip_false_when_hf_size_unknown() -> None:
    """If HF metadata didn't expose a size, refuse to skip.

    Prevents accepting a truncated leftover R2 object from a previous
    run where we couldn't validate the true length.
    """
    assert mirror_to_r2.should_skip(existing_size=500, expected_size=0) is False


def test_r2_head_size_returns_content_length() -> None:
    """A mocked boto3 client returns the size on ``head_object``."""
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 12345}
    got = mirror_to_r2._r2_head_size(client, "bucket", "key")
    assert got == 12345
    client.head_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_r2_head_size_returns_none_on_404() -> None:
    """A boto3 ``ClientError`` with ``Code=404`` maps to ``None``."""
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    assert mirror_to_r2._r2_head_size(client, "bucket", "key") is None


def test_r2_head_size_raises_on_permission_error() -> None:
    """Non-404 errors bubble up (fail-fast on real problems)."""
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "HeadObject"
    )
    with pytest.raises(ClientError):
        mirror_to_r2._r2_head_size(client, "bucket", "key")
