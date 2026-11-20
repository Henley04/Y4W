"""CLI smoke tests (subprocess-level, no GPU required)."""
from __future__ import annotations

import json
import subprocess
import sys

from y4w import __version__


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "y4w", *args],
        capture_output=True, text=True, timeout=120,
    )


def test_cli_version():
    r = _run("--version")
    assert r.returncode == 0
    assert __version__ in (r.stdout + r.stderr)


def test_cli_info_json():
    r = _run("--info")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["y4w_version"] == __version__
    assert set(data["backends"]) == {"cuda", "tensorrt_rtx", "vulkan", "cpu"}
    assert "cpu" in data["available_backends"]


def test_cli_list_backends():
    r = _run("--list-backends")
    assert r.returncode == 0
    for name in ("cuda", "tensorrt_rtx", "vulkan", "cpu"):
        assert name in r.stdout


def test_cli_generate_cpu():
    r = _run("generate", "--backend", "cpu", "--prompt", "hi", "--max-new-tokens", "3", "--seed", "5")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["backend"] == "cpu"
    assert data["token_count"] == 4


def test_cli_generate_output_file(tmp_path):
    out = tmp_path / "result.json"
    r = _run("generate", "--backend", "cpu", "--max-new-tokens", "2", "--output", str(out))
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert json.loads(out.read_text())["backend"] == "cpu"
