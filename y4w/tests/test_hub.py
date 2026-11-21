"""Tests for the ModelScope-backed model download helper (no network)."""
from __future__ import annotations

import sys
import subprocess

import pytest

from y4w import hub


def test_default_model_id(monkeypatch):
    monkeypatch.delenv(hub.MODEL_ID_ENV, raising=False)
    assert hub.get_model_id() == hub.DEFAULT_MODEL_ID
    assert hub.DEFAULT_MODEL_ID == "syxppp/YuE2-3B-OrbitQuant-W4A4"


def test_model_id_env_override(monkeypatch):
    monkeypatch.setenv(hub.MODEL_ID_ENV, "someone/other-model")
    assert hub.get_model_id() == "someone/other-model"


def test_model_url():
    url = hub.model_url()
    assert url.startswith("https://www.modelscope.cn/models/")
    assert hub.DEFAULT_MODEL_ID in url


def test_download_model_forwards_args(monkeypatch):
    captured = {}

    def fake_snapshot_download(rid, **kwargs):
        captured["rid"] = rid
        captured["kwargs"] = kwargs
        return r"C:\models\YuE2-W4A4"

    # The function imports `from modelscope import snapshot_download` lazily, so patch
    # the module in sys.modules instead.
    import types
    stub = types.ModuleType("modelscope")
    stub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", stub)

    path = hub.download_model(cache_dir=r"C:\cache", local_dir=r"C:\models\YuE2-W4A4")
    assert path == r"C:\models\YuE2-W4A4"
    assert captured["rid"] == hub.DEFAULT_MODEL_ID
    assert captured["kwargs"]["cache_dir"] == r"C:\cache"
    assert captured["kwargs"]["local_dir"] == r"C:\models\YuE2-W4A4"
    assert "model.safetensors" in captured["kwargs"]["allow_patterns"]


def test_download_model_custom_patterns(monkeypatch):
    import types
    captured = {}

    def fake(rid, **kwargs):
        captured.update(kwargs)
        return "/tmp/x"

    stub = types.ModuleType("modelscope")
    stub.snapshot_download = fake
    monkeypatch.setitem(sys.modules, "modelscope", stub)

    hub.download_model(allow_patterns=["orbitquant_manifest.json"])
    assert captured["allow_patterns"] == ["orbitquant_manifest.json"]


def test_download_model_missing_sdk_raises_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "modelscope":
            raise ImportError("no modelscope")
        return real_import(name, *a, **kw)

    monkeypatch.delitem(sys.modules, "modelscope", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="pip install -e ."):
        hub.download_model()


def test_cli_download_model_help():
    r = subprocess.run(
        [sys.executable, "-m", "y4w", "download-model", "--help"],
        capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, r.stderr
    assert "--model-id" in r.stdout and "--local-dir" in r.stdout
