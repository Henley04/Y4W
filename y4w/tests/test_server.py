"""Tests for the one-command inference service (REST API + optional WebUI).

These run without a GPU. The Gradio-dependent tests skip cleanly when Gradio
is not installed, while the API tests always run (fastapi/httpx are `test`
extras). Generation uses the CPU backend so results are deterministic.
"""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from y4w.runtime.config import EngineConfig
from y4w.server.app import build_engine_fn, create_app, create_engine

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app(EngineConfig(backend="cpu", seed=1), mount_webui=False)
    with TestClient(app) as c:
        yield c


# --- REST API ---------------------------------------------------------------

def test_health(client):
    d = client.get("/health").json()
    assert d["status"] == "ok"
    assert d["backend"] == "cpu"
    assert "version" in d


def test_status(client):
    d = client.get("/v1/status").json()
    assert d["selected_backend"] == "cpu"
    assert set(d["backends"]) == {"cuda", "tensorrt_rtx", "vulkan", "cpu"}


def test_backends_endpoint(client):
    d = client.get("/v1/backends").json()
    assert d["priority_policy"] in {
        "cuda", "cuda-first", "default", "tensorrt", "tensorrt-first", "trt"
    }
    assert set(d["backends"]) == {"cuda", "tensorrt_rtx", "vulkan", "cpu"}
    assert d["order"][-1] == "cpu"  # CPU is always the last resort


def test_generate_endpoint(client):
    r = client.post("/v1/generate", json={
        "prompt": "hi", "max_new_tokens": 3, "temperature": 0.8, "seed": 5,
    })
    assert r.status_code == 200
    d = r.json()
    assert d["backend"] == "cpu"
    assert d["token_count"] == 4
    assert len(d["tokens"]) == 4
    assert d["elapsed_seconds"] >= 0


def test_generate_minimal_body(client):
    r = client.post("/v1/generate", json={})
    assert r.status_code == 200
    assert r.json()["backend"] == "cpu"


def test_generate_rejects_bad_params(client):
    assert client.post("/v1/generate", json={"max_new_tokens": 0}).status_code == 422
    assert client.post("/v1/generate", json={"temperature": 99}).status_code == 422


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


# --- generator callable / engine factory ------------------------------------

def test_create_engine_overrides():
    eng = create_engine(None, backend="cpu", seed=9)
    assert eng.backend.name == "cpu"
    assert eng.config.seed == 9


def test_build_engine_fn_is_callable_and_thread_safe():
    eng = create_engine(EngineConfig(backend="cpu", seed=2))
    fn = build_engine_fn(eng, threading.Lock())
    result = fn("hello", 2, 0.7, 3)
    assert result["backend"] == "cpu"
    assert result["token_count"] == 3


# --- WebUI mount behaviour --------------------------------------------------

def test_webui_degrades_to_api_when_gradio_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _block_gradio(name, *a, **k):
        if name == "gradio" or name.startswith("gradio."):
            raise ImportError("simulated missing gradio")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _block_gradio)
    app = create_app(EngineConfig(backend="cpu"), mount_webui=True)
    assert app.state.webui_mounted is False
    assert app.state.webui_error is not None
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.post("/v1/generate", json={"max_new_tokens": 1}).status_code == 200


def test_webui_mounted_when_gradio_available():
    pytest.importorskip("gradio")
    app = create_app(EngineConfig(backend="cpu"), mount_webui=True)
    assert app.state.webui_mounted is True, app.state.webui_error
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.get("/health").status_code == 200


# --- CLI --------------------------------------------------------------------

def test_cli_serve_help():
    r = subprocess.run(
        [sys.executable, "-m", "y4w", "serve", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "--port" in r.stdout and "--no-webui" in r.stdout
