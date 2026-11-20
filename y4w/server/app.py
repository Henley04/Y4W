"""FastAPI application exposing the Y4W engine over HTTP.

Endpoints (all work with or without Gradio installed):
    GET  /health          liveness probe
    GET  /v1/status       engine + backend diagnostics
    GET  /v1/backends     backend availability / priority policy
    POST /v1/generate     run inference
    POST /v1/abc2midi     convert a YuE2 ABC score into vocal + accompaniment MIDI

When Gradio is available, its WebUI is mounted at ``/`` so a single
``y4w serve`` process provides the UI, the REST API and OpenAPI docs at
``/docs`` on one port. Generation is serialized with a lock because the
engine owns backend state (CUDA context / native kernel) that is not
thread-safe for concurrent calls.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from .. import __version__
from ..backends.registry import available_backends, backend_priority, detect_backends
from ..runtime.config import EngineConfig
from ..runtime.engine import Engine

try:  # optional: the API works without Gradio
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover - only when serving deps are absent
    _HAS_FASTAPI = False
    FastAPI = object  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore


if _HAS_FASTAPI:

    class GenerateRequest(BaseModel):
        prompt: str | None = None
        max_new_tokens: int = Field(default=16, ge=1, le=4096)
        temperature: float = Field(default=0.8, gt=0, le=20)
        seed: int | None = None

    class GenerateResponse(BaseModel):
        tokens: list[int]
        token_count: int
        backend: str
        elapsed_seconds: float

    class Abc2MidiRequest(BaseModel):
        abc: str = Field(..., description="YuE2 two-voice ABC score text")
else:  # pragma: no cover
    class GenerateRequest:  # type: ignore
        pass

    class GenerateResponse:  # type: ignore
        pass

    class Abc2MidiRequest:  # type: ignore
        pass


def create_engine(config: EngineConfig | None = None, **overrides) -> Engine:
    """Create an engine from a config, optionally overriding selected fields."""
    cfg = config or EngineConfig()
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return Engine(cfg)


def build_engine_fn(engine: Engine, lock: threading.Lock) -> Callable[..., dict[str, Any]]:
    """Return a thread-safe ``(prompt, n, temp, seed) -> result`` callable.

    Shared by the REST route and the Gradio callbacks so both paths use the
    exact same engine instance and serialization semantics.
    """
    def _generate(prompt=None, max_new_tokens=16, temperature=0.8, seed=None):
        with lock:
            return engine.generate(
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                seed=None if seed in (None, "", -1) else int(seed),
            )
    return _generate


def create_app(config: EngineConfig | None = None, *,
               mount_webui: bool = True,
               engine: Engine | None = None) -> "FastAPI":
    """Build the ASGI application.

    Args:
        config: engine configuration; defaults are used when omitted.
        mount_webui: attempt to mount the Gradio WebUI at ``/``. If Gradio is
            not installed (or fails to build) the API still works.
        engine: pre-built engine to reuse (useful for tests / embedding).
    """
    if not _HAS_FASTAPI:  # pragma: no cover
        raise RuntimeError(
            "fastapi is required to serve; install with `pip install y4w[serve]`"
        )

    app = FastAPI(title="Y4W Inference Service", version=__version__)

    engine = engine or create_engine(config)
    lock = threading.Lock()
    generate_fn = build_engine_fn(engine, lock)

    state = app.state
    state.engine = engine
    state.generate_fn = generate_fn
    state.config = engine.config
    state.webui_mounted = False
    state.webui_error = None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "backend": engine.backend.name,
            "webui": state.webui_mounted,
        }

    @app.get("/v1/status")
    def status() -> dict[str, Any]:
        return engine.status()

    @app.get("/v1/backends")
    def backends() -> dict[str, Any]:
        return {
            "priority_policy": backend_priority(),
            "order": available_backends(),
            "backends": {
                name: {"available": info.available, "device": info.device,
                       "reason": info.reason}
                for name, info in detect_backends().items()
            },
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> dict[str, Any]:
        return generate_fn(req.prompt, req.max_new_tokens,
                           req.temperature, req.seed)

    @app.post("/v1/abc2midi")
    def abc2midi(req: Abc2MidiRequest) -> dict[str, Any]:
        """Convert a YuE2 ABC score into two base64-encoded MIDI files."""
        from ..abc2midi import abc_to_midi
        if not req.abc or not req.abc.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="abc must not be empty")
        try:
            result = abc_to_midi(req.abc)
        except Exception as exc:  # malformed ABC -> surface a clear error
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail=f"ABC conversion failed: {type(exc).__name__}: {exc}",
            )
        payload = result.to_dict(base64=True)
        # Return base64 fields only (drop the raw byte-list fields).
        payload.pop("vocal_midi", None)
        payload.pop("accompaniment_midi", None)
        payload["ok"] = True
        return payload

    if mount_webui:
        try:
            from .gradio_ui import build_gradio_interface
            import gradio as gr
            demo = build_gradio_interface(generate_fn, engine)
            gr.mount_gradio_app(app, demo, path="/")
            state.webui_mounted = True
        except Exception as exc:  # missing Gradio or version mismatch -> API only
            state.webui_error = f"{type(exc).__name__}: {exc}"

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, *,
               config: EngineConfig | None = None,
               mount_webui: bool = True,
               log_level: str = "info") -> None:
    """Launch the service with uvicorn (blocking)."""
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is required to serve; install with `pip install y4w[serve]`"
        ) from exc
    app = create_app(config, mount_webui=mount_webui)
    if not app.state.webui_mounted and mount_webui:
        print(f"[y4w] WebUI unavailable ({app.state.webui_error}); API only.")
    print(f"[y4w] Y4W service at http://{host}:{port}  "
          f"(WebUI {'on' if app.state.webui_mounted else 'off'}, "
          f"API docs at /docs)")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
