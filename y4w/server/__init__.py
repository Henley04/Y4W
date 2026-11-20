"""One-command inference service: REST API + Gradio WebUI on a single port.

Exports:
    create_app  -- build the ASGI app (API always, WebUI when Gradio is present)
    run_server  -- launch it with uvicorn (used by ``y4w serve``)
    build_engine_fn -- the callable shared by the API and the WebUI
"""
from __future__ import annotations

from .app import create_app, run_server, create_engine, build_engine_fn

__all__ = ["create_app", "run_server", "create_engine", "build_engine_fn"]
