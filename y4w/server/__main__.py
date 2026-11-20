"""Allow ``python -m y4w.server`` as an alias for ``y4w serve``."""
from __future__ import annotations

import argparse

from .app import run_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="y4w.server", description="Launch the Y4W inference service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--no-webui", action="store_true")
    args = parser.parse_args(argv)
    from ..runtime.config import EngineConfig
    cfg = EngineConfig(model_dir=args.model_dir, backend=args.backend)
    run_server(host=args.host, port=args.port, config=cfg,
               mount_webui=not args.no_webui)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
