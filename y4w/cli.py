"""Command-line interface for y4w."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .backends.registry import (
    available_backends, backend_priority, detect_backends, get_backend_order,
    set_backend_priority,
)
from .kernels import kernel_status
from .runtime.config import EngineConfig
from .runtime.engine import Engine


def _guidance(value: str) -> float:
    v = float(value)
    if not 0.0 <= v <= 20.0:
        raise argparse.ArgumentTypeError("cfg scale must be in [0, 20]")
    return v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="y4w",
        description="Windows-native YuE2 OrbitQuant W4A4 inference engine",
    )
    parser.add_argument("--version", action="version", version=f"y4w {__version__}")
    parser.add_argument("--info", action="store_true",
                        help="Print backend/hardware/kernel diagnostics as JSON")
    parser.add_argument("--list-backends", action="store_true",
                        help="List backends and availability")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Run inference (synthetic model unless --model-dir is given)")
    gen.add_argument("--model-dir", help="Directory with orbitquant_manifest.json + packed weights")
    gen.add_argument("--backend", default="auto",
                     choices=["auto", "cuda", "tensorrt_rtx", "vulkan", "cpu"])
    gen.add_argument("--profile", default="fast",
                     choices=["reference", "fused", "fast", "memory"])
    gen.add_argument("--prompt", default=None)
    gen.add_argument("--max-new-tokens", type=int, default=8)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--kv-cache-dtype", choices=["bf16", "int8"], default="bf16")
    gen.add_argument("--cfg-scale", type=_guidance, default=None)
    gen.add_argument("--no-optimize", action="store_true",
                     help="Disable all accelerations (pure reference path)")
    gen.add_argument("--no-compile", action="store_true",
                     help="Disable torch.compile / CUDA graphs")
    gen.add_argument("--int8-kv", action="store_true",
                     help="Use int8 KV cache (halves KV-cache memory)")
    gen.add_argument("--output", help="Write JSON result to this file")

    srv = sub.add_parser(
        "serve",
        help="One-command inference service: Gradio WebUI + REST API on one port",
    )
    srv.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    srv.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    srv.add_argument("--model-dir", help="Directory with orbitquant_manifest.json + packed weights")
    srv.add_argument("--backend", default="auto",
                     choices=["auto", "cuda", "tensorrt_rtx", "vulkan", "cpu"])
    srv.add_argument("--profile", default="fast",
                     choices=["reference", "fused", "fast", "memory"])
    srv.add_argument("--backend-priority", choices=["cuda", "tensorrt"], default=None,
                     help="auto-mode priority: cuda-first (default) or tensorrt-first")
    srv.add_argument("--no-webui", action="store_true",
                     help="Run the REST API only (skip the Gradio WebUI)")
    srv.add_argument("--no-optimize", action="store_true",
                     help="Disable all accelerations (pure reference path)")
    srv.add_argument("--no-compile", action="store_true",
                     help="Disable torch.compile / CUDA graphs")
    srv.add_argument("--int8-kv", action="store_true",
                     help="Use int8 KV cache (halves KV-cache memory)")

    am = sub.add_parser(
        "abc2midi",
        help="Convert a YuE2 two-voice ABC score into vocal + accompaniment MIDI files",
    )
    src = am.add_mutually_exclusive_group(required=True)
    src.add_argument("--abc", help="ABC score text (inline)")
    src.add_argument("--abc-file", help="Path to a .abc file")
    am.add_argument("--out-dir", default=".",
                    help="Directory for vocal.mid / accompaniment.mid (default: .)")
    am.add_argument("--vocal-filename", default="vocal.mid")
    am.add_argument("--accompaniment-filename", default="accompaniment.mid")
    return parser


def _accel_config(base_engine_config, args):
    """Apply CLI acceleration flags to an EngineConfig (or dict of overrides)."""
    values = {
        "optimize": not args.no_optimize,
        "compile_model": not args.no_compile,
        "int8_kv_cache": args.int8_kv,
    }
    if hasattr(base_engine_config, "__dict__") and hasattr(base_engine_config, "optimize"):
        for k, v in values.items():
            setattr(base_engine_config, k, v)
        return base_engine_config
    return values


def cmd_info() -> dict:
    infos = detect_backends()
    return {
        "y4w_version": __version__,
        "python": sys.version.split()[0],
        "backend_priority_policy": backend_priority(),
        "backend_priority_order": [c.name for c in get_backend_order()],
        "available_backends": available_backends(),
        "backends": {name: {
            "available": i.available,
            "capability": int(i.capability),
            "device": i.device,
            "reason": i.reason,
            "detail": i.detail,
        } for name, i in infos.items()},
        "kernels": kernel_status(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.info:
        print(json.dumps(cmd_info(), indent=2, ensure_ascii=False))
        return 0

    if args.list_backends:
        infos = detect_backends()
        for name, info in infos.items():
            mark = "OK" if info.available else "--"
            print(f"[{mark}] {name:<14} {info.reason}")
        print("available (auto priority): " + " > ".join(available_backends()))
        return 0

    if args.command == "generate":
        cfg = EngineConfig(
            model_dir=args.model_dir,
            backend=args.backend,
            profile=args.profile,
            kv_cache_dtype=args.kv_cache_dtype,
            seed=args.seed,
        )
        cfg = _accel_config(cfg, args)
        engine = Engine(cfg)
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        if args.cfg_scale is not None:
            result["cfg_scale"] = args.cfg_scale
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
        print(payload)
        return 0

    if args.command == "serve":
        if args.backend_priority:
            set_backend_priority(args.backend_priority)
        from .server import run_server
        cfg = EngineConfig(
            model_dir=args.model_dir,
            backend=args.backend,
            profile=args.profile,
        )
        cfg = _accel_config(cfg, args)
        run_server(host=args.host, port=args.port, config=cfg,
                   mount_webui=not args.no_webui)
        return 0

    if args.command == "abc2midi":
        from .abc2midi import abc_to_midi_files
        if args.abc_file:
            with open(args.abc_file, "r", encoding="utf-8") as fh:
                abc_text = fh.read()
        else:
            abc_text = args.abc
        data = abc_to_midi_files(
            abc_text,
            args.out_dir,
            vocal_filename=args.vocal_filename,
            accompaniment_filename=args.accompaniment_filename,
        )
        payload = {
            "ok": True,
            "vocal_path": data["vocal_path"],
            "accompaniment_path": data["accompaniment_path"],
            "vocal_note_count": data["vocal_note_count"],
            "accompaniment_note_count": data["accompaniment_note_count"],
            "voices": data["voices"],
            "warnings": data["warnings"],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
