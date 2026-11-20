"""Command-line interface for y4w."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .backends.registry import available_backends, detect_backends
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
    gen.add_argument("--output", help="Write JSON result to this file")
    return parser


def cmd_info() -> dict:
    infos = detect_backends()
    return {
        "y4w_version": __version__,
        "python": sys.version.split()[0],
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
