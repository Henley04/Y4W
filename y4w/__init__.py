"""y4w - Windows-native YuE2 OrbitQuant W4A4 inference engine.

The engine is backend-agnostic: it detects and dispatches to one of three GPU
backends (CUDA, TensorRT-RTX, Vulkan) and gracefully falls back to a portable
CPU / Torch path so that the package imports and passes its test-suite on
machines without an accelerator (notably GitHub-hosted Windows runners).

Model format compatibility target: ``WaveCut/YuE2-3B-OrbitQuant-W4A4``
(OrbitQuant W4A4: 4-bit packed weights + 4-bit activations, 16-entry
Lloyd-Max codebook, per-row bfloat16 scales, RPBH rotation).
"""

from __future__ import annotations

__version__ = "0.2.1"
__all__ = ["__version__", "get_engine", "Engine", "detect_backends"]

from .runtime.engine import Engine, get_engine
from .backends.registry import detect_backends
