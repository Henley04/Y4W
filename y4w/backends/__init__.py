"""Backend implementations for y4w.

Priority order when ``backend='auto'``: CUDA -> TensorRT-RTX -> Vulkan -> CPU.
CUDA and TensorRT-RTX are the NVIDIA fast paths required by the engine; Vulkan
is a portable cross-vendor path; CPU is the always-available reference path.
"""
from __future__ import annotations

from .base import Backend, BackendInfo, Capability
from .registry import (
    detect_backends,
    select_backend,
    available_backends,
    BACKEND_ORDER,
)

__all__ = [
    "Backend", "BackendInfo", "Capability", "detect_backends",
    "select_backend", "available_backends", "BACKEND_ORDER",
]
