"""Backend implementations for y4w.

Priority order when ``backend='auto'`` is configurable. Default:
CUDA -> TensorRT-RTX -> Vulkan -> CPU. CUDA is first by default because it is
the only backend that loads the native fused W4A4 kernel; TensorRT-RTX can be
moved ahead via ``Y4W_BACKEND_PRIORITY=tensorrt``. All backends degrade
gracefully when their SDK / device is absent.
"""
from __future__ import annotations

from .base import Backend, BackendInfo, Capability
from .registry import (
    detect_backends,
    select_backend,
    available_backends,
    backend_priority,
    get_backend_order,
    set_backend_priority,
    BACKEND_ORDER,
)

__all__ = [
    "Backend", "BackendInfo", "Capability", "detect_backends",
    "select_backend", "available_backends", "backend_priority",
    "get_backend_order", "set_backend_priority", "BACKEND_ORDER",
]
