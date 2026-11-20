"""Backend discovery and selection.

``auto`` mode chooses the highest-priority backend that is actually usable on
the current host. On a CPU-only Windows CI runner it resolves to ``cpu``; on an
NVIDIA Windows workstation it resolves to ``cuda`` by default.

Priority is configurable because the two NVIDIA backends have different
maturity today:

* The **CUDA** backend is the only one that loads the optional native fused
  W4A4 kernel (``load_native_kernel`` -> ``w4a4_linear``). It is therefore the
  default first choice.
* The **TensorRT-RTX** backend currently probes the SDK and can load a
  serialized engine, but its ``linear()`` executes the same PyTorch W4A4 math.
  Putting it first would select the less capable path by default.

Set ``Y4W_BACKEND_PRIORITY=tensorrt`` (or pass ``--backend-priority tensorrt``
to ``y4w serve``) to reverse the two. Every other backend degrades gracefully.
"""
from __future__ import annotations

import os
from typing import Iterable

from ..errors import BackendUnavailable
from .base import Backend, BackendInfo
from .cpu import CPUBackend
from .cuda import CUDABackend
from .tensorrt_rtx import TensorRTRTXBackend
from .vulkan import VulkanBackend

# Priority presets. CUDA-first is the default; see module docstring for why.
_CUDA_FIRST: tuple[type[Backend], ...] = (
    CUDABackend,
    TensorRTRTXBackend,
    VulkanBackend,
    CPUBackend,
)
_TENSORRT_FIRST: tuple[type[Backend], ...] = (
    TensorRTRTXBackend,
    CUDABackend,
    VulkanBackend,
    CPUBackend,
)

PRIORITY_PRESETS: dict[str, tuple[type[Backend], ...]] = {
    "cuda": _CUDA_FIRST,
    "cuda-first": _CUDA_FIRST,
    "default": _CUDA_FIRST,
    "tensorrt": _TENSORRT_FIRST,
    "tensorrt-first": _TENSORRT_FIRST,
    "trt": _TENSORRT_FIRST,
    "trt-first": _TENSORRT_FIRST,
}

ENV_PRIORITY = "Y4W_BACKEND_PRIORITY"

# Backward-compatible alias used by existing callers/tests as the default
# order. Code that must honour configuration should call get_backend_order().
BACKEND_ORDER: tuple[type[Backend], ...] = _CUDA_FIRST

_REGISTRY: dict[str, type[Backend]] = {cls.name: cls for cls in _CUDA_FIRST}


def backend_priority() -> str:
    """Return the active priority policy name (env-driven, default ``cuda``)."""
    return (os.getenv(ENV_PRIORITY) or "cuda").lower()


def get_backend_order(policy: str | None = None) -> tuple[type[Backend], ...]:
    """Resolve an ordered tuple of backend classes for a policy string.

    Unknown policies fall back to the CUDA-first default rather than raising,
    so a typo in an environment variable cannot break startup.
    """
    key = (policy or backend_priority()).lower()
    return PRIORITY_PRESETS.get(key, _CUDA_FIRST)


def set_backend_priority(policy: str) -> tuple[type[Backend], ...]:
    """Set the process-wide priority policy (used by ``y4w serve``)."""
    os.environ[ENV_PRIORITY] = policy
    return get_backend_order(policy)


def _safe_probe(cls: type[Backend]) -> BackendInfo:
    try:
        return cls.probe()
    except Exception as exc:  # a broken optional dep must not crash discovery
        return BackendInfo(cls.name, False, cls.capability, "unknown",
                           f"probe raised {type(exc).__name__}: {exc}")


def detect_backends() -> dict[str, BackendInfo]:
    """Probe every known backend and return ``{name: info}``."""
    return {cls.name: _safe_probe(cls) for cls in BACKEND_ORDER}


def available_backends() -> list[str]:
    """Names of currently usable backends, in the configured preference order."""
    infos = detect_backends()
    return [cls.name for cls in get_backend_order() if infos[cls.name].available]


def select_backend(name: str = "auto", **kwargs) -> Backend:
    """Instantiate and initialize a backend.

    Args:
        name: one of ``auto``, ``cuda``, ``tensorrt_rtx``, ``vulkan``, ``cpu``.
        **kwargs: forwarded to the backend constructor (e.g. ``device=``).

    Raises:
        BackendUnavailable: if an explicitly requested backend cannot be used.
    """
    name = (name or "auto").lower()
    infos = detect_backends()

    if name == "auto":
        for cls in get_backend_order():
            if infos[cls.name].available:
                backend = cls(**_filtered_kwargs(cls, kwargs))
                backend.initialize()
                return backend
        # CPU is always available, but be defensive.
        raise BackendUnavailable("No backend available, not even CPU")

    if name not in _REGISTRY:
        raise BackendUnavailable(
            f"Unknown backend {name!r}; choose from {sorted(_REGISTRY)} or 'auto'"
        )
    info = infos[name]
    if not info.available:
        raise BackendUnavailable(f"Backend {name!r} unavailable: {info.reason}")
    backend = _REGISTRY[name](**_filtered_kwargs(_REGISTRY[name], kwargs))
    backend.initialize()
    return backend


def _filtered_kwargs(cls: type[Backend], kwargs: dict) -> dict:
    """Keep only kwargs accepted by the target backend constructor."""
    import inspect
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params and k != "self"}
