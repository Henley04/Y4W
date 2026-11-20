"""Backend discovery and selection.

``auto`` mode chooses the highest-capability backend that is actually usable on
the current host. On a CPU-only Windows CI runner it resolves to ``cpu``; on an
NVIDIA Windows workstation it resolves to ``cuda`` (or ``tensorrt_rtx`` when
explicitly requested and installed).
"""
from __future__ import annotations

from typing import Iterable

from ..errors import BackendUnavailable
from .base import Backend, BackendInfo
from .cpu import CPUBackend
from .cuda import CUDABackend
from .tensorrt_rtx import TensorRTRTXBackend
from .vulkan import VulkanBackend

# Descending preference. TRT-RTX is compiled/fastest but opt-in because it needs
# an extra SDK; CUDA is the default NVIDIA path; Vulkan is portable; CPU always.
BACKEND_ORDER: tuple[type[Backend], ...] = (
    CUDABackend,
    TensorRTRTXBackend,
    VulkanBackend,
    CPUBackend,
)

_REGISTRY: dict[str, type[Backend]] = {cls.name: cls for cls in BACKEND_ORDER}


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
    """Names of currently usable backends, in preference order."""
    infos = detect_backends()
    return [cls.name for cls in BACKEND_ORDER if infos[cls.name].available]


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
        for cls in BACKEND_ORDER:
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
