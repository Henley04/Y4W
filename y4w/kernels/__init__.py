"""Optional native kernel loading.

The Windows source tree ships CUDA C++/SPIR-V kernels under this directory
(see ``README_WINDOWS.md`` for the build commands). At runtime we try to import
the compiled extension (``y4w.kernels._w4a4_native``) and fall back silently to
the portable Torch path when it is not built. This keeps ``import y4w`` working
on every platform, including CPU-only CI.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

from ..quant import w4a4_linear as _portable_w4a4_linear


class _PortableKernel:
    """Adapter exposing the native kernel's method signature."""

    name = "portable-torch"

    @staticmethod
    def w4a4_linear(x, packed, scales, codebook=None):
        return _portable_w4a4_linear(x, packed, scales, codebook)


def load_native_kernel():
    """Return a native kernel module, or ``None`` if unavailable.

    Never raises: accelerated backends treat ``None`` as "use reference math".
    """
    try:
        module = importlib.import_module("y4w.kernels._w4a4_native")
        if hasattr(module, "w4a4_linear"):
            return module
    except Exception:
        return None
    return None


def kernel_status() -> dict:
    """Diagnostic info used by ``y4w --info``."""
    native = load_native_kernel()
    return {
        "native_loaded": native is not None,
        "native_name": getattr(native, "name", None) if native else None,
        "fallback": _PortableKernel.name,
    }


__all__ = ["load_native_kernel", "kernel_status"]
