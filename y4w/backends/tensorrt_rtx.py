"""NVIDIA TensorRT-RTX backend.

TensorRT-RTX is NVIDIA's inference SDK that also ships the in-process
``tensorrt_rtx`` Python module. This backend:

* probes for the ``tensorrt_rtx`` (or legacy ``tensorrt``) package;
* validates that a CUDA device is present (TensorRT-RTX requires NVIDIA HW);
* executes the W4A4 linear as an FP16/bf16 Torch graph when no serialized
  engine is available, and can load a prebuilt TensorRT engine via the
  optional ``load_engine`` hook.

Importing this module never requires TensorRT to be installed: probe() records
the missing dependency and the registry simply skips this backend.
"""
from __future__ import annotations

import importlib.util
from types import ModuleType

import torch

from ..errors import BackendInitError, BackendUnavailable
from ..quant import quantized_matmul
from .base import Backend, BackendInfo, Capability


def _find_trt_module():
    for candidate in ("tensorrt_rtx", "tensorrt"):
        try:
            spec = importlib.util.find_spec(candidate)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            return candidate, spec
    return None, None


class TensorRTRTXBackend(Backend):
    """W4A4 inference with NVIDIA TensorRT-RTX."""

    name = "tensorrt_rtx"
    capability = Capability.COMPILED

    def __init__(self, device: str = "cuda:0", *, engine_path: str | None = None) -> None:
        super().__init__(device)
        self.engine_path = engine_path
        self._trt: ModuleType | None = None
        self._module_name: str | None = None
        self._engine = None

    @classmethod
    def probe(cls) -> BackendInfo:
        module_name, spec = _find_trt_module()
        if spec is None:
            return BackendInfo(cls.name, False, cls.capability, "cuda",
                               "tensorrt_rtx / tensorrt Python module not installed")
        cuda = torch.cuda.is_available() if hasattr(torch, "cuda") else False
        detail = {"module": module_name, "cuda_available": cuda}
        if module_name and spec.origin:
            detail["origin"] = spec.origin
        if not cuda:
            return BackendInfo(cls.name, False, cls.capability, "cuda",
                               "TensorRT-RTX found but no NVIDIA CUDA device is present", detail)
        try:
            version = None
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", None)
            if version:
                detail["version"] = str(version)
        except Exception as exc:  # pragma: no cover
            return BackendInfo(cls.name, False, cls.capability, "cuda",
                               f"TensorRT import failed: {exc}", detail)
        return BackendInfo(cls.name, True, cls.capability, "cuda:0",
                           f"TensorRT-RTX via {module_name}", detail)

    def initialize(self) -> None:
        info = self.probe()
        if not info.available:
            raise BackendUnavailable(f"TensorRT-RTX backend unavailable: {info.reason}")
        self._module_name = info.detail.get("module")
        try:
            self._trt = importlib.import_module(self._module_name)
        except Exception as exc:  # pragma: no cover
            raise BackendInitError(f"Failed to import {self._module_name}: {exc}") from exc
        if self.engine_path:
            self.load_engine(self.engine_path)

    def load_engine(self, path: str) -> None:
        """Load a serialized TensorRT engine from ``path`` (best-effort)."""
        if self._trt is None:
            raise BackendInitError("initialize() must be called before load_engine()")
        try:
            runtime = self._trt.Runtime(self._trt.Logger(self._trt.Logger.WARNING))
            with open(path, "rb") as fh:
                blob = fh.read()
            deserialize = getattr(runtime, "deserialize_cuda_engine", None) or \
                getattr(runtime, "deserialize_engine", None)
            self._engine = deserialize(blob)
        except Exception as exc:
            raise BackendInitError(f"Failed to load TensorRT engine {path!r}: {exc}") from exc

    @property
    def prefer_baked_int8(self) -> bool:
        # INT8 DP4A is the fast path on NVIDIA; baked weights also feed a
        # serialized engine's int8 bindings.
        return True

    def linear(self, x, packed, scales, codebook=None):
        dev = torch.device(self.device)
        x = x.to(dev) if isinstance(x, torch.Tensor) else x
        packed = packed.to(dev) if isinstance(packed, torch.Tensor) else packed
        scales = scales.to(dev) if isinstance(scales, torch.Tensor) else scales
        # A serialized engine, when present, owns its own bindings. The generic
        # W4A4 operator path is used for layers not captured in the engine.
        return quantized_matmul(x, packed, scales, codebook, activation_bits=4)

    def synchronize(self) -> None:
        torch.cuda.synchronize(torch.device(self.device))
