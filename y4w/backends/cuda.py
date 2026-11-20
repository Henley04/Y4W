"""NVIDIA CUDA backend.

Target compute capabilities mirror the upstream YuE2 OrbitQuant release:
SM89 (Ada / RTX 40) and SM120 (Blackwell consumer). On a CUDA-capable Windows
machine the backend runs packed W4A4 matmul through PyTorch; when the optional
native extension (``y4w.kernels``) is built it is used for the fused decode
kernel, otherwise the portable reference math is used on device tensors.

No CUDA is required to import this module: probe() safely reports availability.
"""
from __future__ import annotations

import torch

from ..errors import BackendUnavailable, UnsupportedGPU
from ..quant import quantized_matmul
from .base import Backend, BackendInfo, Capability

SUPPORTED_CAPABILITIES = {(8, 9), (12, 0)}  # SM89, SM120


class CUDABackend(Backend):
    """W4A4 inference on NVIDIA GPUs via CUDA."""

    name = "cuda"
    capability = Capability.ACCELERATED

    def __init__(self, device: str = "cuda:0", *, strict_capability: bool = True) -> None:
        super().__init__(device)
        self.strict_capability = strict_capability
        self._native = None

    @classmethod
    def probe(cls) -> BackendInfo:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            return BackendInfo(cls.name, False, cls.capability, "cuda", f"torch import failed: {exc}")
        if not torch.cuda.is_available():
            return BackendInfo(cls.name, False, cls.capability, "cuda",
                               "torch.cuda.is_available() is False (no driver / no GPU)")
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            count = torch.cuda.device_count()
        except Exception as exc:
            return BackendInfo(cls.name, False, cls.capability, "cuda", f"driver query failed: {exc}")
        supported = tuple(cap) in SUPPORTED_CAPABILITIES
        return BackendInfo(
            name=cls.name,
            available=True,
            capability=cls.capability,
            device="cuda:0",
            reason="NVIDIA CUDA runtime detected" + ("" if supported else " (GPU not in validated SM set)"),
            detail={"device_name": name, "capability": f"{cap[0]}.{cap[1]}",
                    "device_count": count, "validated_sm": supported,
                    "torch_cuda": torch.version.cuda},
        )

    def initialize(self) -> None:
        info = self.probe()
        if not info.available:
            raise BackendUnavailable(f"CUDA backend unavailable: {info.reason}")
        if self.strict_capability and not info.detail.get("validated_sm"):
            cap = torch.cuda.get_device_capability(0)
            raise UnsupportedGPU((int(cap[0]), int(cap[1])))
        torch.cuda.set_device(torch.device(self.device))
        # Try to load the optional native fused kernel; never fatal if absent.
        try:
            from ..kernels import load_native_kernel
            self._native = load_native_kernel()
        except Exception:
            self._native = None

    @property
    def uses_native_kernel(self) -> bool:
        return self._native is not None

    def linear(self, x, packed, scales, codebook=None):
        dev = torch.device(self.device)
        x = x.to(dev) if isinstance(x, torch.Tensor) else x
        packed = packed.to(dev) if isinstance(packed, torch.Tensor) else packed
        scales = scales.to(dev) if isinstance(scales, torch.Tensor) else scales
        if self._native is not None:
            try:
                return self._native.w4a4_linear(x, packed, scales, codebook)
            except Exception:
                # Fall back to device-side reference math on any kernel error.
                self._native = None
        return quantized_matmul(x, packed, scales, codebook, activation_bits=4)

    @property
    def prefer_baked_int8(self) -> bool:
        # DP4A is available on SM89/SM120 (the validated target set).
        return True

    def linear_int8(self, x, w_int8, w_scales, bias=None):
        from ..quant.int8 import int8_linear
        dev = torch.device(self.device)
        x = x.to(dev) if isinstance(x, torch.Tensor) else x
        w_int8 = w_int8.to(dev) if isinstance(w_int8, torch.Tensor) else w_int8
        w_scales = w_scales.to(dev) if isinstance(w_scales, torch.Tensor) else w_scales
        if bias is not None:
            bias = bias.to(dev)
        # torch._int_mm (DP4A) requires CUDA int8; fall back to portable if the
        # build lacks it or the shapes are unsupported.
        return int8_linear(x, w_int8, w_scales, bias, use_dp4a=True)

    def synchronize(self) -> None:
        torch.cuda.synchronize(torch.device(self.device))

    @classmethod
    def prepare_weight(cls, weight: torch.Tensor, packed: torch.Tensor,
                       scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Move a packed weight + scales onto the GPU (helper for the engine)."""
        return packed.to("cuda"), scales.to("cuda")
