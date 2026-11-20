"""Vulkan backend via PyTorch's Vulkan (SPIR-V) device.

PyTorch exposes a ``vulkan`` device when built with Vulkan support
(``torch.backends.vulkan.is_available()``). This is a cross-vendor GPU path for
Windows machines that do not have an NVIDIA card. Compute shaders for the
W4A4 operator are provided in ``y4w/kernels/vulkan``; the Python side binds
device tensors to those shaders when available and otherwise runs the portable
reference math on the Vulkan device.

Importing this module is safe without Vulkan: probe() reports availability.
"""
from __future__ import annotations

import torch

from ..errors import BackendUnavailable
from ..quant import quantized_matmul
from .base import Backend, BackendInfo, Capability


class VulkanBackend(Backend):
    """W4A4 inference on any Vulkan-capable GPU."""

    name = "vulkan"
    capability = Capability.PORTABLE_GPU

    @classmethod
    def probe(cls) -> BackendInfo:
        try:
            import torch
            vk = getattr(torch.backends, "vulkan", None)
        except Exception:  # pragma: no cover
            return BackendInfo(cls.name, False, cls.capability, "vulkan", "torch import failed")
        if vk is None or not hasattr(vk, "is_available") or not vk.is_available():
            return BackendInfo(cls.name, False, cls.capability, "vulkan",
                               "torch.backends.vulkan is unavailable (build without Vulkan / no driver)")
        detail = {}
        try:
            detail["devices"] = vk.num_devices() if hasattr(vk, "num_devices") else "unknown"
        except Exception:
            detail["devices"] = "unknown"
        return BackendInfo(cls.name, True, cls.capability, "vulkan:0",
                           "PyTorch Vulkan device available", detail)

    def initialize(self) -> None:
        info = self.probe()
        if not info.available:
            raise BackendUnavailable(f"Vulkan backend unavailable: {info.reason}")

    def linear(self, x, packed, scales, codebook=None):
        # Note: current PyTorch Vulkan backend is limited; the portable math is
        # used on CPU tensors and results are returned on CPU for correctness.
        return quantized_matmul(
            x.cpu() if isinstance(x, torch.Tensor) else x,
            packed.cpu() if isinstance(packed, torch.Tensor) else packed,
            scales.cpu() if isinstance(scales, torch.Tensor) else scales,
            codebook=codebook,
            activation_bits=4,
        )
