"""Backend discovery, selection and graceful degradation (GPU-free)."""
from __future__ import annotations

import pytest

from y4w.backends.base import BackendInfo, Capability
from y4w.backends.registry import (
    BACKEND_ORDER, available_backends, detect_backends, select_backend,
)
from y4w.backends.cpu import CPUBackend
from y4w.errors import BackendUnavailable


def test_all_four_backends_are_discovered():
    infos = detect_backends()
    assert set(infos) == {"cuda", "tensorrt_rtx", "vulkan", "cpu"}
    assert len(BACKEND_ORDER) == 4


def test_cpu_always_available():
    info = CPUBackend.probe()
    assert info.available
    assert info.capability == Capability.REFERENCE
    assert bool(info) is True


def test_capability_values_are_distinct_and_ordered():
    # Higher capability == faster path. TRT-RTX (compiled) outranks CUDA,
    # which outranks Vulkan, which outranks the CPU reference path.
    assert Capability.REFERENCE < Capability.PORTABLE_GPU < Capability.ACCELERATED < Capability.COMPILED
    caps = {cls.name: int(cls.capability) for cls in BACKEND_ORDER}
    assert caps["tensorrt_rtx"] > caps["cuda"] > caps["vulkan"] > caps["cpu"]


def test_auto_priority_puts_nvidia_paths_first():
    # auto prefers an NVIDIA fast path; TRT-RTX is opt-in (extra SDK) so CUDA
    # is tried first in the default order.
    names = [cls.name for cls in BACKEND_ORDER]
    assert names == ["cuda", "tensorrt_rtx", "vulkan", "cpu"]


def test_unavailable_backend_bool_false():
    info = BackendInfo("fake", False)
    assert not info
    assert bool(info) is False


def test_auto_selects_a_usable_backend():
    b = select_backend("auto")
    assert b.name in available_backends()
    assert b.name in {"cuda", "tensorrt_rtx", "vulkan", "cpu"}


def test_explicit_cpu_backend_works():
    b = select_backend("cpu")
    assert isinstance(b, CPUBackend)


def test_explicit_unknown_backend_raises():
    with pytest.raises(BackendUnavailable):
        select_backend("does_not_exist")


def test_unavailable_gpu_backend_raises_cleanly():
    infos = detect_backends()
    for name in ("cuda", "tensorrt_rtx", "vulkan"):
        if not infos[name].available:
            with pytest.raises(BackendUnavailable):
                select_backend(name)
            break


def test_available_backends_preference_order():
    avail = available_backends()
    index = {cls.name: i for i, cls in enumerate(BACKEND_ORDER)}
    assert all(index[a] <= index[b] for a, b in zip(avail, avail[1:])), \
        "available_backends must follow priority order"


def test_probe_never_raises_even_if_class_broken(monkeypatch):
    from y4w.backends import registry
    class Broken:
        name = "broken"
        capability = Capability.REFERENCE
        @staticmethod
        def probe():
            raise RuntimeError("boom")
    info = registry._safe_probe(Broken)
    assert info.available is False
    assert "boom" in info.reason
