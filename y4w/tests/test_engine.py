"""Engine construction, config validation, layers and generation."""
from __future__ import annotations

import pytest
import torch

from y4w.runtime.config import EngineConfig
from y4w.runtime.linear import W4A4Linear
from y4w.runtime.engine import Engine, get_engine
from y4w.backends.cpu import CPUBackend


def test_config_defaults():
    c = EngineConfig()
    assert c.backend == "auto"
    assert c.profile == "fast"
    assert c.kv_cache_dtype == "bf16"
    assert c.ode_steps == 8


def test_config_roundtrip():
    c = EngineConfig(model_dir=".", backend="cpu", profile="memory", seed=42)
    d = EngineConfig.from_dict(c.to_dict())
    assert d.backend == "cpu" and d.profile == "memory" and d.seed == 42


@pytest.mark.parametrize("field,value", [
    ("backend", "tpu"),
    ("profile", "ultra"),
    ("kv_cache_dtype", "fp8"),
    ("vae_precision", "int4"),
    ("ode_steps", 0),
])
def test_config_rejects_invalid(field, value):
    with pytest.raises(ValueError):
        EngineConfig(**{field: value})


def test_engine_builds_on_cpu():
    e = Engine(EngineConfig(backend="cpu"))
    assert e.backend.name == "cpu"
    assert e.model is not None
    assert set(e.status()["backends"]) == {"cuda", "tensorrt_rtx", "vulkan", "cpu"}


def test_engine_status_serialisable():
    import json
    e = Engine(EngineConfig(backend="cpu"))
    json.dumps(e.status())


def test_engine_generate_deterministic():
    a = Engine(EngineConfig(backend="cpu", seed=7)).generate("hello world", max_new_tokens=6, seed=7)
    b = Engine(EngineConfig(backend="cpu", seed=7)).generate("hello world", max_new_tokens=6, seed=7)
    assert a["tokens"] == b["tokens"]
    assert a["token_count"] == 7  # 1 prompt token + 6 generated
    assert a["backend"] == "cpu"


def test_generate_with_token_list_prompt():
    e = Engine(EngineConfig(backend="cpu"))
    r = e.generate([3, 5, 7], max_new_tokens=2, seed=0)
    assert r["token_count"] == 5


def test_get_engine_is_cached():
    a = get_engine(EngineConfig(backend="cpu", seed=1))
    b = get_engine(EngineConfig(backend="cpu", seed=1))
    assert a is b


def test_w4a4_linear_load_dense_and_forward():
    from y4w.quant import dequantize_w4a4
    backend = CPUBackend()
    torch.manual_seed(0)
    layer = W4A4Linear(32, 16, backend)
    dense = torch.randn(16, 32) * 0.05
    layer.load_dense_weight(dense)
    # Contract 1: per-row-scaled 4-bit weights reconstruct the dense weight well.
    recon = dequantize_w4a4(layer.packed, layer.scales, layer.codebook)
    rel = (recon - dense).norm() / dense.norm().clamp_min(1e-9)
    assert rel.item() < 0.15, f"weight reconstruction error {rel.item()}"
    # Contract 2: forward (with 4-bit activation quantization) is finite and shaped.
    x = torch.randn(4, 32) * 0.05
    y = layer(x)
    assert y.shape == (4, 16)
    assert torch.isfinite(y).all()


def test_w4a4_linear_rejects_odd_in_features():
    with pytest.raises(ValueError):
        W4A4Linear(31, 16, CPUBackend())


def test_w4a4_linear_load_packed_shape_check():
    layer = W4A4Linear(32, 16, CPUBackend())
    with pytest.raises(ValueError):
        layer.load_packed(torch.zeros(16, 7, dtype=torch.uint8), torch.ones(16))
