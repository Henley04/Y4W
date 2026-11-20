"""Tests for the baked-int8 fast path and acceleration config (GPU-free).

The DP4A kernel itself is CUDA-only, but the math, baking, memory accounting
and graceful degradation are fully exercised on CPU here.
"""
from __future__ import annotations

import torch

from y4w.quant import (
    bake_int8_from_float, bake_int8_from_packed, quantize_activation_int8,
    int8_linear, lloyd_max_codebook, pack_nibbles, quantize_to_codes,
    dequantize_w4a4,
)
from y4w.runtime.config import EngineConfig
from y4w.runtime.engine import Engine
from y4w.runtime.linear import W4A4Linear
from y4w.backends.cpu import CPUBackend
from y4w.backends.cuda import CUDABackend


def test_bake_int8_preserves_values():
    torch.manual_seed(3)
    w = torch.randn(32, 64) * 0.3
    wi, ws = bake_int8_from_float(w)
    assert wi.shape == w.shape
    assert wi.dtype == torch.int8
    assert ws.shape == (32, 1)
    # Reconstruct: w ≈ wi * scale, relative error from int8 rounding is small.
    recon = wi.float() * ws
    rel = (recon - w).abs().mean() / w.abs().mean()
    assert rel.item() < 0.02, f"int8 bake error too large: {rel.item()}"


def test_bake_grouped_int8():
    torch.manual_seed(4)
    w = torch.randn(16, 128) * 0.2
    wi, ws = bake_int8_from_float(w, group_size=32)
    assert wi.shape == (16, 128)
    assert ws.shape == (16, 4)


def test_activation_int8_roundtrip():
    torch.manual_seed(5)
    x = torch.randn(8, 64) * 0.5
    xq, xs = quantize_activation_int8(x)
    assert xq.dtype == torch.int8
    recon = xq.float() * xs.unsqueeze(1)
    rel = (recon - x).abs().mean() / x.abs().mean()
    assert rel.item() < 0.02


def test_int8_linear_matches_dequant():
    torch.manual_seed(6)
    out, ink, m = 32, 64, 8
    w = torch.randn(out, ink) * 0.05
    wi, ws = bake_int8_from_float(w)
    x = torch.randn(m, ink) * 0.05
    y = int8_linear(x, wi, ws, use_dp4a=False)
    ref = x @ w.t()
    rel = (y - ref).abs().mean() / ref.abs().mean()
    assert y.shape == (m, out)
    assert rel.item() < 0.05, f"int8 linear drifted: {rel.item()}"


def test_int8_linear_with_bias_and_batched():
    torch.manual_seed(7)
    wi, ws = bake_int8_from_float(torch.randn(16, 32) * 0.05)
    bias = torch.randn(16) * 0.01
    x = torch.randn(2, 3, 32)
    y = int8_linear(x, wi, ws, bias, use_dp4a=False)
    assert y.shape == (2, 3, 16)


def test_bake_from_packed_uses_codebook():
    cb = lloyd_max_codebook()
    dense = torch.randn(16, 32) * 0.02
    scale = dense.abs().amax(1) / cb.abs().max()
    codes = quantize_to_codes(dense / scale.reshape(-1, 1), cb)
    packed = pack_nibbles(codes)
    wi, ws = bake_int8_from_packed(packed, scale, cb)
    assert wi.shape == (16, 32)
    assert wi.dtype == torch.int8


def test_w4a4_linear_bake_on_cpu_uses_packed():
    # CPU backend does not prefer baked int8 -> stays packed, minimal memory.
    be = CPUBackend()
    layer = W4A4Linear(32, 16, be)
    assert layer.bake_int8 is False
    assert layer._baked is False
    assert layer.memory_bytes() == layer.packed.numel() + layer.scales.numel() * 4


def test_w4a4_linear_explicit_bake():
    be = CPUBackend()
    layer = W4A4Linear(32, 16, be, bake_int8=True)
    layer.load_dense_weight(torch.randn(16, 32) * 0.05)
    layer.to_device("cpu")
    assert layer._baked is True
    assert layer.w_int8.dtype == torch.int8
    assert layer.memory_bytes() == layer.w_int8.numel() + layer.w_scales.numel() * 4


def test_cuda_prefers_baked_int8():
    assert CUDABackend().prefer_baked_int8 is True


def test_config_acceleration_flags():
    cfg = EngineConfig()
    assert cfg.optimize is True
    assert cfg.compile_model is True
    assert cfg.bake_int8 is True
    assert cfg.sdpa_attention is True
    assert cfg.int8_kv_cache is False


def test_engine_acceleration_status():
    e = Engine(EngineConfig(backend="cpu", seed=1))
    r = e.generate("hi", max_new_tokens=2)
    acc = r["acceleration"]
    assert "baked_int8" in acc
    assert "torch_compile" in acc
    assert "dp4a" in acc
    assert "sdpa" in acc
    assert acc["weight_bytes"] > 0


def test_memory_profile_disables_bake_and_compile():
    e = Engine(EngineConfig(backend="cpu", profile="memory", seed=1))
    acc = e.acceleration_status()
    assert acc["baked_int8"] is False
    assert acc["torch_compile"] is False


def test_no_optimize_disables_all():
    e = Engine(EngineConfig(backend="cpu", optimize=False, seed=1))
    acc = e.acceleration_status()
    assert acc["torch_compile"] is False
    assert acc["sdpa"] is False
