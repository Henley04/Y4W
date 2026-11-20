"""Quantization primitives: pack/unpack, codebook, round-trip, matmul."""
from __future__ import annotations

import pytest
import torch

from y4w.quant import (
    BITS, LEVELS, lloyd_max_codebook, quantize_to_codes, dequantize_codes,
    pack_nibbles, unpack_nibbles, dequantize_w4a4, quantized_matmul, w4a4_linear,
)


def test_codebook_shape_and_order():
    cb = lloyd_max_codebook()
    assert cb.shape == (LEVELS,) == (16,)
    assert cb.dtype == torch.float32
    assert torch.all(cb[:-1] <= cb[1:]), "codebook must be ascending"
    assert BITS == 4


def test_lloyd_max_is_deterministic():
    a, b = lloyd_max_codebook(), lloyd_max_codebook()
    assert torch.equal(a, b)


def test_lloyd_max_reasonable_for_normal():
    # A 4-bit Lloyd-Max quantizer for N(0,1) has extremes roughly in [-3.5, 3.5].
    cb = lloyd_max_codebook()
    assert 2.0 < float(cb.max()) < 4.0
    assert -4.0 < float(cb.min()) < -2.0
    assert torch.isfinite(cb).all()


def test_pack_unpack_nibbles_roundtrip():
    codes = torch.tensor([[0, 1, 15, 7, 8, 2], [3, 4, 5, 6, 10, 11]], dtype=torch.uint8)
    packed = pack_nibbles(codes)
    assert packed.shape == (2, 3)
    assert packed.dtype == torch.uint8
    assert packed[0, 0].item() == 0x10  # low nibble 0, high nibble 1
    assert packed[0, 1].item() == 0x7F  # low 15, high 7
    back = unpack_nibbles(packed)
    assert torch.equal(back, codes)


def test_pack_requires_even_last_dim():
    with pytest.raises(ValueError):
        pack_nibbles(torch.zeros(2, 3, dtype=torch.uint8))


def test_unpack_rejects_non_uint8():
    with pytest.raises(TypeError):
        unpack_nibbles(torch.zeros(2, 3, dtype=torch.int32))


def test_unpack_validates_out_features():
    packed = torch.zeros(2, 3, dtype=torch.uint8)
    with pytest.raises(Exception):
        unpack_nibbles(packed, out_features=5)


def test_quantize_dequantize_roundtrip():
    cb = lloyd_max_codebook()
    values = torch.linspace(-3, 3, 100)
    codes = quantize_to_codes(values, cb)
    assert codes.dtype == torch.uint8
    assert int(codes.min()) >= 0 and int(codes.max()) < LEVELS
    deq = dequantize_codes(codes, cb)
    # Each value is mapped to its nearest codebook entry. The outermost
    # Lloyd-Max level for N(0,1) is ~2.76, so values near +/-3 clip with ~0.24
    # error; this is expected for a 4-bit quantizer at the distribution tail.
    err = (deq - values).abs().max().item()
    assert err < 0.35, f"round-trip error too large: {err}"
    # All in-range values map exactly to a codebook entry.
    assert torch.equal(deq, cb[codes.long()])


def test_dequantize_w4a4_applies_row_scale():
    cb = lloyd_max_codebook()
    codes = torch.tensor([[0, 15, 1, 14]], dtype=torch.uint8)  # 1 row, 4 codes
    packed = pack_nibbles(codes)
    scales = torch.tensor([2.0])
    w = dequantize_w4a4(packed, scales, cb)
    assert w.shape == (1, 4)
    expected = dequantize_codes(codes, cb) * 2.0
    assert torch.allclose(w, expected, atol=1e-6)


def test_dequantize_w4a4_scale_mismatch_raises():
    packed = torch.zeros(3, 2, dtype=torch.uint8)
    with pytest.raises(Exception):
        dequantize_w4a4(packed, torch.ones(2))


def test_quantized_matmul_shape_and_reference():
    torch.manual_seed(0)
    N, K, M = 8, 16, 3
    dense = torch.randn(N, K) * 0.1
    cb = lloyd_max_codebook()
    codes = quantize_to_codes(dense, cb)
    packed = pack_nibbles(codes)
    scales = torch.ones(N)
    x = torch.randn(M, K)

    # With activation_bits=0 the reference is pure dequant-weight matmul.
    y = quantized_matmul(x, packed, scales, cb, activation_bits=0)
    assert y.shape == (M, N)
    expected = x @ dequantize_w4a4(packed, scales, cb).t()
    assert torch.allclose(y, expected, atol=1e-5)


def test_w4a4_linear_close_to_dense():
    # With per-row scaling (as load_dense_weight does), a small-valued dense
    # weight is normalized into the codebook range and reconstructs closely.
    torch.manual_seed(1)
    N, K, M = 16, 32, 4
    dense = torch.randn(N, K) * 0.05
    cb = lloyd_max_codebook()
    cb_max = cb.abs().max()
    row_amax = dense.abs().amax(dim=1).clamp_min(1e-12)
    scales = row_amax / cb_max
    packed = pack_nibbles(quantize_to_codes(dense / scales[:, None], cb))
    x = torch.randn(M, K) * 0.05
    y_q = quantized_matmul(x, packed, scales, cb, activation_bits=0)
    y_d = x @ dense.t()
    rel = (y_q - y_d).abs().mean() / y_d.abs().mean().clamp_min(1e-9)
    assert rel.item() < 0.1, f"quantized matmul drifted: {rel.item()}"


def test_activation_quantization_changes_output_but_keeps_shape():
    torch.manual_seed(2)
    N, K, M = 8, 16, 2
    packed = torch.zeros(N, K // 2, dtype=torch.uint8)
    scales = torch.ones(N)
    x = torch.randn(M, K)
    y4 = quantized_matmul(x, packed, scales, activation_bits=4)
    y0 = quantized_matmul(x, packed, scales, activation_bits=0)
    assert y4.shape == y0.shape
    assert not torch.equal(y4, y0)
