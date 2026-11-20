"""Benchmark W4A4 compute paths and the baked-int8 acceleration.

Run:  python tools/benchmark.py
Works on CPU (portable path) and CUDA (DP4A path). Reports per-forward latency
for a realistic 2048x2048 projection and the memory footprint of each weight
layout, so you can see the ~10-15x gain from baking once instead of decoding
packed weights on every forward pass.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from y4w.quant import lloyd_max_codebook
from y4w.quant.packed import pack_nibbles, quantized_matmul, dequantize_w4a4
from y4w.quant.int8 import bake_int8_from_packed, int8_linear


def bench(fn, n=40, warm=8):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")

    out, in_f, m = 2048, 2048, 8
    cb = lloyd_max_codebook().to(device)
    dense = torch.randn(out, in_f, device=device) * 0.02
    scale = dense.abs().amax(1) / cb.abs().max()
    codes = torch.round(dense / scale.reshape(-1, 1)).clamp(-8, 7).to(torch.int64) + 8
    packed = pack_nibbles(codes)
    wi, ws = bake_int8_from_packed(packed, scale, cb)
    w16 = dequantize_w4a4(packed, scale, cb).half().contiguous()
    x = torch.randn(m, in_f, device=device) * 0.02
    x16 = x.half()

    results = {
        "W4A4 decode-every-forward (fp32)":
            lambda: quantized_matmul(x, packed, scale, cb, activation_bits=4),
        "baked int8 + portable GEMM":
            lambda: int8_linear(x, wi, ws, use_dp4a=False),
        "baked int8 + DP4A (CUDA _int_mm)":
            lambda: int8_linear(x, wi, ws, use_dp4a=True),
        "fp16 pre-dequant GEMM":
            lambda: x16 @ w16.t(),
    }
    timings = {k: bench(v) for k, v in results.items()}
    base = timings["W4A4 decode-every-forward (fp32)"]
    print("\n=== per-forward latency (lower is better) ===")
    for k, ms in sorted(timings.items(), key=lambda kv: kv[1]):
        print(f"  {k:<40}{ms:8.3f} ms  {base/ms:6.1f}x vs decode-every-forward")

    print("\n=== compute-weight memory (bytes per weight) ===")
    print("  packed W4A4 (archive) : 0.50")
    print("  baked int8 (compute)  : 1.00   <- 1/2 of fp16, DP4A-ready")
    print("  fp16                  : 2.00")
    print("  fp32                  : 4.00")


if __name__ == "__main__":
    main()
