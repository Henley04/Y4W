// y4w native W4A4 kernel (Windows / CUDA).
//
// Minimal, self-contained fused dequantize + GEMV for the OrbitQuant W4A4
// layout: two 4-bit codes per uint8 byte, a 16-entry float codebook and a
// per-output-row bfloat16 scale. This is a reference-grade kernel intended to
// be built with nvcc on Windows (see README_WINDOWS.md); the Python package
// works identically without it via the portable Torch path.
//
// It deliberately avoids CUTLASS/ATen-internal headers so it compiles with a
// plain CUDA Toolkit + PyTorch include dirs on MSVC.

#include <cuda_runtime.h>
#include <cstdint>

namespace y4w {

// Decode a 4-bit unsigned code to a signed codebook index (-8..7) then look up
// a float codebook value. Codebook is supplied from the host as a device ptr.
__device__ __forceinline__ float decode_code(uint8_t packed_byte, bool high,
                                             const float* __restrict__ codebook) {
    uint32_t code = high ? ((packed_byte >> 4) & 0x0Fu) : (packed_byte & 0x0Fu);
    return codebook[code];
}

// One thread per output row: dot(row of activations, dequantized weight row).
// Shapes: x [K], packed [N, K/2], scales [N], codebook [16], out [N].
__global__ void w4a4_gemv_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ codebook,
    float* __restrict__ out,
    int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const uint8_t* wrow = packed + (size_t)row * (K / 2);
    float acc = 0.0f;
    for (int j = 0; j < K / 2; ++j) {
        uint8_t byte = wrow[j];
        acc += x[2 * j] * decode_code(byte, false, codebook);
        acc += x[2 * j + 1] * decode_code(byte, true, codebook);
    }
    out[row] = acc * scales[row];
}

// Host launcher: x [M,K] float, packed [N,K/2] uint8, scales [N], cb [16],
// out [M,N] float. Uses one kernel launch per input row (decode path, M small).
void w4a4_gemv(const float* x, const uint8_t* packed, const float* scales,
               const float* codebook, float* out, int M, int N, int K,
               cudaStream_t stream) {
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;
    for (int m = 0; m < M; ++m) {
        w4a4_gemv_kernel<<<blocks, threads, 0, stream>>>(
            x + (size_t)m * K, packed, scales, codebook, out + (size_t)m * N, K, N);
    }
}

}  // namespace y4w
