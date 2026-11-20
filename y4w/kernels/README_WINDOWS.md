# Building the optional native kernel on Windows

The `y4w` Python package runs everywhere without this extension (portable Torch
W4A4 math). Building it is optional and only needed for the fused decode path.

## Prerequisites (Windows 10/11, x64)

1. Visual Studio 2022 Build Tools with **Desktop development with C++**.
2. CUDA Toolkit 12.x (includes `nvcc`).
3. Python 3.10+ with PyTorch installed (CUDA build recommended).

## Build

From a **x64 Native Tools Command Prompt for VS**:

```bat
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
python setup_build.py build_ext --inplace
```

This produces `_w4a4_native.cp3XX-win_amd64.pyd` next to this file. The
registry / kernel loader import it automatically when present; if import fails
for any reason the engine transparently uses the reference path.

## SPIR-V / Vulkan

The Vulkan backend uses PyTorch's Vulkan device for binding and the portable
reference math for correctness. A production deployment can replace
`VulkanBackend.linear` with a dispatch to a SPIR-V shader; the hook point is
documented in `y4w/backends/vulkan.py`.

## CI note

GitHub-hosted Windows runners have no GPU, so the workflow validates the
Windows build configuration (syntax / import / dry compile of host code) and
runs the whole test-suite on the CPU backend. GPU execution is validated on a
self-hosted runner with an SM89/SM120 card (see root README).
