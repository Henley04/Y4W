"""Optional native-extension build for y4w (used on Windows with CUDA).

Kept separate from pyproject.toml so a normal `pip install` never requires nvcc.
See y4w/kernels/README_WINDOWS.md for the Windows build instructions.
"""
from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CudaExtension(Extension):
    def __init__(self, name: str, src: str) -> None:
        super().__init__(name, sources=[src])
        self.src = src


class CudaBuildExt(build_ext):
    def build_extension(self, ext: CudaExtension) -> None:  # type: ignore[override]
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths

        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME is not set; install the CUDA Toolkit")

        src = Path(ext.src)
        obj = Path(self.build_temp) / (src.stem + ".o")
        obj.parent.mkdir(parents=True, exist_ok=True)

        includes = " ".join(f'-I"{p}"' for p in include_paths())
        defines = "-DWIN32_LEAN_AND_MEAN" if os.name == "nt" else ""
        nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
        ext_full = Path(self.get_ext_fullpath(ext.name))
        out = ext_full.with_suffix(".pyd" if os.name == "nt" else ext_full.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)

        arch = os.environ.get("Y4W_CUDA_ARCH", "sm_89")
        compile_cmd = f'"{nvcc}" -c {src} -o {obj} {includes} {defines} -gencode=arch=compute_{arch[3:]},code={arch} -O3 -Xcompiler /EHsc'  # noqa: E501
        link_libs = " ".join(f'"{p}"' for p in library_paths())
        link_cmd = (
            f'"{nvcc}" {obj} -shared -o {out} '
            f'-L"{os.path.join(CUDA_HOME, "lib", "x64" if os.name == "nt" else "lib64")}" '
            f'-lcudart {link_libs}'
        )
        self.spawn(compile_cmd if os.name != "nt" else ["cmd", "/c", compile_cmd])
        self.spawn(link_cmd if os.name != "nt" else ["cmd", "/c", link_cmd])


if __name__ == "__main__":
    setup(
        name="y4w_native",
        version="0.1.0",
        ext_modules=[CudaExtension("y4w.kernels._w4a4_native", "y4w/kernels/w4a4_native.cu")],
        cmdclass={"build_ext": CudaBuildExt},
    )
