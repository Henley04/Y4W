# Y4W — Windows YuE2 OrbitQuant W4A4 推理引擎

[![Build & Test (Windows)](https://github.com/Henley04/Y4W/actions/workflows/build-test.yml/badge.svg)](https://github.com/Henley04/Y4W/actions/workflows/build-test.yml)

`y4w` 是面向 **Windows** 的 YuE2 / OrbitQuant W4A4 推理引擎。它实现了与
[`WaveCut/YuE2-3B-OrbitQuant-W4A4`](https://huggingface.co/WaveCut/YuE2-3B-OrbitQuant-W4A4)
一致的 4-bit 打包权重格式（Lloyd-Max codebook、per-row bfloat16 scale、RPBH 旋转），
并提供统一的多后端执行层。

## 后端支持

| 后端 | 状态 | 说明 |
|---|---|---|
| **CUDA** | ✅ 必须支持，已实现 | NVIDIA GPU，目标 SM89 (Ada/RTX40) 与 SM120 (Blackwell)。支持原生融合 kernel（可选编译）；未编译时回退到设备端 PyTorch 参考实现。 |
| **TensorRT-RTX** | ✅ 已实现 | NVIDIA TensorRT-RTX SDK，自动探测 `tensorrt_rtx` / `tensorrt` Python 模块，可加载序列化 engine。 |
| **Vulkan** | ✅ 已实现 | PyTorch Vulkan (SPIR-V) 设备，跨厂商 GPU 路径。 |
| **CPU** | ✅ 始终可用 | 纯 PyTorch 参考实现，是数值基准，也是无 GPU 环境（如 GitHub Actions Windows runner）的执行路径。 |

`backend="auto"` 按 **CUDA → TensorRT-RTX → Vulkan → CPU** 优先级自动选择当前可用的
最高性能后端；任一后端不可用都会**优雅降级**，绝不影响导入或测试。

## 为什么这样设计

上游模型只发布了 **Linux + 编译好的 `.so` CUDA kernel**（PyTorch 2.10 / CUDA 12.8，
且硬性校验 GPU 计算能力必须是 SM89/SM120）。Windows 上这些 `.so` 无法使用，因此 `y4w`：

1. 用**纯 PyTorch** 重写 W4A4 打包/反量化/矩阵乘，作为跨平台的数值参考与 CPU 回退；
2. 抽象出后端接口，把 CUDA / TensorRT-RTX / Vulkan 的差异隔离在各自模块内；
3. 可选编译 Windows 原生 CUDA kernel（`.pyd`），编译后自动加载，编译失败也不影响主流程；
4. 探测逻辑全部安全：没有对应 SDK / 驱动时只上报"不可用"，不抛异常、不阻断导入。

## 安装

```powershell
# 核心（CPU 参考路径始终可用）
pip install -e .

# 需要 GPU 时，先安装匹配你显卡的 CUDA 版 PyTorch，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu128

# 可选：模型下载与音频写出
pip install -e .[hub]
```

要求 Python ≥ 3.10。

## 快速开始

```powershell
# 查看后端 / 硬件 / kernel 诊断
python -m y4w --info
python -m y4w --list-backends

# 推理（未指定模型目录时运行内置的确定性合成 MoT 网络，用于冒烟/测试）
python -m y4w generate --backend auto --prompt "a rainy city at night" --max-new-tokens 16 --seed 42

# 指定后端
python -m y4w generate --backend cuda --model-dir .\YuE2-W4A4 --prompt prompt.json
```

Python API：

```python
from y4w import get_engine
from y4w.runtime.config import EngineConfig

cfg = EngineConfig(model_dir=r".\YuE2-W4A4", backend="auto", profile="fast", seed=42)
engine = get_engine(cfg)
print(engine.status())
result = engine.generate("style lyrics here", max_new_tokens=16)
```

## 加载真实 3B 权重

完整模型权重与独立的 VAE 由上游 HuggingFace 仓库分发（约 3 GB + 0.53 GB）。
下载后把目录传给 `--model-dir` / `EngineConfig(model_dir=...)`，引擎会读取
`orbitquant_manifest.json`，按名称把量化投影装入 `W4A4Linear`，并分发到所选后端。

> 注意：`y4w` 提供完整的量化 / 后端 / 运行时底座。合成网络仅用于 CI 与冒烟测试，
> 不代表音乐质量；高质量生成需要加载真实的 YuE2-3B 权重。

## 编译 Windows 原生 CUDA kernel（可选）

见 [`y4w/kernels/README_WINDOWS.md`](y4w/kernels/README_WINDOWS.md)。
简言之：在 **x64 Native Tools Command Prompt for VS** 中配置 `CUDA_HOME`，运行
`python setup_build.py build_ext --inplace` 生成 `.pyd`，引擎会自动加载。

## 测试

```powershell
pip install -e .[test]
python -m pytest y4w/tests -v
```

GitHub Actions 在 `windows-latest`（Python 3.10/3.11/3.12）上自动完成安装、构建与测试，
无需 GPU。GPU 执行在带 SM89/SM120 显卡的自托管 runner 上验证。

## 致谢与许可

量化格式来自 OrbitQuant 与 `WaveCut/YuE2-3B-OrbitQuant-W4A4`（CC-BY-NC-4.0）。
本项目遵循相同的非商业研究许可。
