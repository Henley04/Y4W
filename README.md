# Y4W — Windows YuE2 OrbitQuant W4A4 推理引擎

[![Build & Test (Windows)](https://github.com/Henley04/Y4W/actions/workflows/build-test.yml/badge.svg)](https://github.com/Henley04/Y4W/actions/workflows/build-test.yml)
[![Release](https://github.com/Henley04/Y4W/actions/workflows/release.yml/badge.svg)](https://github.com/Henley04/Y4W/actions/workflows/release.yml)

`y4w` 是面向 **Windows** 的 YuE2 / OrbitQuant W4A4 推理引擎。它实现了与
[`WaveCut/YuE2-3B-OrbitQuant-W4A4`](https://huggingface.co/WaveCut/YuE2-3B-OrbitQuant-W4A4)
一致的 4-bit 打包权重格式（Lloyd-Max codebook、per-row bfloat16 scale、RPBH 旋转），
并提供统一的多后端执行层。

## 三种使用方式（任选其一）

| 方式 | 命令 | 适合场景 |
|---|---|---|
| **CLI** | `y4w generate ...` | 单次推理、脚本、批处理 |
| **一键服务** | `y4w serve` | 一个进程同时开 Gradio WebUI + REST API |
| **Python API** | `from y4w import get_engine` | 集成到你自己的程序 |

WebUI、API、CLI 三者**共用同一个引擎与后端**，结果一致。

---

## 安装

```powershell
# 核心引擎（CPU 参考路径始终可用）
pip install -e .

# 一键服务所需依赖（Gradio WebUI + REST API）
pip install -e .[serve]

# 需要 GPU 时，先装匹配显卡的 CUDA 版 PyTorch，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

也可直接安装 Release 里发布的 wheel（见文末「Release」）。要求 Python ≥ 3.10。

---

## 1. CLI 推理

```powershell
# 硬件 / 后端 / kernel 诊断
y4w --info
y4w --list-backends

# 推理（未指定 --model-dir 时运行内置合成网络，用于冒烟/演示）
y4w generate --backend auto --prompt "a rainy city at night" --max-new-tokens 16 --seed 42

# 指定后端与模型目录
y4w generate --backend cuda --model-dir .\YuE2-W4A4 --prompt prompt.json --output result.json

# 等价于 python -m y4w ...
python -m y4w generate --backend cpu --prompt test --max-new-tokens 4
```

常用参数：`--backend`（auto/cuda/tensorrt_rtx/vulkan/cpu）、`--profile`
（reference/fused/fast/memory）、`--max-new-tokens`、`--temperature`、`--seed`、
`--kv-cache-dtype`、`--output`。

## 2. 一键服务：WebUI + API + CLI

```powershell
# 默认监听 127.0.0.1:8000，同时提供 WebUI、REST API、OpenAPI 文档
y4w serve

# 自定义地址/端口，加载真实模型
y4w serve --host 0.0.0.0 --port 7860 --model-dir .\YuE2-W4A4

# 只要 API，不挂 WebUI（例如部署在无图形界面环境）
y4w serve --no-webui

# 让 auto 模式优先尝试 TensorRT-RTX（默认优先 CUDA，详见下文）
y4w serve --backend-priority tensorrt
```

启动后一个端口同时得到：

| 入口 | 地址 | 说明 |
|---|---|---|
| **Gradio WebUI** | http://127.0.0.1:8000/ | 文本框输入 prompt，滑块调 tokens/temperature/seed |
| **REST API** | http://127.0.0.1:8000/v1/generate | JSON 请求/响应 |
| **API 文档** | http://127.0.0.1:8000/docs | Swagger UI，可在线调试 |

REST API 端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活探针，返回后端与 WebUI 状态 |
| GET | `/v1/status` | 引擎状态、后端能力、配置、manifest 摘要 |
| GET | `/v1/backends` | 后端可用性、优先级策略、auto 顺序 |
| POST | `/v1/generate` | 推理，`{prompt, max_new_tokens, temperature, seed}` |

curl 示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a rainy city at night","max_new_tokens":16,"temperature":0.8,"seed":42}'
```

> **说明**：Gradio 是可选依赖。未安装时 `y4w serve` 仍会启动纯 REST API
> 并在终端提示 WebUI 不可用，不会报错。生成请求在服务内串行执行（后端持有
> CUDA context / 原生 kernel，非线程安全）。

## 3. Python API

```python
from y4w import get_engine
from y4w.runtime.config import EngineConfig

cfg = EngineConfig(model_dir=r".\YuE2-W4A4", backend="auto", profile="fast", seed=42)
engine = get_engine(cfg)
print(engine.status())
result = engine.generate("style lyrics here", max_new_tokens=16)
print(result["tokens"], result["backend"])
```

---

## 后端优先级：为什么默认 CUDA 在前，以及如何改

`backend="auto"` 会按优先级选择当前可用的后端。默认顺序：

```
CUDA → TensorRT-RTX → Vulkan → CPU
```

**CUDA 默认优先于 TensorRT-RTX，是刻意设计**，原因很具体：

- **CUDA 后端是唯一加载原生融合 W4A4 kernel 的路径**（`load_native_kernel()`
  → `w4a4_linear`），它直接对 packed nibble 权重做 GEMV，避免解包开销。
- **TensorRT-RTX 后端目前负责 SDK 探测与序列化 engine 加载**，其 `linear()`
  在没有预编译 engine 时走的是与 CUDA 相同的 PyTorch W4A4 数学路径。
  若把它排在 CUDA 前面，默认反而会选中功能更弱的后端。
- 因此「能力等级」上 TRT-RTX 虽标记为 COMPILED（高于 CUDA 的 ACCELERATED），
  但那是预留给未来预编译 engine 的；实际执行能力当前 CUDA 更强。

如果你的机器上 TensorRT-RTX 已编译好对应 engine、希望优先用它，可切换：

```powershell
# 方式一：CLI 参数（仅对该次 serve 生效）
y4w serve --backend-priority tensorrt

# 方式二：环境变量（对所有 y4w 进程生效）
$env:Y4W_BACKEND_PRIORITY = "tensorrt"
y4w serve
```
切到 `tensorrt` 后顺序变为 `TensorRT-RTX → CUDA → Vulkan → CPU`。无论哪种策略，
**CPU 始终是最后的回退**，任一 GPU 后端不可用都会优雅降级，不会阻断启动。

---

## 后端支持一览

| 后端 | 状态 | 说明 |
|---|---|---|
| **CUDA** | ✅ 必须支持，已实现 | NVIDIA GPU，目标 SM89 (Ada/RTX40) 与 SM120 (Blackwell)。支持原生融合 kernel（可选编译）；未编译时回退设备端 PyTorch。 |
| **TensorRT-RTX** | ✅ 已实现 | 探测 `tensorrt_rtx`/`tensorrt` 模块，可加载序列化 engine；无 engine 时走 PyTorch W4A4 路径。 |
| **Vulkan** | ✅ 已实现 | PyTorch Vulkan (SPIR-V) 设备，跨厂商 GPU 路径。 |
| **CPU** | ✅ 始终可用 | 纯 PyTorch 参考实现，数值基准，也是无 GPU 环境的执行路径。 |

---

## 加载真实 3B 权重

完整权重与独立 VAE 由上游 HuggingFace 分发（约 3 GB + 0.53 GB）。下载后把目录传给
`--model-dir` / `EngineConfig(model_dir=...)`，引擎读取 `orbitquant_manifest.json`，
按名称把量化投影装入 `W4A4Linear` 并分发到所选后端。

> 未加载真实权重时运行的是内置确定性合成 MoT 网络，仅用于 CI 与冒烟测试，
> 不代表音乐质量。

## 编译 Windows 原生 CUDA kernel（可选）

见 [`y4w/kernels/README_WINDOWS.md`](y4w/kernels/README_WINDOWS.md)。在
**x64 Native Tools Command Prompt for VS** 中配置 `CUDA_HOME`，运行
`python setup_build.py build_ext --inplace` 生成 `.pyd`，引擎会自动加载。

## 测试与 CI

```powershell
pip install -e .[test]
python -m pytest y4w/tests -v
```

GitHub Actions 在 `windows-latest`（Python 3.10/3.11/3.12 + service smoke）自动完成
安装、测试、打包与服务验证，无需 GPU。GPU 执行在带 SM89/SM120 显卡的机器验证。

## Release（构建物下载）

每次推送 `v*` 标签（如 `git tag v0.1.0; git push origin v0.1.0`）或手动运行
`release.yml`，都会在 Windows 上构建 `sdist` + `wheel` 并上传到
[Releases](https://github.com/Henley04/Y4W/releases)。直接安装发布物：

```powershell
pip install https://github.com/Henley04/Y4W/releases/download/v0.1.0/y4w-0.1.0-py3-none-any.whl
```

## 致谢与许可

量化格式来自 OrbitQuant 与 `WaveCut/YuE2-3B-OrbitQuant-W4A4`（CC-BY-NC-4.0）。
本项目遵循相同的非商业研究许可。
