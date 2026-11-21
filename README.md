# Y4W — Windows YuE2 OrbitQuant W4A4 推理引擎

[![Build & Test (Windows)](https://github.com/Henley04/Y4W/actions/workflows/build-test.yml/badge.svg)](https://github.com/Henley04/Y4W/actions/workflows/build-test.yml)
[![Release](https://github.com/Henley04/Y4W/actions/workflows/release.yml/badge.svg)](https://github.com/Henley04/Y4W/actions/workflows/release.yml)
[![Portable Bundle](https://github.com/Henley04/Y4W/actions/workflows/portable.yml/badge.svg)](https://github.com/Henley04/Y4W/actions/workflows/portable.yml)

`y4w` 是面向 **Windows** 的 YuE2 / OrbitQuant W4A4 推理引擎。它实现了与
[`WaveCut/YuE2-3B-OrbitQuant-W4A4`](https://huggingface.co/WaveCut/YuE2-3B-OrbitQuant-W4A4)
一致的 4-bit 打包权重格式（Lloyd-Max codebook、per-row bfloat16 scale、RPBH 旋转），
并提供统一的多后端执行层。模型的 **ModelScope 镜像**在
[`syxppp/YuE2-3B-OrbitQuant-W4A4`](https://www.modelscope.cn/models/syxppp/YuE2-3B-OrbitQuant-W4A4)，
下载与推理均走 ModelScope（不依赖 HuggingFace）。

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

# 需要从 ModelScope 下载模型时（可选，含 modelscope SDK）
pip install -e .[hub]

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
| **Gradio WebUI** | http://127.0.0.1:8000/ | 两个 Tab：文本生成；ABC→MIDI（双声部拆两个文件下载） |
| **REST API** | http://127.0.0.1:8000/v1/generate | JSON 请求/响应 |
| **API 文档** | http://127.0.0.1:8000/docs | Swagger UI，可在线调试 |

REST API 端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活探针，返回后端与 WebUI 状态 |
| GET | `/v1/status` | 引擎状态、后端能力、配置、manifest 摘要 |
| GET | `/v1/backends` | 后端可用性、优先级策略、auto 顺序 |
| POST | `/v1/generate` | 推理，`{prompt, max_new_tokens, temperature, seed}` |
| POST | `/v1/abc2midi` | ABC 乐谱 → `{vocal_midi_b64, accompaniment_midi_b64, ...}` |

curl 示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a rainy city at night","max_new_tokens":16,"temperature":0.8,"seed":42}'
```

> **说明**：Gradio 是可选依赖。未安装时 `y4w serve` 仍会启动纯 REST API
> 并在终端提示 WebUI 不可用，不会报错。生成请求在服务内串行执行（后端持有
> CUDA context / 原生 kernel，非线程安全）。

## 2.1 ABC → MIDI：歌声 / 伴奏双文件导出

YuE2 的符号规划输出的是**双声部 ABC 乐谱**（`V: Vocal` 歌声带引号和弦、
`V: Ins` 伴奏）。Y4W 内置**零依赖**的 `y4w.abc2midi`（纯标准库，不需要
`abc2midi` 二进制、music21 或 mido），按声部拆成两个标准 MIDI 文件：

- `vocal.mid` — 歌声旋律轨
- `accompaniment.mid` — 伴奏轨

输出为 SMF format 1（480 PPQ），含速度、拍号、调号、program change 与
note on/off。三种入口任选：

```bash
# CLI：推荐用 .abc 文件
y4w abc2midi --abc-file score.abc --out-dir ./out

# 也可以内联；bash 需用 $'...' 才会把 \n 解释成换行
y4w abc2midi --abc $'X:1\nK:C\nV: Vocal C2E2|\nV: Ins C,2G,2|\n' --out-dir ./out
# Windows PowerShell 建议直接用 --abc-file，避免引号转义问题

# WebUI：y4w serve 后切到 "ABC → MIDI" Tab，粘贴 ABC，点“转换为 MIDI”下载两个文件

# REST API
curl -X POST http://127.0.0.1:8000/v1/abc2midi \
  -H "Content-Type: application/json" -d '{"abc":"X:1\nK:C\nV: Vocal C2E2|\nV: Ins C,2G,2|\n"}'
```

```python
from y4w.abc2midi import abc_to_midi, abc_to_midi_files

r = abc_to_midi(abc_text)               # 得到 raw bytes（API 用）
r.to_dict(base64=True)                   # JSON 友好的 base64 结果
abc_to_midi_files(abc_text, "./out")     # 直接写出两个 .mid 文件
```

> 说明：转换器只解析音符时值与音高，引号内的和弦标记（如 `"C"`、`"Am7"`）
> 作为伴奏提示被忽略，不写入音符轨；多首曲子只转换第一首（`X:1`）。
> 解析不出任何声部/音符时，REST 返回 `422`、CLI 以非零码退出，不会静默产出空 MIDI。

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

## 推理加速与显存优化

`y4w` 默认开启全部可用加速（`EngineConfig.optimize=True`）。关键点是：
**原始 W4A4 路径每次前向都要把 4-bit 权重解包还原成 fp32，实测单次
2048×2048 投影要 30~55ms，比 fp16 GEMM 慢 10~25 倍**——4-bit 只省了存储，没省算力。
优化的核心是**把这一步从"每次前向"移到"加载时只做一次"**。

| 加速手段 | 开关 | 收益 | 显存 |
|---|---|---|---|
| **烘焙 int8 权重** | `bake_int8=True` | 消除重复解包，实测 **7~15× 加速**；CUDA 上走 DP4A | 1 B/权重（fp16 的一半）|
| **DP4A GEMM** | CUDA 自动 | `torch._int_mm` 原生 int8 矩阵乘（SM89/SM120）| 同上 |
| **torch.compile + CUDA Graph** | `compile_model=True` | 算子融合 + 消除 kernel launch 开销，`mode="reduce-overhead"` | 少量编译缓存 |
| **SDPA / FlashAttention** | `sdpa_attention=True` | `F.scaled_dot_product_attention` 高效注意力 | 更低 |
| **int8 KV cache** | `int8_kv_cache=True` / `--int8-kv` | KV 缓存显存减半 | 长序列省 ~50% KV |

**显存优先级**：`--profile memory` 会关闭烘焙与编译，保留最紧凑的 packed W4A4
（**0.5 B/权重**，最小显存），代价是每次前向都要解包（慢）。默认 `--profile fast`
则用烘焙 int8（1 B/权重）换取 7~15× 加速。两者都远小于 fp16 的 2 B/权重。

所有加速在**无 GPU / 无编译器环境自动降级**，CI（Windows runner，CPU-only）全程通过。

```powershell
# 默认：全速（烘焙 int8 + compile + SDPA + DP4A 若可用）
y4w generate --backend auto --prompt "..." --max-new-tokens 16

# 最小显存（保留 packed W4A4，不烘焙不编译）
y4w generate --backend auto --profile memory --prompt "..."

# int8 KV cache（长序列省显存）
y4w serve --int8-kv --port 8000

# 关闭全部加速（纯参考路径，用于数值对照）
y4w generate --no-optimize --backend cpu

# 查看当前生效的加速状态
y4w generate --backend cpu --max-new-tokens 1 | Select-String acceleration
```

`--info` / `/v1/status` 会报告 `acceleration` 字段（`baked_int8`、`torch_compile`、
`dp4a`、`sdpa`、`int8_kv_cache`、`weight_bytes`），方便确认实际启用了哪些加速。

量化方案说明：烘焙采用 **per-output-row 对称 int8 + per-token 激活 int8**，
int8 舍入相对误差 <2%，端到端投影相对误差 <5%（见 `y4w/tests/test_accel.py`）。
这与官方 `fast` 模式的 INT8 KV / DP4A GEMV 思路一致。真实 3B 权重的精度/显存
需在带 SM89/SM120 显卡的机器上实测（见下节）。

可复现的延迟/显存对比：`python tools/benchmark.py`。

---

## 加载真实 3B 权重（ModelScope）

权重托管在 ModelScope 镜像 [`syxppp/YuE2-3B-OrbitQuant-W4A4`](https://www.modelscope.cn/models/syxppp/YuE2-3B-OrbitQuant-W4A4)（约 3 GB，含 `orbitquant_manifest.json`、量化 `model.safetensors` 与 tokenizer）。用内置命令一键下载：

```powershell
pip install -e .[hub]
y4w download-model --local-dir D:\models\YuE2-W4A4
# CLI / serve 用 --model-dir 指向该目录即可：
y4w generate --model-dir D:\models\YuE2-W4A4 --prompt "..."
```

也可用 Python API：

```python
from y4w.hub import download_model
path = download_model(local_dir=r"D:\models\YuE2-W4A4")  # 返回快照目录
```

或直接用 ModelScope CLI：`modelscope download --model syxppp/YuE2-3B-OrbitQuant-W4A4 --local_dir D:\models\YuE2-W4A4`。
模型 ID 可用环境变量 `Y4W_MODEL_ID` 覆盖。`download-model` 默认只拉取引擎所需的运行时文件；如需完整源码/内核树，去掉该限制即可。

引擎读取目录里的 `orbitquant_manifest.json`，按名称把量化投影装入 `W4A4Linear` 并分发到所选后端。

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

### 免安装整合包（解压即用，CUDA 13.2）

同一个 Release 里还有一个**完整的 Windows 运行环境整合包**
`Y4W-Portable-CUDA132-*.zip*`，内置 Python 3.11 + PyTorch 2.15.0 (cu132) +
CUDA 13.2 运行库 + `y4w[serve]`，**无需安装 Python、无需 pip、解压即用**：

1. 下载 Release 里**全部**分卷（`.z01` / `.z02` / … / `.zip`）到同一文件夹
   （因超过 GitHub 2 GiB 单文件上限而分卷，最后一个是 `.zip`）；
2. 右键最后的 `.zip` →「全部解压缩」，或用 7-Zip/WinRAR 解压；
3. 进入解压目录，双击：

| 脚本 | 作用 |
|---|---|
| `1-START-WebUI.bat` | 一键启动服务：WebUI `http://127.0.0.1:8000/` + REST API `/docs` |
| `2-RUN-CLI.bat` | 命令行单次推理，按提示输入 prompt |
| `3-CHECK-Env.bat` | 自检 Python/PyTorch/CUDA/GPU 型号/后端可用性 |

- 默认 `--backend auto`：CUDA → TensorRT-RTX → Vulkan → CPU，无 GPU 自动回退。
- CUDA 加速需支持 CUDA 13.2 的 NVIDIA 驱动（建议 580+）；驱动过旧会回退 CPU。
- 默认跑内置合成网络；加载真实权重见解压包内 `README.txt`。
- 整合包由 `portable.yml` 在同标签上构建并上传，可单独手动重跑。

## 致谢与许可

量化格式来自 OrbitQuant 与 `WaveCut/YuE2-3B-OrbitQuant-W4A4`（CC-BY-NC-4.0）。
本项目遵循相同的非商业研究许可。
