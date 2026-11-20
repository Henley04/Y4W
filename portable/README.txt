================================================================
 Y4W Windows 免安装整合包  (CUDA 13.2 / PyTorch 2.15.0)
================================================================

一、解压（重要）
----------------------------------------------------------------
本整合包因体积超过 GitHub 单文件 2 GiB 限制，被拆成多个分卷：
    Y4W-Portable-CUDA132-v0.2.1.z01
    Y4W-Portable-CUDA132-v0.2.1.z02
    ...
    Y4W-Portable-CUDA132-v0.2.1.zip   <- 最后一个是 .zip

请把【所有分卷】下载到同一个文件夹，然后：
  * 用 Windows 资源管理器：右键 .zip -> "全部解压缩"；或双击 .zip
  * 或用 7-Zip / WinRAR：打开 .zip，解压到当前文件夹
解压后得到一个目录，里面有 python\ 子目录和三个 .bat 脚本。

二、一键使用（双击即可）
----------------------------------------------------------------
  1-START-WebUI.bat    启动推理服务
                       - Gradio WebUI : http://127.0.0.1:8000/
                         * Tab「生成」     : 文本生歌
                         * Tab「ABC->MIDI」: 粘贴 YuE2 双声部 ABC，下载
                                             vocal.mid(歌声) 与
                                             accompaniment.mid(伴奏)
                       - REST API 文档: http://127.0.0.1:8000/docs
                         * POST /v1/abc2midi 也可直接转 MIDI
                       - 关闭：在该窗口按 Ctrl+C

  2-RUN-CLI.bat        命令行单次推理，按提示输入 prompt
                       也可在 bat 后附加参数，例如：
                       2-RUN-CLI.bat --max-new-tokens 32 --seed 7

  ABC 转 MIDI（命令行）:
                       python\python.exe -m y4w abc2midi --abc-file 谱.abc --out-dir .\out
                       生成 out\vocal.mid 与 out\accompaniment.mid

  3-CHECK-Env.bat      环境自检：Python / PyTorch / CUDA /
                       GPU 型号 / 四个后端可用性 / 完整诊断 JSON

三、后端选择
----------------------------------------------------------------
默认 --backend auto，优先级：CUDA -> TensorRT-RTX -> Vulkan -> CPU。
无 NVIDIA GPU 时自动回退到 CPU 参考后端，不会报错。
如需指定后端，可编辑 bat 里的 --backend 参数为：
  auto / cuda / tensorrt_rtx / vulkan / cpu

四、加载真实模型权重（可选）
----------------------------------------------------------------
本整合包默认运行内置的确定性合成网络（用于开箱验证）。
要跑真实的 YuE2-3B-OrbitQuant-W4A4 模型：
  1) 从 Hugging Face 下载模型（约 3 GB + VAE 0.53 GB）：
     hf-mirror.com/WaveCut/YuE2-3B-OrbitQuant-W4A4
  2) 把权重放到一个文件夹，例如  D:\models\YuE2-W4A4\
  3) 在 bat 命令末尾加上：--model-dir "D:\models\YuE2-W4A4"
     例如编辑 1-START-WebUI.bat 最后一行为：
     "%PY%" -m y4w serve --host 127.0.0.1 --port 8000 --model-dir "D:\models\YuE2-W4A4"

五、环境说明
----------------------------------------------------------------
  * Python 3.11 (x64, 内置于 python\)
  * PyTorch 2.15.0 + CUDA 13.2 (cu132 wheel，含 CUDA 运行库)
  * y4w 0.2.1 + [serve] 依赖 (fastapi / uvicorn / httpx / gradio)
  * 内置零依赖 abc2midi：ABC 乐谱按声部拆成歌声/伴奏两个 MIDI
  * 无需系统安装 Python，无需 pip install，解压即用
  * CUDA 加速需要 NVIDIA 驱动支持 CUDA 13.2（建议 580+ 驱动）；
    驱动过旧时 torch.cuda.is_available() 会是 False，自动回退 CPU
  * 完全离线运行，不联网

六、目录结构
----------------------------------------------------------------
  Y4W-Portable-CUDA132-v0.2.1\
    python\                    内置 Python 运行时 + 所有依赖
    1-START-WebUI.bat          一键服务
    2-RUN-CLI.bat              一键 CLI
    3-CHECK-Env.bat            环境自检
    README.txt                 本说明

源码与问题反馈: github.com/Henley04/Y4W
许可证: CC-BY-NC-4.0
