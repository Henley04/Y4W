"""Gradio WebUI mounted by ``y4w serve``.

Kept in its own module and imported lazily so the REST API and the whole test
suite work on machines (and CI runners) without Gradio installed.

Two tabs:
    * 生成 - text-to-score inference via the shared engine.
    * ABC → MIDI - convert a YuE2 two-voice ABC score into separate
      ``vocal.mid`` and ``accompaniment.mid`` downloads.
"""
from __future__ import annotations

import json
import tempfile
from typing import Any, Callable


def build_gradio_interface(generate_fn: Callable[..., dict[str, Any]],
                           engine: Any = None):
    """Build a Gradio Blocks UI bound to the shared thread-safe generator."""
    import gradio as gr

    from ..abc2midi import abc_to_midi_files

    backend = getattr(getattr(engine, "backend", None), "name", "auto")

    def _run(prompt, max_new_tokens, temperature, seed):
        try:
            result = generate_fn(prompt, max_new_tokens, temperature, seed)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:  # surface errors in the UI instead of crashing
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"},
                              ensure_ascii=False, indent=2)

    def _abc_to_midi(abc_text: str):
        """Convert ABC text -> two downloadable MIDI files + info JSON."""
        if not abc_text or not abc_text.strip():
            raise gr.Error("请先粘贴 ABC 乐谱文本")
        out_dir = tempfile.mkdtemp(prefix="y4w_abc_")
        try:
            data = abc_to_midi_files(abc_text, out_dir)
        except Exception as exc:
            raise gr.Error(f"ABC 转换失败: {type(exc).__name__}: {exc}")
        info = {
            "声部": data["voices"],
            "歌声音符数": data["vocal_note_count"],
            "伴奏音符数": data["accompaniment_note_count"],
            "时长(秒)": data["duration_seconds"],
            "调号": data["key"],
            "拍号": data["meter"],
            "速度(BPM)": data["tempo_bpm"],
            "警告": data["warnings"],
        }
        return (
            data["vocal_path"],
            data["accompaniment_path"],
            json.dumps(info, ensure_ascii=False, indent=2),
        )

    sample_abc = (
        "X:1\nT:示例\nM:4/4\nL:1/16\nQ:1/4=88\n"
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
        'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
        "K:C\n"
        'V: Vocal "C"E2G2A2G2E2D2C4|"G"D2E2G2E2D2C2D4|\n'
        "V: Ins C2E2G2c2|G2E2C2G,2|\n"
    )

    with gr.Blocks(title="Y4W · YuE2 OrbitQuant") as demo:
        gr.Markdown(
            "# Y4W · YuE2 OrbitQuant W4A4 推理\n"
            f"后端：`{backend}` · 同一服务同时提供 REST API（`/docs`）和 CLI（`y4w generate`）。\n\n"
            "> 未加载真实 3B 权重时运行内置合成网络，仅用于演示与冒烟测试。"
        )

        with gr.Tabs():
            with gr.Tab("生成 / Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            value="a rainy city at night, lo-fi beats",
                            placeholder="输入歌词 / 风格描述",
                        )
                    with gr.Column(scale=1):
                        max_new_tokens = gr.Slider(
                            minimum=1, maximum=256, value=16, step=1,
                            label="Max new tokens",
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                            label="Temperature",
                        )
                        seed = gr.Number(value=42, label="Seed（-1 表示随机）")

                run_btn = gr.Button("生成 / Generate", variant="primary")
                output = gr.Textbox(label="Result (JSON)", lines=14)
                run_btn.click(
                    fn=_run,
                    inputs=[prompt, max_new_tokens, temperature, seed],
                    outputs=output,
                )

                gr.Examples(
                    examples=[
                        ["a rainy city at night", 16, 0.8, 42],
                        ["energetic drum and bass", 24, 1.0, 7],
                        ["calm ambient piano", 12, 0.6, 123],
                    ],
                    inputs=[prompt, max_new_tokens, temperature, seed],
                )

            with gr.Tab("ABC → MIDI"):
                gr.Markdown(
                    "把 YuE2 生成的双声部 ABC 乐谱拆成两个 MIDI 文件："
                    "`vocal.mid`（歌声）与 `accompaniment.mid`（伴奏）。"
                    "识别 `V: Vocal` / `V: Ins` 声部，零依赖转换。"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        abc_text = gr.Textbox(
                            label="ABC 乐谱",
                            lines=12,
                            value=sample_abc,
                            placeholder="粘贴 X:1 ... K:C ... 的 ABC 文本",
                        )
                    with gr.Column(scale=1):
                        abc_btn = gr.Button("转换为 MIDI", variant="primary")
                        vocal_file = gr.File(label="歌声 vocal.mid")
                        accomp_file = gr.File(label="伴奏 accompaniment.mid")
                abc_info = gr.Textbox(label="转换信息", lines=10)
                abc_btn.click(
                    fn=_abc_to_midi,
                    inputs=[abc_text],
                    outputs=[vocal_file, accomp_file, abc_info],
                )

    return demo
