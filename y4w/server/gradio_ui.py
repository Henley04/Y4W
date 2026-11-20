"""Gradio WebUI mounted by ``y4w serve``.

Kept in its own module and imported lazily so the REST API and the whole test
suite work on machines (and CI runners) without Gradio installed.
"""
from __future__ import annotations

import json
from typing import Any, Callable


def build_gradio_interface(generate_fn: Callable[..., dict[str, Any]],
                           engine: Any = None):
    """Build a Gradio Blocks UI bound to the shared thread-safe generator."""
    import gradio as gr

    backend = getattr(getattr(engine, "backend", None), "name", "auto")

    def _run(prompt, max_new_tokens, temperature, seed):
        try:
            result = generate_fn(prompt, max_new_tokens, temperature, seed)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:  # surface errors in the UI instead of crashing
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"},
                              ensure_ascii=False, indent=2)

    with gr.Blocks(title="Y4W · YuE2 OrbitQuant") as demo:
        gr.Markdown(
            "# Y4W · YuE2 OrbitQuant W4A4 推理\n"
            f"后端：`{backend}` · 同一服务同时提供 REST API（`/docs`）和 CLI（`y4w generate`）。\n\n"
            "> 未加载真实 3B 权重时运行内置合成网络，仅用于演示与冒烟测试。"
        )
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

    return demo
