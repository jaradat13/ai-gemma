#!/usr/bin/env python3
import os
import sys
import gradio as gr
from pathlib import Path
from llama_cpp import Llama

from llm_cli import load_config, DEFAULT_CONFIG

# ── 1. Config ─────────────────────────────────────────────────────────────────
config_path = "config.yaml"
cfg = load_config(config_path) if os.path.exists(config_path) else DEFAULT_CONFIG

mc  = cfg["model"]
ic  = cfg["inference"]
sc  = cfg["session"]
wc  = cfg.get("web", DEFAULT_CONFIG["web"])

# ── 2. Model ──────────────────────────────────────────────────────────────────
model_path = str(Path(mc["path"]).expanduser())
if not os.path.isfile(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

os.environ.setdefault("LLAMA_LOG_LEVEL", "error")

print(f"Loading {model_path}...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=mc["gpu_layers"],
    n_ctx=mc["ctx"],
    n_batch=ic["n_batch"],
    flash_attn=mc.get("flash_attn", False),
    verbose=sc["verbose"],
)
print("✓ Model loaded.")

# ── 3. Inference ───────────────────────────────────────────────────────────────
def predict(message, history):
    system_prompt = sc.get("system_prompt", "You are a helpful assistant.")
    formatted_messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        if isinstance(turn, (list, tuple)):
            user_hist, assistant_hist = turn
            if user_hist:
                formatted_messages.append({"role": "user", "content": str(user_hist)})
            if assistant_hist:
                formatted_messages.append({"role": "assistant", "content": str(assistant_hist)})
        elif isinstance(turn, dict):
            formatted_messages.append({"role": turn["role"], "content": turn["content"]})

    if isinstance(message, dict):
        user_text = message.get("text", "").strip()
        attached_files = message.get("files", [])
    else:
        user_text = str(message).strip()
        attached_files = []

    file_contents_buffer = []
    for file_info in attached_files:
        file_path = file_info if isinstance(file_info, str) else file_info.get("path")
        p = Path(file_path)
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                file_contents_buffer.append(f"--- File: {p.name} ---\n{content}\n")
            except Exception as e:
                file_contents_buffer.append(f"[Error reading {p.name}: {e}]\n")

    full_prompt = ("\n".join(file_contents_buffer) + f"\nUser Question:\n{user_text}") \
                  if file_contents_buffer else user_text

    formatted_messages.append({"role": "user", "content": full_prompt})

    try:
        stream = llm.create_chat_completion(
            messages=formatted_messages,
            max_tokens=ic["max_tokens"],
            temperature=ic["temperature"],
            top_p=ic["top_p"],
            top_k=ic["top_k"],
            repeat_penalty=ic["repeat_penalty"],
            stream=True,
        )
        response_text = ""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                response_text += token
                yield response_text
    except ValueError as e:
        if "exceed context window" in str(e):
            yield "⚠️ Context limit exceeded. Click **Clear** to reset history."
        else:
            yield f"⚠️ Error: {str(e)}"

# ── 4. UI (all values from config.yaml → web section) ─────────────────────────
theme_cfg = wc.get("theme", {})
theme = gr.themes.Soft(
    primary_hue=theme_cfg.get("primary_hue", "cyan"),
    secondary_hue=theme_cfg.get("secondary_hue", "slate"),
)

with gr.Blocks(title=wc.get("title", "Local LLM")) as demo:
    gr.Markdown(f"## ⚡ {wc.get('title', 'Local LLM')} — `{Path(model_path).name}`")
    gr.Markdown(wc.get("description", ""))

    gr.ChatInterface(
        fn=predict,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            placeholder="Type a message or drop a file...",
            file_types=wc.get("file_types", [".txt", ".py", ".md", ".log"]),
            file_count="multiple",
        ),
    )

if __name__ == "__main__":
    demo.launch(
        server_name=wc.get("host", "127.0.0.1"),
        server_port=wc.get("port", 7860),
        share=wc.get("share", False),
        theme=theme,
    )