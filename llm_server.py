#!/usr/bin/env python3
"""
Minimal OpenAI-compatible API server for Continue.dev / VSCode.
Implements /v1/models and /v1/chat/completions with streaming.
Reads all settings from config.yaml.

Usage:
    python llm_server.py
    python llm_server.py --config /path/to/config.yaml
"""

import os
import sys
import json
import time
import uuid
import argparse
from pathlib import Path
from typing import Iterator

from llm_cli import load_config, DEFAULT_CONFIG

# ── Load model ─────────────────────────────────────────────────────────────────
def load_llm(cfg: dict):
    os.environ.setdefault("LLAMA_LOG_LEVEL", "error")
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed.")
        sys.exit(1)

    mc = cfg["model"]
    ic = cfg["inference"]
    sc = cfg["session"]

    model_path = str(Path(mc["path"]).expanduser())
    if not Path(model_path).is_file():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=mc.get("gpu_layers", 99),
        n_ctx=mc.get("ctx", 8192),
        n_batch=ic.get("n_batch", 512),
        flash_attn=mc.get("flash_attn", False),
        verbose=sc.get("verbose", False),
    )
    print("✓ Model loaded.\n")
    return llm, model_path


# ── FastAPI app ────────────────────────────────────────────────────────────────
def create_server(llm, model_path: str, cfg: dict):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse, JSONResponse
        import uvicorn
    except ImportError:
        print("Error: fastapi/uvicorn not installed.")
        print("Run: pip install fastapi uvicorn")
        sys.exit(1)

    ic = cfg["inference"]
    model_name = Path(model_path).stem
    app = FastAPI(title="Local LLM API")

    # ── GET /v1/models ─────────────────────────────────────────────────────────
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id":       model_name,
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "local",
            }]
        }

    # ── POST /v1/chat/completions ──────────────────────────────────────────────
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body        = await request.json()
        messages    = body.get("messages", [])
        stream      = body.get("stream", False)
        max_tokens  = body.get("max_tokens",  ic.get("max_tokens",  1024))
        temperature = body.get("temperature", ic.get("temperature", 0.7))
        top_p       = body.get("top_p",       ic.get("top_p",       0.95))
        top_k       = body.get("top_k",       ic.get("top_k",       40))
        repeat_pen  = body.get("repeat_penalty", ic.get("repeat_penalty", 1.1))

        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        def generate() -> Iterator[str]:
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_pen,
                stream=True,
            )
            for chunk in output:
                delta   = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                finish  = chunk["choices"][0].get("finish_reason")
                data = {
                    "id":      req_id,
                    "object":  "chat.completion.chunk",
                    "created": created,
                    "model":   model_name,
                    "choices": [{
                        "index":         0,
                        "delta":         {"content": content} if content else {},
                        "finish_reason": finish,
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(generate(), media_type="text/event-stream")

        # Non-streaming fallback
        full_text = ""
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_pen,
            stream=True,
        ):
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            full_text += content

        return {
            "id":      req_id,
            "object":  "chat.completion",
            "created": created,
            "model":   model_name,
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible LLM server")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host",   default=None)
    parser.add_argument("--port",   default=None, type=int)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    sv   = cfg.get("server", {})
    host = args.host or sv.get("host", "127.0.0.1")
    port = args.port or sv.get("port", 8000)

    llm, model_path = load_llm(cfg)
    app = create_server(llm, model_path, cfg)

    print(f"Endpoint   : http://{host}:{port}/v1")
    print(f"Continue.dev apiBase → http://{host}:{port}/v1\n")

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()