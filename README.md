# AI Gemma

Local Gemma GGUF chat utilities powered by `llama-cpp-python`.

This project provides three ways to run the same local model:

- `llm_cli.py` - interactive terminal chat.
- `llm_web.py` - Gradio web chat with optional file attachments.
- `llm_server.py` - minimal OpenAI-compatible API server for tools such as Continue.dev or VS Code extensions.

## Requirements

- Python 3.10+
- A GGUF model file
- `llama-cpp-python`
- `PyYAML`
- Optional web/API dependencies: `gradio`, `fastapi`, `uvicorn`

For CUDA builds of `llama-cpp-python`, install it with the flags appropriate for your system. For example:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

Install the remaining Python dependencies as needed:

```bash
pip install pyyaml gradio fastapi uvicorn
```

## Model Setup

Place your `.gguf` model under `models/` or update `model.path` in `config.yaml`.

The current config expects:

```text
~/ai-gemma/models/gemma-4-E2B-it-Q4_K_M.gguf
```

GGUF files are intentionally ignored by Git because they are large local artifacts.

## Configuration

Edit `config.yaml` to control:

- Model path, GPU layers, context size, and flash attention.
- Generation settings such as temperature, top-p, top-k, and max tokens.
- CLI session defaults.
- Gradio web UI host, port, title, and upload file types.
- OpenAI-compatible API host and port.

CLI arguments override values from `config.yaml`.

## Usage

Run the interactive CLI:

```bash
python llm_cli.py
```

Use a custom config or model:

```bash
python llm_cli.py --config /path/to/config.yaml
python llm_cli.py --model /path/to/model.gguf --gpu-layers 99
```

Run the Gradio web UI:

```bash
python llm_web.py
```

Run the OpenAI-compatible API server:

```bash
python llm_server.py
```

The API server exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`

By default, the server base URL is:

```text
http://127.0.0.1:8000/v1
```

## CLI Commands

Inside the interactive CLI:

- `/help` - show available commands.
- `/exit` or `/quit` - exit.
- `/clear` - clear conversation history.
- `/system <msg>` - set the system prompt.
- `/history` - show conversation history.
- `/info` - show model and config details.
- `/save <file>` - save the conversation.
