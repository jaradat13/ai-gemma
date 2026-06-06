#!/usr/bin/env python3
"""
Local LLM Interactive CLI
Uses llama-cpp-python to load a GGUF model directly.

Usage:
    python llm_cli.py
    python llm_cli.py --config /path/to/config.yaml
    python llm_cli.py --model /path/to/model.gguf --gpu-layers 26

CLI args always override config.yaml values.

Commands (during chat):
    /exit   or  /quit   — exit the session
    /clear              — clear conversation history
    /system <msg>       — set or replace the system prompt
    /history            — show conversation history
    /info               — show current model/config info
    /save <file>        — save conversation to a text file
    /help               — show this help message
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# ── ANSI colors ────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"

def print_banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════╗
║        Local LLM Interactive CLI     ║
║   llama-cpp-python · GGUF · Direct   ║
╚══════════════════════════════════════╝{C.RESET}
Type {C.YELLOW}/help{C.RESET} for commands, {C.YELLOW}/exit{C.RESET} to quit.
""")

def print_help():
    cmds = [
        ("/exit, /quit",    "Exit the session"),
        ("/clear",          "Clear conversation history (keeps system prompt)"),
        ("/system <msg>",   "Set or replace the system prompt"),
        ("/history",        "Show full conversation history"),
        ("/info",           "Show model & config info"),
        ("/save <file>",    "Save conversation to a file (default: chat_log.txt)"),
        ("/help",           "Show this help message"),
    ]
    print(f"\n{C.BOLD}Available commands:{C.RESET}")
    for cmd, desc in cmds:
        print(f"  {C.YELLOW}{cmd:<22}{C.RESET} {desc}")
    print()


# ── Config loader ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model": {
        "path":       None,
        "gpu_layers": 26,
        "ctx":        4096,
    },
    "inference": {
        "max_tokens":     1024,
        "temperature":    0.7,
        "top_p":          0.95,
        "top_k":          40,
        "repeat_penalty": 1.1,
        "n_batch":        512,
    },
    "session": {
        "system_prompt": "You are a helpful assistant.",
        "verbose":       False,
        "auto_save":     None,
    },
    "web": {
        "host":        "127.0.0.1",
        "port":        7860,
        "share":       False,
        "title":       "Local LLM",
        "description": "Running on your GPU via llama-cpp-python.",
        "theme": {
            "primary_hue":   "cyan",
            "secondary_hue": "slate",
        },
        "file_types": [".txt", ".py", ".cpp", ".h", ".json", ".yaml", ".md", ".log"],
    },
}

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result

def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print(f"{C.RED}Error: PyYAML not installed. Run: pip install pyyaml{C.RESET}")
        sys.exit(1)

    path = Path(config_path).expanduser()
    if not path.is_file():
        print(f"{C.YELLOW}Warning: Config file not found at '{config_path}'. Using defaults.{C.RESET}\n")
        return DEFAULT_CONFIG

    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    print(f"{C.DIM}Loaded config: {path}{C.RESET}")
    return cfg

def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI args take priority over config file values."""
    if args.model:
        cfg["model"]["path"] = args.model
    if args.gpu_layers is not None:
        cfg["model"]["gpu_layers"] = args.gpu_layers
    if args.ctx is not None:
        cfg["model"]["ctx"] = args.ctx
    if args.max_tokens is not None:
        cfg["inference"]["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        cfg["inference"]["temperature"] = args.temperature
    if args.system:
        cfg["session"]["system_prompt"] = args.system
    if args.verbose:
        cfg["session"]["verbose"] = True
    return cfg


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model(cfg: dict):
    # Suppress llama.cpp info/warning noise (keep errors visible)
    os.environ.setdefault("LLAMA_LOG_LEVEL", "error")

    try:
        from llama_cpp import Llama
    except ImportError:
        print(f"{C.RED}Error: llama-cpp-python is not installed.{C.RESET}")
        print("Install with CUDA support:")
        print("  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-cache-dir")
        sys.exit(1)

    model_path = str(Path(cfg["model"]["path"]).expanduser())
    if not os.path.isfile(model_path):
        print(f"{C.RED}Error: Model file not found: {model_path}{C.RESET}")
        print("Update 'model.path' in config.yaml or pass --model <path>")
        sys.exit(1)

    mc = cfg["model"]
    ic = cfg["inference"]
    sc = cfg["session"]

    print(f"{C.DIM}Loading: {model_path}{C.RESET}")
    print(f"{C.DIM}GPU layers: {mc['gpu_layers']} | Context: {mc['ctx']} tokens{C.RESET}")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=mc["gpu_layers"],
        n_ctx=mc["ctx"],
        n_batch=ic["n_batch"],
        flash_attn=mc.get("flash_attn", False),
        verbose=sc["verbose"],
    )
    print(f"{C.GREEN}✓ Model loaded successfully{C.RESET}\n")
    return llm


# ── Chat session ───────────────────────────────────────────────────────────────
class ChatSession:
    def __init__(self, llm, cfg: dict):
        self.llm = llm
        self.cfg = cfg
        self.system_prompt: str = cfg["session"]["system_prompt"]
        self.history: list[dict] = []

    def _build_messages(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        return messages

    def _trim_history(self):
        """Drop oldest message pairs until safely under context limit."""
        ctx_limit = self.cfg["model"]["ctx"]
        while len(self.history) > 2:
            messages = self._build_messages()
            tokens = self.llm.tokenize(
                " ".join(m["content"] for m in messages).encode()
            )
            if len(tokens) < int(ctx_limit * 0.85):
                break
            self.history = self.history[2:]
            print(f"{C.YELLOW}[context trimmed — oldest messages removed]{C.RESET}")

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        self._trim_history()
        ic = self.cfg["inference"]

        print(f"\n{C.CYAN}{C.BOLD}Assistant:{C.RESET} ", end="", flush=True)
        full_response = ""

        try:
            stream = self.llm.create_chat_completion(
                messages=self._build_messages(),
                max_tokens=ic["max_tokens"],
                temperature=ic["temperature"],
                top_p=ic["top_p"],
                top_k=ic["top_k"],
                repeat_penalty=ic["repeat_penalty"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response += token
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}[interrupted]{C.RESET}", end="")
        except ValueError as e:
            if "exceed context window" in str(e):
                print(f"\n{C.RED}[input too long — try /clear or shorten your message]{C.RESET}")
                self.history.pop()
                return ""
            raise

        print()
        self.history.append({"role": "assistant", "content": full_response})
        return full_response

    def clear(self):
        self.history = []
        print(f"{C.YELLOW}History cleared.{C.RESET}")

    def set_system(self, prompt: str):
        self.system_prompt = prompt
        print(f"{C.YELLOW}System prompt updated.{C.RESET}")

    def show_history(self):
        if not self.history:
            print(f"{C.DIM}No conversation history yet.{C.RESET}")
            return
        print(f"\n{C.BOLD}── Conversation History ──{C.RESET}")
        if self.system_prompt:
            print(f"{C.DIM}[system] {self.system_prompt}{C.RESET}")
        for msg in self.history:
            color = C.GREEN if msg["role"] == "user" else C.CYAN
            print(f"{color}[{msg['role']}]{C.RESET} {msg['content']}")
        print()

    def show_info(self):
        mc = self.cfg["model"]
        ic = self.cfg["inference"]
        sc = self.cfg["session"]
        print(f"\n{C.BOLD}── Session Info ──{C.RESET}")
        print(f"  {'Model':<16}: {mc['path']}")
        print(f"  {'GPU layers':<16}: {mc['gpu_layers']}")
        print(f"  {'Context':<16}: {mc['ctx']} tokens")
        print(f"  {'Max tokens':<16}: {ic['max_tokens']}")
        print(f"  {'Temperature':<16}: {ic['temperature']}")
        print(f"  {'Top-p':<16}: {ic['top_p']}")
        print(f"  {'Top-k':<16}: {ic['top_k']}")
        print(f"  {'Repeat penalty':<16}: {ic['repeat_penalty']}")
        print(f"  {'System prompt':<16}: {self.system_prompt or '(none)'}")
        print(f"  {'History':<16}: {len(self.history)} messages")
        print()

    def save(self, filepath: str):
        filepath = str(Path(filepath).expanduser())
        lines = []
        if self.system_prompt:
            lines.append(f"[SYSTEM]\n{self.system_prompt}\n")
        for msg in self.history:
            lines.append(f"[{msg['role'].upper()}]\n{msg['content']}\n")
        with open(filepath, "w") as f:
            f.write(f"Chat log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(lines))
        print(f"{C.GREEN}Saved to: {filepath}{C.RESET}")

    def auto_save(self):
        path = self.cfg["session"].get("auto_save")
        if path:
            self.save(path)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for local GGUF models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config",      default="config.yaml",  help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--model",       default=None,           help="Override: path to .gguf file")
    parser.add_argument("--gpu-layers",  type=int, default=None, help="Override: GPU layers to offload")
    parser.add_argument("--ctx",         type=int, default=None, help="Override: context window size")
    parser.add_argument("--max-tokens",  type=int, default=None, help="Override: max tokens per response")
    parser.add_argument("--temperature", type=float,default=None,help="Override: sampling temperature")
    parser.add_argument("--system",      default=None,           help="Override: system prompt")
    parser.add_argument("--verbose",     action="store_true",    help="Override: show llama.cpp debug output")
    args = parser.parse_args()

    print_banner()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if not cfg["model"]["path"]:
        print(f"{C.RED}Error: No model path set.{C.RESET}")
        print("Set 'model.path' in config.yaml or pass --model <path>")
        sys.exit(1)

    llm = load_model(cfg)
    session = ChatSession(llm, cfg)

    while True:
        try:
            user_input = input(f"{C.GREEN}{C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Goodbye.{C.RESET}")
            session.auto_save()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                print(f"{C.DIM}Goodbye.{C.RESET}")
                session.auto_save()
                break
            elif cmd == "/clear":
                session.clear()
            elif cmd == "/system":
                if arg:
                    session.set_system(arg)
                else:
                    print(f"{C.RED}Usage: /system <prompt text>{C.RESET}")
            elif cmd == "/history":
                session.show_history()
            elif cmd == "/info":
                session.show_info()
            elif cmd == "/save":
                session.save(arg.strip() or "chat_log.txt")
            elif cmd == "/help":
                print_help()
            else:
                print(f"{C.RED}Unknown command: {cmd}. Type /help for available commands.{C.RESET}")
            continue

        session.chat(user_input)


if __name__ == "__main__":
    main()