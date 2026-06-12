#!/usr/bin/env python3
import os
import sys
import json
import base64
import mimetypes
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

from llm_cli import load_config, DEFAULT_CONFIG

# ── 1. Config ──────────────────────────────────────────────────────────────────
config_path = "config.yaml"
cfg = load_config(config_path) if os.path.exists(config_path) else DEFAULT_CONFIG

mc = cfg["model"]
ic = cfg["inference"]
sc = cfg["session"]
wc = cfg.get("web", DEFAULT_CONFIG["web"])

# ── 2. Model ───────────────────────────────────────────────────────────────────
model_path  = str(Path(mc["path"]).expanduser())
mmproj_path = str(Path(mc["mmproj"]).expanduser()) if mc.get("mmproj") else None

if not os.path.isfile(model_path):
    print(f"Error: model not found at {model_path}", file=sys.stderr)
    sys.exit(1)

os.environ.setdefault("LLAMA_LOG_LEVEL", "error")

print(f"Loading {model_path} ...")

if mmproj_path and os.path.isfile(mmproj_path):
    print(f"Vision enabled: {mmproj_path}")
    chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
    VISION = True
else:
    print("No mmproj found — text only.", file=sys.stderr)
    chat_handler = None
    VISION = False

llm = Llama(
    model_path=model_path,
    chat_handler=chat_handler,
    n_gpu_layers=mc["gpu_layers"],
    n_ctx=mc["ctx"],
    n_batch=ic["n_batch"],
    n_threads=os.cpu_count() or 4,  # Use all available CPU threads
    flash_attn=mc.get("flash_attn", False),
    verbose=sc["verbose"],
    logits_all=True if chat_handler else False,
)
print("✓ Model loaded.")
print("start chatting with the model at http://{}:{}/".format(wc.get("host", "localhost"), wc.get("port", 7860)))

SYSTEM_PROMPT = sc.get("system_prompt", "You are a helpful assistant.")
MODEL_NAME    = Path(model_path).stem
CTX_LIMIT     = mc["ctx"]
CTX_BUDGET    = int(CTX_LIMIT * 0.90 * 4)

# ── 3. Inference ───────────────────────────────────────────────────────────────
def _content_chars(content) -> int:
    """Estimate char count of a message content (str or list of parts)."""
    if isinstance(content, str):
        return len(content)
    total = 0
    for part in content:
        if part.get("type") == "text":
            total += len(part.get("text", ""))
        # skip image_url parts — already stripped from history
    return total

def _strip_images(messages: list[dict]) -> list[dict]:
    """Remove image_url parts from history messages (keep text only)."""
    cleaned = []
    for msg in messages:
        if isinstance(msg["content"], list):
            text_parts = [p for p in msg["content"] if p.get("type") == "text"]
            text = " ".join(p["text"] for p in text_parts)
            cleaned.append({"role": msg["role"], "content": text})
        else:
            cleaned.append(msg)
    return cleaned

def _trim_history(history: list[dict], new_message_chars: int) -> tuple[list[dict], int]:
    """Efficiently trim history to fit within context budget."""
    fixed   = len(SYSTEM_PROMPT) + new_message_chars
    trimmed = list(history)

    # Pre-calculate content lengths to avoid repeated computation
    content_lengths = [_content_chars(m["content"]) for m in trimmed]
    total_history_chars = sum(content_lengths)

    while trimmed:
        if fixed + total_history_chars <= CTX_BUDGET:
            break
        # Remove oldest message pair (user + assistant)
        removed_chars = content_lengths[0] + (content_lengths[1] if len(content_lengths) > 1 else 0)
        trimmed = trimmed[2:] if len(trimmed) >= 2 else trimmed[1:]
        content_lengths = content_lengths[2:] if len(content_lengths) >= 2 else content_lengths[1:]
        total_history_chars -= removed_chars

    return trimmed, len(history) - len(trimmed)

def stream_response(history: list[dict], user_message: str, image_b64: str | None, image_mime: str):
    # Build user content
    if image_b64 and VISION:
        data_uri = f"data:{image_mime};base64,{image_b64}"
        user_content = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text",      "text": user_message or "Describe this image."},
        ]
    else:
        user_content = user_message

    # Strip images from history (they're too large to resend each turn)
    clean_history = _strip_images(history)
    msg_chars     = len(user_message) + (len(image_b64) // 4 if image_b64 else 0)
    trimmed, dropped = _trim_history(clean_history, msg_chars)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_content})

    if dropped:
        yield f"data: {json.dumps({'token': f'*[{dropped} earlier messages removed to fit context]*' + chr(10) + chr(10)})}\n\n"

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=ic["max_tokens"],
        temperature=ic["temperature"],
        top_p=ic["top_p"],
        top_k=ic["top_k"],
        repeat_penalty=ic["repeat_penalty"],
        stream=True,
    )
    for chunk in stream:
        token = chunk["choices"][0].get("delta", {}).get("content", "")
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"

    yield "data: [DONE]\n\n"

# ── 4. FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/chat")
async def chat(request: Request):
    body       = await request.json()
    history    = body.get("history", [])
    message    = body.get("message", "").strip()
    image_b64  = body.get("image_b64")   # base64 string, no data URI prefix
    image_mime = body.get("image_mime", "image/jpeg")

    if not message and not image_b64:
        return {"error": "empty message"}

    def generate():
        try:
            yield from stream_response(history, message, image_b64, image_mime)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/vision")
async def vision_status():
    return {"vision": VISION}

# ── 5. UI ──────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{MODEL_NAME}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:        #f6f8fa;
    --surface:   #ffffff;
    --border:    #d0d7de;
    --border-2:  #eaeef2;
    --text:      #1f2328;
    --text-2:    #57606a;
    --text-3:    #8c959f;
    --blue:      #0969da;
    --blue-dark: #0860ca;
    --user-text: #ffffff;
    --green:     #1a7f37;
    --green-bg:  #dafbe1;
    --green-bd:  #aceebb;
    --red:       #cf222e;
    --red-bg:    #ffebe9;
    --font:      'JetBrains Mono', 'Fira Code', monospace;
    --radius:    12px;
  }}
  html, body {{ height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; line-height: 1.6; }}

  #app {{ display: flex; flex-direction: column; height: 100vh; max-width: 860px; margin: 0 auto; }}

  /* header */
  #header {{ padding: 14px 24px 11px; border-bottom: 1px solid var(--border); background: var(--surface); flex-shrink: 0; display: flex; align-items: center; gap: 12px; }}
  #header h1 {{ font-size: 0.92rem; font-weight: 600; color: var(--text); }}
  .tag {{ font-size: 0.67rem; border-radius: 4px; padding: 2px 7px; display: inline-block; letter-spacing: 0.03em; border: 1px solid; }}
  .tag-green {{ color: var(--green); background: var(--green-bg); border-color: var(--green-bd); }}
  .tag-gray  {{ color: var(--text-2); background: var(--border-2); border-color: var(--border); }}

  /* messages */
  #messages {{ flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }}
  .msg-row {{ display: flex; flex-direction: column; }}
  .msg-row.user {{ align-items: flex-end; }}
  .msg-row.bot  {{ align-items: flex-start; }}
  .bubble {{ max-width: 72%; padding: 10px 14px; border-radius: var(--radius); font-size: 0.875rem; line-height: 1.65; word-break: break-word; white-space: pre-wrap; }}
  .msg-row.user .bubble {{ background: var(--blue); color: var(--user-text); border-radius: var(--radius) var(--radius) 3px var(--radius); }}
  .msg-row.bot  .bubble {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius) var(--radius) var(--radius) 3px; }}

  /* image preview in bubble */
  .bubble img.msg-img {{ max-width: 260px; max-height: 200px; border-radius: 8px; display: block; margin-bottom: 6px; border: 1px solid rgba(255,255,255,0.2); }}

  /* markdown */
  .bubble p {{ margin: 0 0 6px; }} .bubble p:last-child {{ margin-bottom: 0; }}
  .bubble code {{ background: var(--border-2); border-radius: 4px; padding: 1px 5px; font-family: var(--font); font-size: 0.82em; }}
  .bubble pre {{ background: var(--border-2); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; overflow-x: auto; margin: 6px 0; }}
  .bubble pre code {{ background: none; padding: 0; font-size: 0.83em; }}
  .bubble strong {{ font-weight: 600; }} .bubble em {{ font-style: italic; }}
  .cursor {{ display: inline-block; width: 2px; height: 1em; background: var(--text-3); vertical-align: text-bottom; animation: blink 0.8s step-end infinite; margin-left: 1px; }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}

  /* image attach preview */
  #img-preview-wrap {{
    display: none;
    align-items: center;
    gap: 8px;
    padding: 8px 0 4px;
  }}
  #img-preview {{
    width: 52px; height: 52px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
  }}
  #img-remove {{
    background: none;
    border: none;
    cursor: pointer;
    color: var(--text-3);
    padding: 2px;
    border-radius: 4px;
    display: flex;
    align-items: center;
  }}
  #img-remove:hover {{ color: var(--red); }}
  #img-name {{ font-size: 0.75rem; color: var(--text-2); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  /* input */
  #input-area {{ padding: 10px 24px 14px; background: var(--surface); border-top: 1px solid var(--border); flex-shrink: 0; }}
  #input-row {{ display: flex; gap: 8px; align-items: flex-end; }}
  #msg-input {{ flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 10px; color: var(--text); font-family: var(--font); font-size: 0.875rem; padding: 10px 14px; resize: none; outline: none; min-height: 42px; max-height: 160px; overflow-y: auto; transition: border-color 0.15s, box-shadow 0.15s; line-height: 1.5; }}
  #msg-input:focus {{ border-color: var(--blue); box-shadow: 0 0 0 3px rgba(9,105,218,0.1); }}
  #msg-input::placeholder {{ color: var(--text-3); }}

  .icon-btn {{ background: none; border: 1px solid var(--border); border-radius: 8px; color: var(--text-2); width: 42px; height: 42px; flex-shrink: 0; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.15s; }}
  .icon-btn:hover:not(:disabled) {{ background: var(--border-2); color: var(--text); }}
  .icon-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  #send-btn {{ background: var(--blue); border-color: var(--blue-dark); color: #fff; }}
  #send-btn:hover:not(:disabled) {{ background: var(--blue-dark); }}
  #attach-btn.has-image {{ border-color: var(--blue); color: var(--blue); background: #dbeafe; }}

  ::-webkit-scrollbar {{ width: 5px; }} ::-webkit-scrollbar-track {{ background: transparent; }} ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }} ::-webkit-scrollbar-thumb:hover {{ background: var(--text-3); }}
</style>
</head>
<body>
<div id="app">
  <div id="header">
    <h1>⚡ Local LLM</h1>
    <span class="tag tag-green" style="font-size:0.62rem">{MODEL_NAME[:40]}</span>
    <span class="tag tag-gray" id="vision-tag" style="display:none">vision</span>
  </div>

  <div id="messages"></div>

  <div id="input-area">
    <div id="img-preview-wrap">
      <img id="img-preview" src="" alt="">
      <span id="img-name"></span>
      <button id="img-remove" title="Remove image">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"/></svg>
      </button>
    </div>
    <div id="input-row">
      <button id="clear-btn" class="icon-btn" title="Clear chat">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M11 1.75V3h2.25a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1 0-1.5H5V1.75C5 .784 5.784 0 6.75 0h2.5C10.216 0 11 .784 11 1.75ZM4.496 6.675l.66 6.6a.25.25 0 0 0 .249.225h5.19a.25.25 0 0 0 .249-.225l.66-6.6a.75.75 0 0 1 1.492.149l-.66 6.6A1.748 1.748 0 0 1 10.595 15h-5.19a1.75 1.75 0 0 1-1.741-1.575l-.66-6.6a.75.75 0 1 1 1.492-.15ZM6.75 1.5h2.5a.25.25 0 0 1 .25.25V3h-3V1.75a.25.25 0 0 1 .25-.25Z"/></svg>
      </button>
      <button id="attach-btn" class="icon-btn" title="Attach image" style="display:none">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M1.75 2h7.5a.75.75 0 0 1 0 1.5h-7.5a.25.25 0 0 0-.25.25v9.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25v-4.5a.75.75 0 0 1 1.5 0v4.5A1.75 1.75 0 0 1 11.25 15h-9.5A1.75 1.75 0 0 1 0 13.25v-9.5C0 2.784.784 2 1.75 2Zm10.5 0h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5h-1.5a.75.75 0 0 1 0-1.5h1.5V.5a.75.75 0 0 1 1.5 0Z"/></svg>
      </button>
      <input type="file" id="file-input" accept="image/*" style="display:none">
      <textarea id="msg-input" placeholder="Message..." rows="1"></textarea>
      <button id="send-btn" class="icon-btn" title="Send">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M.989 8 .064 2.68a1.342 1.342 0 0 1 1.85-1.462l13.402 5.744a1.13 1.13 0 0 1 0 2.076L1.913 14.782a1.342 1.342 0 0 1-1.85-1.463L.99 8Zm.603-5.429.93 4.801H7.5a.75.75 0 0 1 0 1.5H2.522l-.93 4.8 11.908-5.1L1.592 2.57Z"/></svg>
      </button>
    </div>
  </div>
</div>

<input type="file" id="file-input" accept="image/*" style="display:none">

<script>
let history  = [];
let streaming = false;
let pendingImage = null; // {{ b64, mime, dataUrl, name }}

const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('msg-input');
const sendBtn     = document.getElementById('send-btn');
const clearBtn    = document.getElementById('clear-btn');
const attachBtn   = document.getElementById('attach-btn');
const fileInput   = document.getElementById('file-input');
const previewWrap = document.getElementById('img-preview-wrap');
const previewImg  = document.getElementById('img-preview');
const imgName     = document.getElementById('img-name');
const imgRemove   = document.getElementById('img-remove');
const visionTag   = document.getElementById('vision-tag');

// Check vision capability
fetch('/vision').then(r => r.json()).then(d => {{
  if (d.vision) {{
    attachBtn.style.display = 'flex';
    visionTag.style.display = 'inline-block';
  }}
}});

function escapeHtml(t) {{
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
function renderMarkdown(text) {{
  text = text.replace(/```(\\w*)\\n?([\\s\\S]*?)```/g, (_, l, c) =>
    `<pre><code>${{escapeHtml(c.trimEnd())}}</code></pre>`);
  text = text.replace(/`([^`]+)`/g, (_, c) => `<code>${{escapeHtml(c)}}</code>`);
  text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  text = text.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
  return text.split(/\\n\\n+/).map(p => `<p>${{p.replace(/\\n/g,'<br>')}}</p>`).join('');
}}
function scrollBottom() {{ messagesEl.scrollTop = messagesEl.scrollHeight; }}

function addBubble(role, text, imgDataUrl) {{
  const row = document.createElement('div');
  row.className = `msg-row ${{role}}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (imgDataUrl) {{
    const img = document.createElement('img');
    img.src = imgDataUrl;
    img.className = 'msg-img';
    bubble.appendChild(img);
  }}
  if (role === 'user') {{
    if (text) {{
      const span = document.createElement('span');
      span.textContent = text;
      bubble.appendChild(span);
    }}
  }} else {{
    const content = document.createElement('span');
    content.innerHTML = renderMarkdown(text);
    bubble.appendChild(content);
  }}
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollBottom();
  return bubble;
}}

function setEnabled(on) {{
  streaming = !on;
  sendBtn.disabled = !on;
  inputEl.disabled = !on;
  attachBtn.disabled = !on;
  if (on) inputEl.focus();
}}

// Auto-resize
inputEl.addEventListener('input', () => {{
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}});
inputEl.addEventListener('keydown', e => {{
  if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); if (!streaming) sendMessage(); }}
}});
sendBtn.addEventListener('click', () => {{ if (!streaming) sendMessage(); }});
clearBtn.addEventListener('click', () => {{
  if (streaming) return;
  history = [];
  messagesEl.innerHTML = '';
  clearPendingImage();
}});

// ── Image attach ───────────────────────────────────────────────────────────────
attachBtn.addEventListener('click', () => fileInput.click());
imgRemove.addEventListener('click', clearPendingImage);

fileInput.addEventListener('change', () => {{
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {{
    const dataUrl = e.target.result;
    const mime    = file.type || 'image/jpeg';
    const b64     = dataUrl.split(',')[1];
    pendingImage  = {{ b64, mime, dataUrl, name: file.name }};
    previewImg.src   = dataUrl;
    imgName.textContent = file.name;
    previewWrap.style.display = 'flex';
    attachBtn.classList.add('has-image');
  }};
  reader.readAsDataURL(file);
  fileInput.value = '';
}});

// Paste image from clipboard
document.addEventListener('paste', e => {{
  if (!attachBtn.style.display || attachBtn.style.display === 'none') return;
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {{
    if (item.type.startsWith('image/')) {{
      const file   = item.getAsFile();
      const reader = new FileReader();
      reader.onload = ev => {{
        const dataUrl = ev.target.result;
        const mime    = item.type;
        const b64     = dataUrl.split(',')[1];
        pendingImage  = {{ b64, mime, dataUrl, name: 'pasted-image.png' }};
        previewImg.src   = dataUrl;
        imgName.textContent = 'pasted image';
        previewWrap.style.display = 'flex';
        attachBtn.classList.add('has-image');
      }};
      reader.readAsDataURL(file);
      break;
    }}
  }}
}});

function clearPendingImage() {{
  pendingImage = null;
  previewWrap.style.display = 'none';
  previewImg.src = '';
  imgName.textContent = '';
  attachBtn.classList.remove('has-image');
}}

// ── Send ───────────────────────────────────────────────────────────────────────
async function sendMessage() {{
  const text = inputEl.value.trim();
  if (!text && !pendingImage) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';

  const imgSnap = pendingImage;
  clearPendingImage();
  setEnabled(false);

  addBubble('user', text, imgSnap?.dataUrl);

  const botBubble    = addBubble('bot', '');
  const contentSpan  = botBubble.querySelector('span');
  const cursor       = document.createElement('span');
  cursor.className   = 'cursor';
  botBubble.appendChild(cursor);

  let fullText = '';

  try {{
    const res = await fetch('/chat', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        message:    text,
        history:    history,
        image_b64:  imgSnap?.b64  ?? null,
        image_mime: imgSnap?.mime ?? 'image/jpeg',
      }})
    }});

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {{ stream: true }});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {{
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') break;
        const parsed = JSON.parse(data);
        if (parsed.error) {{
          fullText += `\\n⚠️ ${{parsed.error}}`;
        }} else if (parsed.token) {{
          fullText += parsed.token;
        }}
        cursor.remove();
        contentSpan.innerHTML = renderMarkdown(fullText);
        botBubble.appendChild(cursor);
        scrollBottom();
      }}
    }}
  }} catch (err) {{
    fullText = `⚠️ Connection error: ${{err.message}}`;
  }}

  cursor.remove();
  contentSpan.innerHTML = renderMarkdown(fullText);
  scrollBottom();

  // Store in history — text only for images (strip base64)
  const histUserContent = imgSnap
    ? [ {{type:'image_url', image_url:{{url: imgSnap.dataUrl}}}},
        {{type:'text', text: text || 'Describe this image.'}} ]
    : text;
  history.push({{ role: 'user',      content: histUserContent }});
  history.push({{ role: 'assistant', content: fullText }});

  setEnabled(true);
}}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

# ── 6. Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=wc.get("host", "127.0.0.1"),
        port=wc.get("port", 7860),
        log_level="warning",
    )