#!/usr/bin/env python3
"""Single-file web UI to query a served LoRA adapter (or the base model) via vLLM.

Stdlib only -- no pip installs. Serves a small HTML page with a question box and
a model picker, and proxies requests to the vLLM OpenAI server.

Run (on the box, where vLLM is on :8000):
    python3 webui.py                 # serve on http://0.0.0.0:8080
    PORT=9000 webui.py               # custom port
    VLLM_URL=http://localhost:8000 python3 webui.py

Then open http://<host>:8080  (from a laptop, SSH-forward it:
    ssh -L 8080:localhost:8080 ec2-user@<instance> ).

Endpoints:
    GET  /          the web page
    GET  /models    proxy of vLLM /v1/models (populates the dropdown)
    POST /ask       {question, model, max_tokens, temperature} -> {answer}
    POST /load      {lora_name} -> loads the adapter via the manager (:8001)
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
MANAGER_URL = os.environ.get("MANAGER_URL", "http://localhost:8001").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "jokeans")
PORT = int(os.environ.get("PORT", "8080"))

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>autotune · ask a LoRA</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; }
  label { font-size: .85rem; opacity: .8; display: block; margin: .75rem 0 .25rem; }
  textarea, select, input { width: 100%; padding: .6rem; font: inherit; border-radius: 8px;
    border: 1px solid #8884; background: transparent; color: inherit; box-sizing: border-box; }
  textarea { min-height: 4rem; resize: vertical; }
  .row { display: flex; gap: 1rem; align-items: end; }
  .row > div { flex: 1; }
  button { margin-top: 1rem; padding: .6rem 1.2rem; font: inherit; border: 0; border-radius: 8px;
    background: #4f46e5; color: #fff; cursor: pointer; white-space: nowrap; }
  button.secondary { background: #6b7280; margin-top: 0; }
  button:disabled { opacity: .5; cursor: default; }
  #answer { white-space: pre-wrap; margin-top: 1.25rem; padding: 1rem; border-radius: 8px;
    background: #8881; min-height: 2rem; }
  .meta { font-size: .8rem; opacity: .6; margin-top: .5rem; }
</style>
</head>
<body>
  <h1>Ask a LoRA</h1>
  <div class="row">
    <div>
      <label for="model">Model</label>
      <select id="model"></select>
    </div>
  </div>
  <label for="loadName">Load an adapter (pulls from S3 into vLLM)</label>
  <div class="row">
    <div><input id="loadName" placeholder="jokeans"></div>
    <button id="load" class="secondary">Load adapter</button>
  </div>
  <div class="meta" id="loadStatus"></div>
  <label for="q">Question</label>
  <textarea id="q" placeholder="What is the capital of Brazil?"></textarea>
  <button id="go">Ask</button>
  <div id="answer"></div>
  <div class="meta" id="meta"></div>

<script>
const $ = s => document.querySelector(s);

async function loadModels() {
  try {
    const r = await fetch('/models');
    const data = await r.json();
    const ids = (data.data || []).map(m => m.id);
    const sel = $('#model');
    sel.innerHTML = '';
    for (const id of ids) {
      const o = document.createElement('option');
      o.value = o.textContent = id;
      if (id === %DEFAULT_MODEL%) o.selected = true;
      sel.appendChild(o);
    }
    if (!ids.length) sel.innerHTML = '<option>(no models — is vLLM up?)</option>';
  } catch (e) {
    $('#model').innerHTML = '<option>(could not reach server)</option>';
  }
}

async function ask() {
  const q = $('#q').value.trim();
  if (!q) return;
  $('#go').disabled = true;
  $('#answer').textContent = '…thinking';
  $('#meta').textContent = '';
  const t0 = performance.now();
  try {
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q, model: $('#model').value})
    });
    const data = await r.json();
    if (data.error) {
      $('#answer').textContent = 'Error: ' + data.error;
    } else {
      $('#answer').textContent = data.answer;
      const ms = Math.round(performance.now() - t0);
      $('#meta').textContent = `${data.model} · ${ms} ms`;
    }
  } catch (e) {
    $('#answer').textContent = 'Request failed: ' + e;
  } finally {
    $('#go').disabled = false;
  }
}

async function loadAdapter() {
  const name = $('#loadName').value.trim();
  if (!name) return;
  $('#load').disabled = true;
  $('#loadStatus').textContent = 'loading ' + name + ' … (may take a bit on first pull from S3)';
  try {
    const r = await fetch('/load', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({lora_name: name})
    });
    const data = await r.json();
    if (data.error) {
      $('#loadStatus').textContent = 'Error: ' + data.error;
    } else {
      $('#loadStatus').textContent = 'loaded "' + name + '" ✓';
      await loadModels();
      $('#model').value = name;
    }
  } catch (e) {
    $('#loadStatus').textContent = 'Request failed: ' + e;
  } finally {
    $('#load').disabled = false;
  }
}

$('#go').addEventListener('click', ask);
$('#load').addEventListener('click', loadAdapter);
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask(); });
loadModels();
</script>
</body>
</html>"""


def vllm_get(path):
    with urllib.request.urlopen(VLLM_URL + path, timeout=30) as r:
        return json.load(r)


def manager_load(name):
    body = json.dumps({"lora_name": name}).encode()
    req = urllib.request.Request(MANAGER_URL + "/v1/loras/load", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:  # first S3 pull can be slow
        return json.load(r)


def vllm_chat(model, question, max_tokens, temperature):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(VLLM_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        payload = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            html = PAGE.replace("%DEFAULT_MODEL%", json.dumps(DEFAULT_MODEL))
            return self._send(200, html, "text/html; charset=utf-8")
        if self.path == "/models":
            try:
                return self._send(200, json.dumps(vllm_get("/v1/models")))
            except Exception as e:  # noqa: BLE001
                return self._send(502, json.dumps({"data": [], "error": str(e)}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"error": f"bad request body: {e}"}))

        if self.path == "/ask":
            try:
                question = (req.get("question") or "").strip()
                if not question:
                    return self._send(400, json.dumps({"error": "empty question"}))
                model = req.get("model") or DEFAULT_MODEL
                answer = vllm_chat(model, question,
                                   int(req.get("max_tokens", 128)),
                                   float(req.get("temperature", 0.7)))
                return self._send(200, json.dumps({"answer": answer, "model": model}))
            except urllib.error.HTTPError as e:
                return self._send(502, json.dumps({"error": f"vLLM HTTP {e.code}: {e.read().decode()[:300]}"}))
            except Exception as e:  # noqa: BLE001
                return self._send(502, json.dumps({"error": str(e)}))

        if self.path == "/load":
            try:
                name = (req.get("lora_name") or "").strip()
                if not name:
                    return self._send(400, json.dumps({"error": "empty lora_name"}))
                result = manager_load(name)
                return self._send(200, json.dumps({"ok": True, "result": result}))
            except urllib.error.HTTPError as e:
                return self._send(502, json.dumps({"error": f"manager HTTP {e.code}: {e.read().decode()[:300]}"}))
            except Exception as e:  # noqa: BLE001
                return self._send(502, json.dumps({"error": str(e)}))

        self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *args):  # quieter console
        pass


if __name__ == "__main__":
    print(f"autotune web UI on http://0.0.0.0:{PORT}  (vLLM at {VLLM_URL}, default model {DEFAULT_MODEL})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
