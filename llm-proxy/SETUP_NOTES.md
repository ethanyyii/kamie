# llm-proxy — the thinking switch (v14)

AIRI's "思考模式" setting is real and AIRI really does send `"think": false`, but
Ollama's OpenAI-compatible endpoint — the one AIRI's `/v1/` Base URL lands on —
**ignores that field**. Measured: `think:false`, `think:true` and no field at all
all produce ~300 characters of hidden reasoning and ~1.0s to the first visible
token. Only the native `/api/chat` honours it, and AIRI never calls that.

`reasoning_effort: "none"` is the switch that works there: 0.150s to first token,
0 reasoning characters. AIRI has no UI for extra body fields, so this proxy adds
it on the way through.

## Start

```powershell
cd llm-proxy
.\start_proxy.ps1                # thinking off (fast) - default
.\start_proxy.ps1 -Thinking on   # thinking on (model default)
```

Ollama must already be running on 11434. The proxy reuses the omnivoice venv
rather than building another one.

## AIRI setting

`設定 → 服務來源 → Chat 分頁 → Ollama → Base URL`

```
http://localhost:11435/v1/
```

That is the only change. Point it back at `http://localhost:11434/v1/` to bypass
the proxy entirely.

## Switch thinking without restarting

```
curl http://127.0.0.1:11435/config
curl -X POST http://127.0.0.1:11435/config -H "Content-Type: application/json" -d "{\"thinking\":\"on\"}"
curl -X POST http://127.0.0.1:11435/config -H "Content-Type: application/json" -d "{\"thinking\":\"off\"}"
```

## What it does and does not touch

- Injects `reasoning_effort: "none"` **only** when thinking is off, and only when
  the caller has not already set that field.
- Everything else — `messages`, `model`, `tools`, `stream`, `temperature` — is
  forwarded unchanged.
- `/v1/models` and every other path pass straight through, so AIRI's provider
  validation and model list keep working.

## Verified

`verify_proxy.py` checks all of it on the real path (port 11435,
`/v1/chat/completions`, streaming):

| Check | Result |
|---|---|
| `/v1/models` through the proxy | qwen3:4b, qwen3:1.7b, qwen3:8b |
| Streaming is incremental, not buffered | 171 chunks over 0.86s, quartiles 0.38/0.60/0.82s — same shape as direct |
| thinking=off | first token 0.146s, **0** reasoning chars |
| thinking=on | first token 1.081s, 330 reasoning chars |
| Extra hop cost | **+0.003s** |

## Timing log

`timing.jsonl` gets one line per chat request with four wall-clock instants
(received / forwarded / first upstream byte / finished), so proxy overhead stays
separable from Ollama's own time.
