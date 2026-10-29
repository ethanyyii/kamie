"""Acceptance checks for the reasoning proxy, run against the proxy itself.

The point of every check here is that it is done on the path AIRI actually uses
(port 11435, `/v1/chat/completions`, streaming), not on an equivalent-looking
one — the whole reason this proxy exists is that a check on Ollama's native
endpoint gave the wrong answer in v13.

Checks:
  1. `/v1/models` still answers, so AIRI's provider validation passes.
  2. The response is genuinely incremental. A proxy that buffered would still
     "work" and would still return the right text, but it would destroy the
     latency it was built to fix. Measured by the spread between the first and
     last chunk arrival: buffered means they all land at once.
  3. thinking=off produces zero reasoning characters.
  4. thinking=on restores the model default.
  5. Direct-to-Ollama vs through-the-proxy, same request, to price the extra hop.
"""

import json
import statistics
import time
import urllib.request

PROXY = "http://127.0.0.1:11435"
DIRECT = "http://127.0.0.1:11434"
MODEL = "qwen3:1.7b"
MSG = [{"role": "user", "content": "你好，簡單說一句話就好"}]


def stream(base, extra=None):
    body = {"model": MODEL, "messages": MSG, "stream": True}
    if extra:
        body.update(extra)
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    arrivals, reasoning, content, first_content = [], 0, "", None
    with urllib.request.urlopen(req) as r:
        for line in r:
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data: ") or s == "data: [DONE]":
                continue
            arrivals.append(time.perf_counter() - t0)
            d = json.loads(s[6:]).get("choices", [{}])[0].get("delta", {}) or {}
            reasoning += len(d.get("reasoning") or d.get("reasoning_content") or "")
            c = d.get("content") or ""
            if c and first_content is None:
                first_content = time.perf_counter() - t0
            content += c
    return {"first_content": first_content, "reasoning": reasoning, "content": content,
            "arrivals": arrivals, "total": time.perf_counter() - t0}


def config(thinking):
    req = urllib.request.Request(f"{PROXY}/config",
                                 data=json.dumps({"thinking": thinking}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main():
    print("1. /v1/models through the proxy")
    with urllib.request.urlopen(f"{PROXY}/v1/models") as r:
        models = json.loads(r.read())
    print(f"   {[m['id'] for m in models.get('data', [])]}\n")

    print("2. streaming really is incremental (thinking=off)")
    config("off")
    stream(PROXY)
    s = stream(PROXY)
    a = s["arrivals"]
    print(f"   {len(a)} SSE chunks | first at {a[0]:.3f}s | last at {a[-1]:.3f}s "
          f"| spread {a[-1] - a[0]:.3f}s")
    print(f"   verdict: {'STREAMING' if (a[-1] - a[0]) > 0.05 else 'BUFFERED - would hurt latency'}\n")

    print("3. thinking=off -> reasoning must be 0")
    rows = [stream(PROXY) for _ in range(3)]
    print(f"   first content {statistics.median([r['first_content'] for r in rows]):.3f}s "
          f"| reasoning {statistics.median([r['reasoning'] for r in rows]):.0f} chars")
    print(f"   sample: {rows[0]['content'][:60]!r}\n")

    print("4. thinking=on -> model default restored")
    config("on")
    stream(PROXY)
    rows_on = [stream(PROXY) for _ in range(3)]
    print(f"   first content {statistics.median([r['first_content'] for r in rows_on]):.3f}s "
          f"| reasoning {statistics.median([r['reasoning'] for r in rows_on]):.0f} chars")
    print(f"   sample: {rows_on[0]['content'][:60]!r}\n")

    print("5. cost of the extra hop (both with reasoning off, same request)")
    config("off")
    stream(PROXY)
    stream(DIRECT, {"reasoning_effort": "none"})
    via = [stream(PROXY)["first_content"] for _ in range(5)]
    direct = [stream(DIRECT, {"reasoning_effort": "none"})["first_content"] for _ in range(5)]
    mv, md = statistics.median(via), statistics.median(direct)
    print(f"   through proxy {mv:.3f}s | direct to Ollama {md:.3f}s | overhead {mv - md:+.3f}s")

    config("off")
    print("\nproxy left at thinking=off")


if __name__ == "__main__":
    main()
