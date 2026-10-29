# -*- coding: utf-8 -*-
# v89 B-2：思考開的請求改送 think_model（轉接層 `think_model`）的離線測試。不啟動服務、不碰 GPU、不寫真的紀錄檔（環境變數全部指到暫存資料夾）。
# 參照＝reasoning_proxy.BACKUP-v88.py（改之前逐位元組的備份）。輸入＝客戶 v88 盲聊的真實 AIRI 請求（mainline-v88\chat-capture\payload-NNN.json）。
# 🔴 關鍵斷言：
#   1) think_model=""（關）：轉送的請求＝v88 逐位元組（兩組預設：ft4b 那組 drop／v1.5 那組 messages+boundary）
#   2) think_model 開、聊天請求（轉接層注入 reasoning_effort none）：也＝v88 逐位元組，計數不動
#   3) 思考開的六種形狀都改送 think_model：呼叫端 reasoning_effort=high／low、全域 thinking=on（不注入）、reasoning:{effort:low}（Minecraft 那條線的形狀）、
#      原生 api/chat 帶 think=true、原生 api/chat 沒有 think 欄位；除了 model 以外每個欄位＝v88 逐位元組；計數＋1、事件檔一行、/health 帶 think_route_last
#   4) 不改送：原生 think=false；model_override 空（不是我們換的模型）；think_model＝model_override
#   5) /config：think_model 設、清空
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
for k, f in (("LLM_PROXY_TOOLS_BASELINE", "baseline.json"), ("LLM_PROXY_MARKER_MEMORY", "marker.jsonl"),
             ("LLM_PROXY_LOG", "timing.jsonl"), ("LLM_PROXY_IDENTITY_EVENTS", "identity.jsonl"),
             ("LLM_PROXY_TOOL_EVENTS", "tools.jsonl"), ("LLM_PROXY_SWALLOW_LOG", "swallow.jsonl"),
             ("LLM_PROXY_MARKERFIX_LOG", "markerfix.jsonl"), ("LLM_PROXY_HISTORY_EVENTS", "history-events.jsonl"),
             ("LLM_PROXY_HISTORY_GIST_MEMORY", "gist-memory.jsonl"), ("LLM_PROXY_REPLY_CAP_LOG", "reply-cap.jsonl"),
             ("LLM_PROXY_THINK_ROUTE_LOG", "think-route.jsonl")):
    os.environ[k] = str(TMP / f)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import reasoning_proxy as P  # noqa: E402

_spec = importlib.util.spec_from_file_location("proxy_v88", HERE / "reasoning_proxy.BACKUP-v88.py")
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

CAP = HERE.parent / "mainline-v88" / "chat-capture"
RAWS = [p.read_bytes() for p in sorted(CAP.glob("payload-[0-9][0-9][0-9].json"))]
assert len(RAWS) >= 50, len(RAWS)

SHIP = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "sampling": "on", "sampling_temperature": 0.7, "sampling_top_p": 0.8,
        "strip_widget": "on", "fix_markers": "on", "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：",
        "history_profile": "off", "reply_cap": "on", "reply_cap_tokens": 512, "pad_ms": 0}
PROFILES = {"ft4b": {"model_override": "v87-qft:q4km", "fewshot_mode": "drop"},
            "v1.5": {"model_override": "qwen3.5:4b", "fewshot_mode": "messages+boundary"}}
n_ok = 0


def setboth(**kw):
    for m in (P, OLD):
        m._settings.update(SHIP)
        m._settings.update(kw)
    P._settings["think_model"] = kw.get("think_model", "")


def run(mod, raw, path="v1/chat/completions"):
    if mod is P:
        info = {}
        out = P._rewrite_all(raw, hist=("off", None, {}), hist_info={}, cap_info={}, route_info=info, path=path)
        return out[0], info.get("route")
    return OLD._rewrite_all(raw, hist=("off", None, {}), hist_info={}, cap_info={})[0], None


def check(cond, what):
    global n_ok
    assert cond, what
    n_ok += 1


# 1)＋2) 預設關、開著時聊天請求：逐位元組＝v88
for name, prof in PROFILES.items():
    for tm in ("", "qwen3.5:4b"):
        if tm == prof["model_override"]:
            continue
        setboth(**prof, think_model=tm)
        before = P._think_route_counters["think_routed"]
        for raw in RAWS:
            new, route = run(P, raw)
            old, _ = run(OLD, raw)
            check(new == old and route is None, f"{name} think_model={tm!r}: chat request differs from v88")
        check(P._think_route_counters["think_routed"] == before, "chat requests must not count")

# 3) 思考開的形狀 → 改送
setboth(**PROFILES["ft4b"], think_model="qwen3.5:4b")


def mutate(raw, **fields):
    b = json.loads(raw)
    b.update(fields)
    return json.dumps(b, ensure_ascii=False).encode("utf-8")


def same_but_model(new, old, want_model):
    a, b = json.loads(new), json.loads(old)
    check(a["model"] == want_model, f"model {a['model']} != {want_model}")
    a.pop("model"), b.pop("model")
    check(a == b, "only the model may differ from v88")


sample = RAWS[-1]
assert json.loads(sample).get("think") is False      # AIRI 的「深度思考模式」＝disable ⇒ 每一則都帶 think:false（v88 盲聊的真實請求）
_nt = json.loads(sample)
_nt.pop("think")
no_think = json.dumps(_nt, ensure_ascii=False).encode("utf-8")
cases = [("caller reasoning_effort=high", mutate(sample, reasoning_effort="high"), "v1/chat/completions", {}),
         ("caller reasoning_effort=low", mutate(sample, reasoning_effort="low"), "v1/chat/completions", {}),
         ("reasoning.effort=low (Minecraft shape)", mutate(sample, reasoning={"effort": "low"}), "v1/chat/completions", {}),
         ("global thinking=on", sample, "v1/chat/completions", {"thinking": "on"}),
         ("native think=true", mutate(sample, think=True), "api/chat", {}),
         ("native no think field", no_think, "api/chat", {}),
         ("native think=true (AIRI 思考模式 enable)", mutate(sample, think=True), "v1/chat/completions", {"thinking": "on"})]
for what, raw, path, extra in cases:
    setboth(**PROFILES["ft4b"], think_model="qwen3.5:4b", **extra)
    c0 = P._think_route_counters["think_routed"]
    lines0 = (TMP / "think-route.jsonl").read_text(encoding="utf-8").count("\n") if (TMP / "think-route.jsonl").exists() else 0
    new, route = run(P, raw, path)
    old, _ = run(OLD, raw)
    same_but_model(new, old, "qwen3.5:4b")
    check(route and route["to"] == "qwen3.5:4b" and route["from"] == "v87-qft:q4km", f"{what}: no route event")
    check(P._think_route_counters["think_routed"] == c0 + 1, f"{what}: counter")
    check((TMP / "think-route.jsonl").read_text(encoding="utf-8").count("\n") == lines0 + 1, f"{what}: event file")
    check(P._think_route_health()["think_route_last"]["why"] == route["why"], f"{what}: /health last")
    print(f"  routed: {what:40s} why={route['why']}")

# 4) 不改送
setboth(**PROFILES["ft4b"], think_model="qwen3.5:4b")
new, route = run(P, mutate(sample, think=False), "api/chat")
check(route is None and json.loads(new)["model"] == "v87-qft:q4km", "native think=false must not route")
setboth(model_override="", fewshot_mode="drop", think_model="qwen3.5:4b")
new, route = run(P, mutate(sample, reasoning_effort="high"))
check(route is None and json.loads(new)["model"] == json.loads(sample)["model"], "no override -> not ours to route")
setboth(**PROFILES["v1.5"], think_model="qwen3.5:4b")
new, route = run(P, mutate(sample, reasoning_effort="high"))
check(route is None and json.loads(new)["model"] == "qwen3.5:4b", "think_model == override -> nothing to do")

# 5) /config
r = P.set_config({"think_model": "qwen3.5:4b"})
check(P._settings["think_model"] == "qwen3.5:4b", "/config set")
r = P.set_config({"think_model": ""})
check(P._settings["think_model"] == "", "/config clear")
check("think_routed" in P.health() and "think_route_last" in P.health(), "/health fields")

print(f"OK {n_ok} checks ({len(RAWS)} real AIRI requests from mainline-v88\\chat-capture)")
