# -*- coding: utf-8 -*-
# v61: 驗證 format_rules 四種模式對 system 文字做的事。不啟動服務、不碰 GPU、不動真的基準線檔。
# 輸入用 v57 從生產路徑抓下來的 payload-007（AIRI 原樣送出的 system 文字：260 前綴＋示範區塊
# ＋人格散文＋1201 字元格式規則＋原廠那句＋441 artistry 後綴），開關全部設成 start_proxy.ps1 的出貨值。
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
for k, f in (("LLM_PROXY_TOOLS_BASELINE", "baseline.json"), ("LLM_PROXY_MARKER_MEMORY", "marker.jsonl"),
             ("LLM_PROXY_LOG", "timing.jsonl"), ("LLM_PROXY_IDENTITY_EVENTS", "identity.jsonl"),
             ("LLM_PROXY_TOOL_EVENTS", "tools.jsonl"), ("LLM_PROXY_SWALLOW_LOG", "swallow.jsonl")):
    os.environ[k] = str(TMP / f)
sys.path.insert(0, str(Path(__file__).parent))

import reasoning_proxy as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = json.loads((ROOT / "mainline-v57" / "capture" / "payload-007.json").read_text(encoding="utf-8"))
SHIPPING = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "fewshot_mode": "messages+boundary",
            "sampling": "off", "model_override": "qwen3.5:4b", "strip_injections": "off", "restore_markers": "off",
            "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：", "user_frame": "",
            "user_persona_position": "tail", "fewshot_user_prefix": "off", "fewshot_prefix_skip_named": "on",
            "anchor_seed": "", "format_rules": "card"}
FACTORY = "最後，做任何你想做的事！"
fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


def run(mode, base=BASE, **over):
    P._settings.update(SHIPPING)
    P._settings.update(over)
    P._settings["format_rules"] = mode
    raw = json.dumps(base, ensure_ascii=False).encode("utf-8")
    out = P._rewrite_all(raw)
    check(f"_rewrite_all returns 10 fields ({mode})", len(out) == 10, f"got {len(out)}")
    body = json.loads(out[0])
    return body, out[9]


def sys_text(body):
    return body["messages"][0]["content"]


def rules_msgs(body):
    return [i for i, m in enumerate(body["messages"]) if i > 0 and m["role"] == "system" and m["content"].startswith(P._FORMAT_BEGIN)]


full_block = None
print("0) 原始 system 文字有兩個標頭、一次原廠那句")
s0 = BASE["messages"][0]["content"]
check("headers present once each", s0.count(P._FORMAT_BEGIN) == 1 and s0.count(P._FORMAT_END) == 1)
full_block = s0[s0.find(P._FORMAT_BEGIN):s0.find(P._FORMAT_END) + len(P._FORMAT_END)]
print(f"     full block {len(full_block)} chars, short block {len(P._FORMAT_SHORT)} chars")
check("short block is really short", len(P._FORMAT_SHORT) < 0.2 * len(full_block))

print("1) card：什麼都不動（與 v60 行為逐位元組相同）")
card, m = run("card")
check("mode applied = ''", m == "")
check("block still in system[0]", full_block in sys_text(card))
check("message count 19 (sys + 16 demos + boundary + user)", len(card["messages"]) == 19, str(len(card["messages"])))
check("no separate rules message", rules_msgs(card) == [])
check("persona paragraph closes system[0]", sys_text(card).rstrip().endswith(SHIPPING["user_persona"]))

print("2) short：換成短版，其餘位置不動")
short, m = run("short")
check("mode applied = 'short'", m == "short")
check("short text in system[0]", P._FORMAT_SHORT in sys_text(short))
check("long text gone (CALL 格式)", "CALL 格式" not in sys_text(short) and "intensity" not in sys_text(short))
check("factory line kept", FACTORY in sys_text(short))
check("factory line still right after the block", (P._FORMAT_END + "\n\n" + FACTORY) in sys_text(short))
check("persona paragraph still closes system[0]", sys_text(short).rstrip().endswith(SHIPPING["user_persona"]))
check("message count unchanged (19)", len(short["messages"]) == 19)
check("personality prose untouched", "你是User創造的。這件事你知道，不否認，也不常提。" in sys_text(short))
check("demo block still lifted (user demo #1 at index 1)", short["messages"][1]["role"] == "user" and short["messages"][1]["content"] == "你是不是有點笨")
saved = len(sys_text(card)) - len(sys_text(short))
print(f"     system[0] shrank by {saved} chars")
check("shrank by exactly full-short", saved == len(full_block) - len(P._FORMAT_SHORT))

print("3) separate：從卡片文字抬出來，變成示範邊界後面的獨立 system 訊息，文字逐字不變")
sep, m = run("separate")
check("mode applied = 'separate'", m == "separate")
check("block gone from system[0]", P._FORMAT_BEGIN not in sys_text(sep))
idx = rules_msgs(sep)
check("exactly one rules message", len(idx) == 1, str(idx))
if idx:
    check("it sits right after the few-shot boundary (index 18)", idx[0] == 18 and sep["messages"][17]["content"] == P._FEWSHOT_BOUNDARY)
    check("its text == the full block, byte for byte", sep["messages"][idx[0]]["content"] == full_block)
    check("the real user turn follows it", sep["messages"][idx[0] + 1]["role"] == "user")
check("message count 20", len(sep["messages"]) == 20)
check("factory line kept in system[0]", FACTORY in sys_text(sep))
check("persona paragraph still closes system[0]", sys_text(sep).rstrip().endswith(SHIPPING["user_persona"]))

print("4) off：整塊拿掉，別的不動")
off, m = run("off")
check("mode applied = 'off'", m == "off")
check("no header anywhere", all(P._FORMAT_BEGIN not in x["content"] for x in off["messages"] if isinstance(x.get("content"), str)))
check("factory line kept", FACTORY in sys_text(off))
check("message count 19", len(off["messages"]) == 19)
check("persona paragraph still closes system[0]", sys_text(off).rstrip().endswith(SHIPPING["user_persona"]))
check("no triple newline left behind", "\n\n\n" not in sys_text(off))

print("5) 冪等：separate 的輸出再過一次，不會多出第二則規則訊息")
again, _ = run("separate", base=sep)
check("still exactly one rules message", len(rules_msgs(again)) == 1, str(rules_msgs(again)))

print("6) fewshot_messages=off 時 separate 落在 system[0] 之後（index 1）")
sep_off, m = run("separate", fewshot_messages="off")
check("mode applied", m == "separate")
check("rules message at index 1", rules_msgs(sep_off) == [1], str(rules_msgs(sep_off)))
check("message count 3 (sys + rules + user)", len(sep_off["messages"]) == 3, str(len(sep_off["messages"])))

print("7) 沒有標頭的卡片（v0.1 形狀）：no-op")
v01 = json.loads(json.dumps(BASE, ensure_ascii=False))
v01["messages"][0]["content"] = s0.replace(P._FORMAT_BEGIN, "").replace(P._FORMAT_END, "")
noop, m = run("off", base=v01)
check("mode applied = ''", m == "")
check("system[0] untouched apart from the lift", "CALL 格式" in sys_text(noop))

print("8) /config 只收四個值")
r = P.set_config({"format_rules": "bogus"})
check("bogus rejected (400)", getattr(r, "status_code", None) == 400)
r = P.set_config({"format_rules": "SHORT"})
check("case-insensitive accept", isinstance(r, dict) and r.get("format_rules") == "short")
P.set_config({"format_rules": "card"})
check("/health carries it", P.health().get("format_rules") == "card")

print()
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
