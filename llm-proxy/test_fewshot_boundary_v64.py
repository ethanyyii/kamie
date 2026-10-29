# -*- coding: utf-8 -*-
# v64 E-2: 驗證兩個新的 fewshot_mode（分界句改 user／併進第一則 system）對真實 payload 做了什麼，
# 以及其他所有模式的輸出跟 v63 備份（reasoning_proxy.BACKUP-v63.py）逐位元組相同。
# 不啟動服務、不碰 GPU、不動真的基準線檔。輸入＝v57 從生產路徑抓下來的 payload-007，開關＝start_proxy.ps1 出貨值。
import importlib.util
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
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import reasoning_proxy as P  # noqa: E402
P._settings["reply_cap"] = "off"   # v84: the reply ceiling adds max_tokens by design; its own test is test_reply_cap_v84.py

_spec = importlib.util.spec_from_file_location("proxy_v63", HERE / "reasoning_proxy.BACKUP-v63.py")
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

ROOT = HERE.parent
BASE = json.loads((ROOT / "mainline-v57" / "capture" / "payload-007.json").read_text(encoding="utf-8"))
SHIPPING = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "fewshot_mode": "messages+boundary",
            "sampling": "off", "model_override": "qwen3.5:4b", "strip_injections": "off", "restore_markers": "off",
            "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：", "user_frame": "",
            "user_persona_position": "tail", "fewshot_user_prefix": "off", "fewshot_prefix_skip_named": "on",
            "anchor_seed": "", "format_rules": "card"}
fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


def run(mod, mode, fr="card", base=BASE):
    mod._settings.update(SHIPPING)
    mod._settings["fewshot_mode"] = mode
    mod._settings["format_rules"] = fr
    out = mod._rewrite_all(json.dumps(base, ensure_ascii=False).encode("utf-8"))
    return json.loads(out[0]), out


def later_system(body):
    return [i for i, m in enumerate(body["messages"]) if i > 0 and m["role"] == "system"]


real = BASE["messages"][1:]
print(f"payload-007: {len(real)} real message(s) after the system prompt")

Pb, Pout = run(P, "messages+boundary")
Bb, Bout = run(P, "messages+boundary_user")
Hb, Hout = run(P, "messages+boundary_head")
Mb, Mout = run(P, "messages")
PM, BM, HM, MM = Pb["messages"], Bb["messages"], Hb["messages"], Mb["messages"]
demo = [i for i in range(1, 17)]

print("1) 出貨（messages+boundary）：示範 16 則之後有一則中途 system＝分界句")
check("16 demo turns alternate user/assistant", all(PM[i]["role"] == ("user" if i % 2 else "assistant") for i in demo))
check("exactly one later system message, at index 17", later_system(Pb) == [17], str(later_system(Pb)))
check("it is the boundary sentence", PM[17]["content"] == P._FEWSHOT_BOUNDARY)

print("2) B（messages+boundary_user）：同一句、同一個位置，角色改 user；其餘逐位元組同出貨")
check("no system message after the first", later_system(Bb) == [], str(later_system(Bb)))
check("index 17 is the boundary as a user message", BM[17] == {"role": "user", "content": P._FEWSHOT_BOUNDARY})
check("boundary carries no user-name prefix", "User：" not in BM[17]["content"])
check("index 18 is the real first user turn and it IS prefixed", BM[18]["role"] == "user" and "User：" in BM[18]["content"], BM[18]["content"][:40])
swapped = json.loads(json.dumps(Bb, ensure_ascii=False))
swapped["messages"][17]["role"] = "system"
check("B == production except that one role", swapped == Pb)
check("converted flag set", Bout[3] is True)

print("3) H（messages+boundary_head）：沒有任何中途訊息，說明在第一則 system 的最後、persona 之前")
note = P._FEWSHOT_BOUNDARY_HEAD.format(n=16, m=17)
check("no system message after the first", later_system(Hb) == [], str(later_system(Hb)))
check("one message fewer than production", len(HM) == len(PM) - 1, f"{len(HM)} vs {len(PM)}")
check("demo turns identical to production", HM[1:17] == PM[1:17])
check("real turns identical to production", HM[17:] == PM[18:])
hs, ps = HM[0]["content"], PM[0]["content"]
check("the note appears exactly once in the system text", hs.count(note) == 1)
check("system text == production system text + the note (nothing else moved)", hs.replace("\n\n" + note, "", 1) == ps)
persona = SHIPPING["user_persona"]
check("note sits before the persona paragraph", 0 <= hs.find(note) < hs.rfind(persona), f"note@{hs.find(note)} persona@{hs.rfind(persona)}")
check("note counts the real number of demo messages (16 -> real talk from 17)", "前 16 則" in note and "第 17 則" in note)
check("boundary sentence itself is gone", P._FEWSHOT_BOUNDARY not in json.dumps(Hb, ensure_ascii=False))

print("4) M（messages，陽性對照）：沒有分界句")
check("no system message after the first", later_system(Mb) == [])
check("no boundary text anywhere", P._FEWSHOT_BOUNDARY not in json.dumps(Mb, ensure_ascii=False))
check("real turns follow the demos directly", MM[17:] == PM[18:] and MM[:17] == PM[:17])

print("5) 跟 format_rules=short 並用：S 臂的規則縮短在 B／H 下一樣生效、仍沒有中途 system")
for mode in ("messages+boundary_user", "messages+boundary_head"):
    b, _ = run(P, mode, "short")
    s0 = b["messages"][0]["content"]
    check(f"{mode}+short: short rules in the system text, full block gone",
          P._FORMAT_SHORT in s0 and P._FORMAT_BEGIN not in s0.replace(P._FORMAT_SHORT, ""))
    check(f"{mode}+short: no later system", later_system(b) == [])

print("6) 其他所有模式 × 四種 format_rules：跟 v63 備份逐位元組相同（新模式沒有動到任何舊路徑）")
for mode in ("messages", "messages+boundary", "single_user", "system_tagged", "system_lines"):
    for fr in ("card", "short", "separate", "off"):
        new, _ = run(P, mode, fr)
        old, _ = run(OLD, mode, fr)
        check(f"{mode} / {fr}: identical to v63", json.dumps(new, ensure_ascii=False) == json.dumps(old, ensure_ascii=False))

print("7) /config 接受新值、拒絕亂值")
P._settings.update(SHIPPING)
r = P.set_config({"fewshot_mode": "messages+boundary_user"})
check("accepts messages+boundary_user", isinstance(r, dict) and r["fewshot_mode"] == "messages+boundary_user")
r = P.set_config({"fewshot_mode": "messages+boundary_head"})
check("accepts messages+boundary_head", isinstance(r, dict) and r["fewshot_mode"] == "messages+boundary_head")
r = P.set_config({"fewshot_mode": "boundary_nonsense"})
check("rejects an unknown mode with 400", getattr(r, "status_code", None) == 400)
check("rejected value did not stick", P._settings["fewshot_mode"] == "messages+boundary_head")
P._settings.update(SHIPPING)
check("shipping default is still messages+boundary", P._settings["fewshot_mode"] == "messages+boundary")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
