# -*- coding: utf-8 -*-
# v84 B-1 #4：回覆長度上限（reply_cap）的離線測試。不啟動服務、不碰 GPU、不寫真的紀錄檔（全部環境變數指到暫存資料夾）。
# 參照＝這支轉接層上限關掉的輸出（歷史 off 時再斷言它＝BACKUP-v83 逐位元組；那份備份是歷史那一層之前的檔，沒有 hist 參數）。
# 輸入＝v83 測試用的同一批真實 AIRI 請求（mainline-v82\capture-e，St／N 兩種殼、有前文與沒前文），開關＝start_proxy.ps1 出貨值。
# 🔴 關鍵斷言：
#   1) 開（預設）：轉送的請求＝BACKUP-v83 的轉送請求＋「只多一個欄位」max_tokens＝上限值；其他每一個欄位、每一則訊息逐位元組相同
#      （所有樣本 × 兩種示範送法 × 四種 format_rules × 歷史 off／record）
#   2) 關：＝BACKUP-v83 逐位元組
#   3) 思考開（thinking=on，不注入 reasoning_effort）⇒ 不加上限（Minecraft 那條線）；呼叫端自帶 reasoning_effort≠none ⇒ 不加；
#      呼叫端自帶 max_tokens／max_completion_tokens ⇒ 原樣、不覆寫
#   4) 碰到就叫：finish_reason=length 且有上限 ⇒ reply_cap_hits＋1、事件檔一行、/health 帶 reply_cap_last；
#      length 但沒上限（context 滿）⇒ length_finishes＋1、事件 kind=length_other；stop ⇒ 什麼都不記
#   5) /config：reply_cap on／off、reply_cap_tokens 整數 ≥1，壞值被拒且不生效
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
             ("LLM_PROXY_HISTORY_GIST_MEMORY", "gist-memory.jsonl"), ("LLM_PROXY_REPLY_CAP_LOG", "reply-cap.jsonl")):
    os.environ[k] = str(TMP / f)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import reasoning_proxy as P  # noqa: E402

_spec = importlib.util.spec_from_file_location("proxy_v83c", HERE / "reasoning_proxy.BACKUP-v83.py")
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

ROOT = HERE.parent
CAP = ROOT / "mainline-v82" / "capture-e"
SAMPLES = {("St", "S1b#0"): 141, ("St", "S1b#5"): 146, ("St", "S1b#33"): 174, ("St", "CTX#0"): 703, ("St", "CTX#22"): 725,
           ("N", "S1b#0"): 73, ("N", "S1b#5"): 78, ("N", "S1b#33"): 106, ("N", "CTX#0"): 743, ("N", "CTX#22"): 765}
RAW = {k: (CAP / f"payload-{n:03d}.json").read_bytes() for k, n in SAMPLES.items()}
SHIPPING = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "fewshot_mode": "messages+boundary",
            "sampling": "on", "sampling_temperature": 0.7, "sampling_top_p": 0.8, "model_override": "qwen3.5:4b",
            "strip_injections": "off", "strip_widget": "on", "fix_markers": "on", "restore_markers": "off",
            "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：", "user_frame": "",
            "user_persona_position": "tail", "fewshot_user_prefix": "off", "fewshot_prefix_skip_named": "on",
            "anchor_seed": "", "format_rules": "card", "history_profile": "off"}
MODE = {"St": "messages+boundary", "N": "drop"}
fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


def setmode(mod, arm, fr="card", cap="on", tokens=512):
    mod._settings.update(SHIPPING)
    mod._settings["fewshot_mode"] = MODE[arm]
    mod._settings["format_rules"] = fr
    if mod is P:
        mod._settings["reply_cap"] = cap
        mod._settings["reply_cap_tokens"] = tokens


print("0) 出貨預設")
check("default reply_cap on", os.environ.get("LLM_PROXY_REPLY_CAP") is None and P._settings["reply_cap"] == "on")
check("default reply_cap_tokens 512", P._settings["reply_cap_tokens"] == 512)

print("1) 開＝v83 備份＋只多 max_tokens；2) 關＝v83 備份逐位元組")
n_on = n_off = n_all = 0
bad = []
for (arm, key) in SAMPLES:
    for fr in ("card", "short", "separate", "off"):
        for prof in (None, "record"):
            hist = None if prof is None else (prof, P._HISTORY_PROFILES[prof], {})
            # reference = this proxy with the ceiling off; for history off it must also equal BACKUP-v83
            # (the pre-history-layer file, which has no `hist` argument) byte for byte
            setmode(P, arm, fr, cap="off")
            old = P._rewrite_all(RAW[(arm, key)], hist=hist, hist_info={})
            if prof is None:
                setmode(OLD, arm, fr)
                if OLD._rewrite_all(RAW[(arm, key)]) != old:
                    bad.append(("cap-off != BACKUP-v83", arm, key, fr))
            setmode(P, arm, fr, cap="on", tokens=777)
            info = {}
            new = P._rewrite_all(RAW[(arm, key)], hist=hist, hist_info={}, cap_info=info)
            ob, nb = json.loads(old[0]), json.loads(new[0])
            extra = {k: nb[k] for k in nb if k not in ob}
            same_rest = all(ob[k] == nb[k] for k in ob) and set(ob) <= set(nb)
            ok_on = (extra == {"max_tokens": 777} and same_rest and info.get("cap") == 777 and old[1:] == new[1:]
                     and json.dumps(ob["messages"], ensure_ascii=False) == json.dumps(nb["messages"], ensure_ascii=False))
            setmode(P, arm, fr, cap="off")
            info2 = {}
            new_off = P._rewrite_all(RAW[(arm, key)], hist=hist, hist_info={}, cap_info=info2)
            ok_off = (new_off == old and info2.get("cap") == 0)
            n_all += 1
            n_on += int(ok_on)
            n_off += int(ok_off)
            if not (ok_on and ok_off):
                bad.append((arm, key, fr, prof, extra))
check("on = BACKUP-v83 + max_tokens only (%d cases)" % n_all, n_on == n_all, "%d/%d %s" % (n_on, n_all, bad[:2]))
check("off = reference byte-identical (%d cases), and history-off reference = BACKUP-v83" % n_all, n_off == n_all and not bad, "%d/%d %s" % (n_off, n_all, bad[:2]))

print("3) 思考開、呼叫端自帶的值")
setmode(P, "St", cap="on")
P._settings["thinking"] = "on"
b = json.loads(P._rewrite_all(RAW[("St", "S1b#5")], cap_info={})[0])
check("thinking=on: no max_tokens, no reasoning_effort", "max_tokens" not in b and "reasoning_effort" not in b)
P._settings["thinking"] = "off"
raw = json.loads(RAW[("St", "S1b#5")])
raw["reasoning_effort"] = "high"
b = json.loads(P._rewrite_all(json.dumps(raw, ensure_ascii=False).encode("utf-8"), cap_info={})[0])
check("caller reasoning_effort=high: kept, no max_tokens", b.get("reasoning_effort") == "high" and "max_tokens" not in b)
for fld in ("max_tokens", "max_completion_tokens"):
    raw = json.loads(RAW[("St", "S1b#5")])
    raw[fld] = 50
    info = {}
    b = json.loads(P._rewrite_all(json.dumps(raw, ensure_ascii=False).encode("utf-8"), cap_info=info)[0])
    check("caller %s wins" % fld, b.get(fld) == 50 and info.get("cap") == 0 and ("max_tokens" not in b or fld == "max_tokens"))
# nothing else active + cap on + thinking off => the cap is the only rewrite and it still applies
for k in ("strip_tools", "fewshot_messages", "sampling", "strip_widget", "fix_markers"):
    P._settings[k] = "off"
P._settings["model_override"] = ""
P._settings["user_persona"] = P._settings["user_name_prefix"] = ""
info = {}
b = json.loads(P._rewrite_all(RAW[("St", "CTX#0")], cap_info=info)[0])
check("cap alone still applies", b.get("max_tokens") == 512 and info["cap"] == 512 and b.get("reasoning_effort") == "none")
setmode(P, "St")

print("4) 碰到就叫")
c0 = dict(P._reply_cap_counters)
ev = P._reply_cap_note(1, 1024, "stop", {"completion_tokens": 30, "prompt_tokens": 900}, "短", "qwen3.5:4b")
check("stop: nothing recorded", ev is None and P._reply_cap_counters["reply_cap_hits"] == c0["reply_cap_hits"]
      and P._reply_cap_counters["length_finishes"] == c0["length_finishes"] and P._reply_cap_counters["reply_cap_applied"] == c0["reply_cap_applied"] + 1)
ev = P._reply_cap_note(2, 1024, "length", {"completion_tokens": 1024, "prompt_tokens": 1500}, "幹嘛" * 600, "pilot:v82")
check("length on our cap: hit", ev["kind"] == "reply_cap_hit" and P._reply_cap_counters["reply_cap_hits"] == c0["reply_cap_hits"] + 1)
ev = P._reply_cap_note(3, 0, "length", {"completion_tokens": 7229, "prompt_tokens": 963}, "x", "pilot:v82")
check("length without cap: length_other", ev["kind"] == "length_other" and P._reply_cap_counters["length_finishes"] == c0["length_finishes"] + 2
      and P._reply_cap_counters["reply_cap_hits"] == c0["reply_cap_hits"] + 1)
lines = [json.loads(l) for l in open(os.environ["LLM_PROXY_REPLY_CAP_LOG"], encoding="utf-8")]
check("event file: 2 lines (hit + length_other)", [l["kind"] for l in lines] == ["reply_cap_hit", "length_other"], str([l["kind"] for l in lines]))
h = P.health()
check("/health carries reply_cap_hits / length_finishes / reply_cap_last",
      h["reply_cap_hits"] >= 1 and h["length_finishes"] >= 2 and h["reply_cap_last"]["seq"] == 2)

print("5) /config")
r = P.set_config({"reply_cap_tokens": 600})
check("reply_cap_tokens 600 accepted", P._settings["reply_cap_tokens"] == 600)
for badv in (0, -1, "x"):
    r = P.set_config({"reply_cap_tokens": badv})
    check("reply_cap_tokens %r rejected, value kept" % (badv,), getattr(r, "status_code", 200) == 400 and P._settings["reply_cap_tokens"] == 600)
r = P.set_config({"reply_cap": "maybe"})
check("reply_cap bad value rejected", getattr(r, "status_code", 200) == 400 and P._settings["reply_cap"] == "on")
P.set_config({"reply_cap": "off"})
check("reply_cap off accepted", P._settings["reply_cap"] == "off")
P.set_config({"reply_cap": "on", "reply_cap_tokens": 512})

print("6) v84 E：pad_ms（盲聊延遲補齊；預設 0、只影響時間）")
check("default pad_ms 0", P._settings["pad_ms"] == 0)
P.set_config({"pad_ms": 300})
check("pad_ms 300 accepted", P._settings["pad_ms"] == 300)
for badv in (-1, 20000, "x"):
    r = P.set_config({"pad_ms": badv})
    check("pad_ms %r rejected, value kept" % (badv,), getattr(r, "status_code", 200) == 400 and P._settings["pad_ms"] == 300)
P.set_config({"pad_ms": 0})
setmode(P, "St", cap="on")
b0 = P._rewrite_all(RAW[("St", "S1b#5")], cap_info={})[0]
P.set_config({"pad_ms": 450})
b1 = P._rewrite_all(RAW[("St", "S1b#5")], cap_info={})[0]
P.set_config({"pad_ms": 0})
check("pad_ms never touches the forwarded body", b0 == b1)

print()
print("ALL PASS" if not fails else "FAILED: %s" % fails)
sys.exit(1 if fails else 0)
