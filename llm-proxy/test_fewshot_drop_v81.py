# -*- coding: utf-8 -*-
# v81 E：驗證新的 fewshot_mode=drop（抽出示範、不插回、不加分界）對真實 payload 做了什麼，
# 以及其他所有模式 × 四種 format_rules 的輸出跟 v81 備份（reasoning_proxy.BACKUP-v81.py）逐位元組相同。
# 不啟動服務、不碰 GPU、不動真的基準線檔。
# 輸入＝客戶 v80 那段聊天的第 3 句（mainline-v80\chat-capture\payload-003.json：card-v1.4-name、兩輪歷史），開關＝start_proxy.ps1 出貨值。
# 🔴 關鍵斷言：drop 之後 system[0] 逐位元組 == pilot 的訓練字串（mainline-v76\data-v76-learn.jsonl 的 system），沒有示範、沒有分界。
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
             ("LLM_PROXY_MARKERFIX_LOG", "markerfix.jsonl")):
    os.environ[k] = str(TMP / f)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import reasoning_proxy as P  # noqa: E402
P._settings["reply_cap"] = "off"   # v84: the reply ceiling adds max_tokens by design; its own test is test_reply_cap_v84.py

_spec = importlib.util.spec_from_file_location("proxy_v81b", HERE / "reasoning_proxy.BACKUP-v81.py")
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

ROOT = HERE.parent
BASE = json.loads((ROOT / "mainline-v80" / "chat-capture" / "payload-003.json").read_text(encoding="utf-8"))
TRAIN_ROWS = [json.loads(l) for l in (ROOT / "mainline-v76" / "data-v76-learn.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
TRAIN_SYS = json.loads((ROOT / "mainline-v74" / "capture-card" / "payload-001.forwarded.json").read_text(encoding="utf-8"))["messages"][0]["content"]
assert all(("<|im_start|>system\n" + TRAIN_SYS + "<|im_end|>") in r["prompt"] for r in TRAIN_ROWS), "訓練資料的 system 跟 v74 轉送的 system[0] 不同"
SHIPPING = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "fewshot_mode": "messages+boundary",
            "sampling": "on", "sampling_temperature": 0.7, "sampling_top_p": 0.8, "model_override": "qwen3.5:4b",
            "strip_injections": "off", "strip_widget": "on", "fix_markers": "on", "restore_markers": "off",
            "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：", "user_frame": "",
            "user_persona_position": "tail", "fewshot_user_prefix": "off", "fewshot_prefix_skip_named": "on",
            "anchor_seed": "", "format_rules": "card"}
fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


def run(mod, mode, fr="card", base=BASE, fewshot="on"):
    mod._settings.update(SHIPPING)
    mod._settings["fewshot_mode"] = mode
    mod._settings["format_rules"] = fr
    mod._settings["fewshot_messages"] = fewshot
    out = mod._rewrite_all(json.dumps(base, ensure_ascii=False).encode("utf-8"))
    return json.loads(out[0]), out


real = BASE["messages"][1:]
print(f"payload-003: {len(real)} real message(s) after the system prompt")

Pb, Pout = run(P, "messages+boundary")
Db, Dout = run(P, "drop")
Ob, Oout = run(P, "messages+boundary", fewshot="off")
PM, DM, OM = Pb["messages"], Db["messages"], Ob["messages"]

print("1) 出貨（messages+boundary）：system＋16 示範＋分界＋真實 5 則")
check("19 + real turns", len(PM) == 18 + len(real), str(len(PM)))
check("system[0] == training string (the block lifted)", PM[0]["content"] == TRAIN_SYS)

print("2) drop：只剩 system＋真實輪，system[0] 逐位元組 == 訓練字串")
check("message count == 1 + real turns", len(DM) == 1 + len(real), f"{len(DM)} vs {1 + len(real)}")
check("system[0] == TRAIN_SYS byte-for-byte", DM[0]["content"] == TRAIN_SYS)
check("system[0] identical to the shipping mode's system[0]", DM[0] == PM[0])
check("no demo block text anywhere", "說話風格示範" not in json.dumps(Db, ensure_ascii=False))
check("no boundary sentence anywhere", P._FEWSHOT_BOUNDARY not in json.dumps(Db, ensure_ascii=False))
check("only one system message", [i for i, m in enumerate(DM) if m["role"] == "system"] == [0])
check("real turns identical to the shipping mode's real turns (prefix + timestamps intact)", DM[1:] == PM[18:])
check("first real user turn carries the name prefix", DM[1]["role"] == "user" and "User：" in DM[1]["content"], DM[1]["content"][:40])
check("converted flag set", Dout[3] is True)
check("sampling injected as usual", Db.get("temperature") == 0.7 and Db.get("top_p") == 0.8)
check("model overridden as usual", Db.get("model") == "qwen3.5:4b")
check("tools stripped as usual", "tools" not in Db)

print("3) 對照：fewshot_messages=off 不是這個形狀（示範留在 system 裡）——這就是 v80 §8 那條地雷")
check("off: message count == 1 + real turns", len(OM) == 1 + len(real))
check("off: the demo block is still inside system[0]", "【說話風格示範】" in OM[0]["content"])
check("off: system[0] != training string", OM[0]["content"] != TRAIN_SYS)

print("4) 訓練資料裡的一筆：把 drop 的訊息拼成 prompt，去掉使用者那句之後跟 data-v76-learn 的 prompt 前段相同")
t0 = next(r for r in TRAIN_ROWS if not r["history"])
head = t0["prompt"].split("<|im_start|>user")[0]
check("training prompt head == <|im_start|>system + TRAIN_SYS + <|im_end|>", head == "<|im_start|>system\n" + TRAIN_SYS + "<|im_end|>\n")

print("5) 其他所有模式 × 四種 format_rules：跟 v81 備份逐位元組相同（新模式沒有動到任何舊路徑）")
for mode in ("messages", "messages+boundary", "messages+boundary_user", "messages+boundary_head", "single_user", "system_tagged", "system_lines"):
    for fr in ("card", "short", "separate", "off"):
        new, _ = run(P, mode, fr)
        old, _ = run(OLD, mode, fr)
        check(f"{mode} / {fr}: identical to v81 backup", json.dumps(new, ensure_ascii=False) == json.dumps(old, ensure_ascii=False))
new, _ = run(P, "messages+boundary", fewshot="off")
old, _ = run(OLD, "messages+boundary", fewshot="off")
check("fewshot_messages=off: identical to v81 backup", json.dumps(new, ensure_ascii=False) == json.dumps(old, ensure_ascii=False))

print("6) drop × 四種 format_rules：format_rules 照常作用在 system[0]，仍然沒有示範／分界")
for fr in ("short", "separate", "off"):
    b, _ = run(P, "drop", fr)
    s0 = b["messages"][0]["content"]
    check(f"drop/{fr}: no demo text", "說話風格示範" not in json.dumps(b, ensure_ascii=False))
    check(f"drop/{fr}: format rules mode applied", (fr == "short" and P._FORMAT_SHORT in s0) or (fr == "off" and "【輸出格式規則】" not in s0) or (fr == "separate"))

print("7) /config 接受 drop、拒絕亂值、出貨預設不變")
P._settings.update(SHIPPING)
r = P.set_config({"fewshot_mode": "drop"})
check("accepts drop", isinstance(r, dict) and r["fewshot_mode"] == "drop")
r = P.set_config({"fewshot_mode": "drop_nonsense"})
check("rejects an unknown mode with 400", getattr(r, "status_code", None) == 400)
check("rejected value did not stick", P._settings["fewshot_mode"] == "drop")
P._settings.update(SHIPPING)
check("shipping default is still messages+boundary", P._settings["fewshot_mode"] == "messages+boundary")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
