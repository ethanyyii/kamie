# -*- coding: utf-8 -*-
# v73 task D：驗證 strip_widget 這個窄開關。不啟動服務、不碰 GPU、不動真的基準線檔。
# 輸入用 v71 從**真實桌面版**抓下來的 payload（AIRI 原樣送出的 system：260 前綴＋人格散文
# ＋1201 格式規則＋原廠那句＋441 artistry 後綴），開關設成 start_proxy.ps1 的出貨值。
#
# 要證明的四件事：
#   1. off  ⇒ 與 v71 真實轉送的 system 逐位元組相同（＝出貨行為沒被動到）
#   2. on   ⇒ 441 字元的 artistry 區塊不見了，而 260 字元的 markdown/LaTeX 前綴**還在**
#             （這正是它跟 strip_injections 的差別；否則 D 量到的是兩個改動的合）
#   3. on   ⇒ 轉接層自己加的身分接縫、人格段落、示範區塊一個字都沒少
#   4. 沒有那個區塊時是 no-op；跟 strip_injections 一起開也不會砍兩次
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "mainline-v71" / "capture-card"
BASE = json.loads((CAP / "payload-001.json").read_text(encoding="utf-8"))
REAL_FWD = json.loads((CAP / "payload-001.forwarded.json").read_text(encoding="utf-8"))
SHIPPING = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "fewshot_mode": "messages+boundary",
            "sampling": "off", "model_override": "qwen3.5:4b", "strip_injections": "off", "strip_widget": "off",
            "restore_markers": "off",
            "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：", "user_frame": "",
            "user_persona_position": "tail", "fewshot_user_prefix": "off", "fewshot_prefix_skip_named": "on",
            "anchor_seed": "", "format_rules": "card"}
PREFIX = P._INJECT_PREFIX_LINES[0]
MARKER = P._INJECT_SUFFIX_MARKER
fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


def run(base=BASE, **over):
    P._settings.update(SHIPPING)
    P._settings.update(over)
    raw = json.dumps(base, ensure_ascii=False).encode("utf-8")
    out = P._rewrite_all(raw)
    return json.loads(out[0]), out[6]


def sys_text(body):
    return body["messages"][0]["content"]


s0 = BASE["messages"][0]["content"]
print("0) 原始 system 文字：前綴在、artistry 區塊在、各一次")
check("markdown/LaTeX 前綴在開頭", s0.startswith(PREFIX))
check("artistry 標記出現正好一次", s0.count(MARKER) == 1, f"count={s0.count(MARKER)}")
WIDGET = s0[s0.find(MARKER):]
print(f"     前綴 {len(PREFIX)} 字元 / 兩條前綴共 {sum(len(x) for x in P._INJECT_PREFIX_LINES)} 字元 / artistry 區塊 {len(WIDGET)} 字元")
check("artistry 區塊裡有三個工具名", all(w in WIDGET for w in ("artistry", "image_journal", "canvas")))

print("1) off：與 v71 真實轉送的 system 逐位元組相同")
off, n_off = run()
check("removed = 0", n_off == 0, str(n_off))
check("system[0] 與真實轉送逐位元組相同", sys_text(off) == REAL_FWD["messages"][0]["content"],
      f"{len(sys_text(off))} vs {len(REAL_FWD['messages'][0]['content'])}")
check("訊息數與真實轉送相同", len(off["messages"]) == len(REAL_FWD["messages"]),
      f"{len(off['messages'])} vs {len(REAL_FWD['messages'])}")

print("2) on：artistry 區塊不見了，260 前綴還在")
on, n_on = run(strip_widget="on")
check("removed == artistry 區塊長度", n_on == len(WIDGET), f"{n_on} vs {len(WIDGET)}")
check("artistry 標記不在 system 裡了", MARKER not in sys_text(on))
for w in ("artistry", "image_journal", "canvas", "Widget Spawning"):
    check(f"「{w}」不在 system 裡了", w not in sys_text(on))
check("🔴 markdown/LaTeX 前綴還在（這是跟 strip_injections 的差別）", sys_text(on).startswith(PREFIX))
check("LaTeX 那一行也還在", P._INJECT_PREFIX_LINES[1] in sys_text(on))

print("3) on：身分接縫、人格、格式規則、示範區塊都沒少")
check("身分接縫仍收尾", sys_text(on).rstrip().endswith(SHIPPING["user_persona"]))
check("人格第 1 行還在", "你是User創造的" in sys_text(on))
check("格式規則區塊還在", P._FORMAT_BEGIN in sys_text(on) and P._FORMAT_END in sys_text(on))
check("原廠那句還在", "最後，做任何你想做的事！" in sys_text(on))
check("訊息數沒變（示範區塊沒動）", len(on["messages"]) == len(off["messages"]),
      f"{len(on['messages'])} vs {len(off['messages'])}")
# 441 減 1：artistry 區塊結尾有一個換行，而 _append_user_persona 是用
# 「c.rstrip() 之後接兩個換行再接身分接縫」接上去的，所以那個換行在 off 那邊
# 本來就會被 rstrip 吃掉 ⇒ 淨差 440，不是 441。
check("唯一的差就是那個區塊（441 − 尾端換行 1 ＝ 440）",
      len(sys_text(off)) - len(sys_text(on)) == len(WIDGET) - 1,
      f"{len(sys_text(off)) - len(sys_text(on))}，區塊 {len(WIDGET)} 字元、結尾 {WIDGET[-1]!r}")
check("on 的 system ＝ off 的 system 砍掉那一段再接回身分接縫",
      sys_text(on) == sys_text(off).replace(WIDGET.rstrip("\n"), "", 1) or
      sys_text(off).startswith(sys_text(on)[:sys_text(on).find(P._FORMAT_END)]))

print("4) no-op 與不重複砍")
clean = json.loads(json.dumps(BASE, ensure_ascii=False))
clean["messages"][0]["content"] = s0[:s0.find(MARKER)]
_, n_clean = run(base=clean, strip_widget="on")
check("沒有那個區塊時 removed = 0", n_clean == 0, str(n_clean))
both, n_both = run(strip_injections="on", strip_widget="on")
only_inj, n_inj = run(strip_injections="on")
check("兩個都開 ＝ 只開 strip_injections（不會砍兩次）", sys_text(both) == sys_text(only_inj))
check("兩個都開時 removed 不會重複計", n_both == n_inj, f"{n_both} vs {n_inj}")
check("strip_injections 單開會連前綴一起砍（所以 D 不能用它）", not sys_text(only_inj).startswith(PREFIX))

print()
if fails:
    print(f"🔴 {len(fails)} 項失敗：{fails}")
    sys.exit(1)
print("全部通過")
