# -*- coding: utf-8 -*-
# v57: 驗證「AIRI 會把這則回覆吞掉」的偵測會不會叫。不啟動服務、不碰 GPU。
#
# 校準的兩邊都要有（`00` §五：設計指標時先問「它可能被什麼繞過」）：
#   正例 = v56 真的掉掉的那兩則原文，逐字。
#   反例 = 同一輪同一批的正常回覆，逐字。反例會叫比正例不叫更糟——
#          偵測器天天狼來了，下一輪就沒有人看它了。
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["LLM_PROXY_TOOLS_BASELINE"] = str(TMP / "baseline.json")
os.environ["LLM_PROXY_MARKER_MEMORY"] = str(TMP / "marker.jsonl")
os.environ["LLM_PROXY_SWALLOW_LOG"] = str(TMP / "swallow.jsonl")
sys.path.insert(0, str(Path(__file__).parent))

import reasoning_proxy as P  # noqa: E402

# --- 正例：v56 真的掉掉的兩則（latency-v56/results-P-run1.json turn 1、run2 turn 12）
LOST_NO_GT = '<|ACT {"emotion":{"name":"curious","intensity":0.8},"motion":"lean in"},{"content":["欸。"]},{"content":["你要說什麼。"]}'
LOST_ANGLE = '<|ACT {"emotion": "think"}>, <motion>: lean_against_wall\n我剛才說過你作業寫不完，這就是你唯一的藉口。\n不然就說——User欠我一頓飯，你逃不過的。|'

# --- 反例：同一批的正常回覆
OK_PLAIN = '<|ACT {"emotion":{"name":"happy","intensity":0.8},"motion":"lean in"}|><|DELAY 1|>哇這麼厲害，今天居然不讓我陪你玩。'
OK_TWO_ACTS = '<|ACT {"emotion":"surprised"}|>明天？<|ACT {"emotion":"curious"}|>那先拿我什麼去換這個決定啊。'
OK_NO_MARKER = '對啊。你跟誰講都不好用User來當藉口。'
OK_CLOSED_TAG = '他說<b>不要</b>，然後就走了。'
OK_BR = ' 哇這麼厲害。 <br>\n 你累到哪裡去啦User？'

fails = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)


print("1) 正例：沒有任何 > 的殘缺標記 -> 要叫，而且要標成 v23 救不了")
f = P._swallow_watch(LOST_NO_GT, 1)
check("unterminated_marker=True", f["unterminated_marker"] is True)
check("recoverable_by_v23 不是 True", f["recoverable_by_v23"] is not True)
check("沒有誤報角度標籤", f["angle_tag"] is None)

print("2) 正例：有 > 但被 <motion> 吃掉的那則 -> 角度標籤要叫")
f = P._swallow_watch(LOST_ANGLE, 2)
check("angle_tag 抓到 motion", f["angle_tag"] == ["motion"])
check("同時標出標記沒關好", f["unterminated_marker"] is True)
check("而且標成 v23 救得了（真的救到了）", f["recoverable_by_v23"] is True)

print("3) 反例：正常回覆一律不准叫")
for name, text in [("單一 ACT", OK_PLAIN), ("兩個 ACT", OK_TWO_ACTS),
                   ("純文字", OK_NO_MARKER), ("成對的 <b>", OK_CLOSED_TAG)]:
    f = P._swallow_watch(text, 3)
    check(f"{name}：沒有 unterminated", f["unterminated_marker"] is False)
    check(f"{name}：沒有 angle_tag", f["angle_tag"] is None)

print("4) 🔴 <br> 不准叫——這是量出來的，不是假設的")
f = P._swallow_watch(OK_BR, 4)
check("<br> 不叫", f["angle_tag"] is None)
print("     把 v56 全部 80 則回覆重放過真實管線（標記解析器→categorizer→filterToSpeech，每三個字餵一次）：")
print("     五則帶 <br> 的都保住文字也真的念出來了；兩則帶未知標籤的（<motion>／<DELAY 1>）掉字。")
print("     所以 void 標籤排除是量出來的結論。⚠️ 代價：帶 <br> 的那則掉了 14 個字，這個偵測器看不到。")

print("5) 計數器與 /health")
h = P._swallow_health()
check("health 有 swallow_seen", h["swallow_seen"] == 7)
check("health 有 unterminated 計數", h["swallow_unterminated_marker"] == 2)
check("health 有 no_gt 計數", h["swallow_unterminated_no_gt"] == 1)
check("health 有 angle_tag 計數", h["swallow_angle_tag"] == 1)

print("6) 事件有落檔，而且含原文")
lines = Path(os.environ["LLM_PROXY_SWALLOW_LOG"]).read_text(encoding="utf-8").strip().split("\n")
check("落了 2 筆", len(lines) == 2)
rec = json.loads(lines[0])
check("含原文", rec["reply"] == LOST_NO_GT)
check("含 seq", rec["seq"] == 1)

print("7) 空回覆不算樣本")
before = P._swallow_health()["swallow_seen"]
P._swallow_watch("", 9)
check("空字串不計數", P._swallow_health()["swallow_seen"] == before)

print(f"\n{'全部通過' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
