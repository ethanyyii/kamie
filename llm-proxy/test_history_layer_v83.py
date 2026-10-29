# -*- coding: utf-8 -*-
# v83 B：歷史那一層（history_profile）的離線測試。不啟動服務、不碰 GPU、不寫真的紀錄檔（全部環境變數指到暫存資料夾）。
# 輸入＝v82 重放台存下來的真實 AIRI 請求（mainline-v82\capture-e\payload-NNN.json：St／N 兩種殼、S1b 第 0／1／5／20／33 則、
# 40 上下文裡有前文與沒前文的），開關＝start_proxy.ps1 出貨值。
# 🔴 關鍵斷言：
#   1) off（hist=None 與 hist=("off", None, {}) 兩種）＝reasoning_proxy.BACKUP-v83.py 逐位元組相同（所有樣本 × 兩種示範送法 × 四種 format_rules）
#   2) 每一種做法：他這一句原樣、在最後；示範（St 前 18 則）與 system[0] 逐位元組不動；他前面的每一句都還在（原位或紀錄裡、逐字）；
#      record＝她的每一句原樣進紀錄、真的對話前文裡沒有 assistant；gist＝她的原句不在（大意用樁）；window2＝最近兩輪原位逐位元組同 off、
#      她更早的話不在紀錄裡；recent1_gist＝最近一輪原位、更早的她是大意
#   3) 大意拿不到：那一句原樣進紀錄，而且會叫（計數器、history-events.jsonl）；不准整段退回原文、不准變成沒有前文
#   4) 不認得的 profile 會叫；/config 的驗證；大意的兩次嘗試（空的、抄原句 ≥8 字 → 重來 → 失敗會叫）
import asyncio
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
             ("LLM_PROXY_HISTORY_GIST_MEMORY", "gist-memory.jsonl")):
    os.environ[k] = str(TMP / f)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import reasoning_proxy as P  # noqa: E402
P._settings["reply_cap"] = "off"   # v84: the reply ceiling adds max_tokens by design; its own test is test_reply_cap_v84.py

_spec = importlib.util.spec_from_file_location("proxy_v83b", HERE / "reasoning_proxy.BACKUP-v83.py")
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

ROOT = HERE.parent
CAP = ROOT / "mainline-v82" / "capture-e"
SAMPLES = {  # (arm shell, where) -> raw payload number（v82 arms-e-v82.json 查的）
    ("St", "S1b#0"): 141, ("St", "S1b#1"): 142, ("St", "S1b#5"): 146, ("St", "S1b#20"): 161, ("St", "S1b#33"): 174,
    ("St", "CTX#0"): 703, ("St", "CTX#22"): 725, ("St", "CTX#30"): 733,
    ("N", "S1b#0"): 73, ("N", "S1b#1"): 74, ("N", "S1b#5"): 78, ("N", "S1b#20"): 93, ("N", "S1b#33"): 106,
    ("N", "CTX#0"): 743, ("N", "CTX#22"): 765, ("N", "CTX#30"): 773,
}
RAW = {k: (CAP / f"payload-{n:03d}.json").read_bytes() for k, n in SAMPLES.items()}
SHIPPING = {"thinking": "off", "strip_tools": "on", "fewshot_messages": "on", "fewshot_mode": "messages+boundary",
            "sampling": "on", "sampling_temperature": 0.7, "sampling_top_p": 0.8, "model_override": "qwen3.5:4b",
            "strip_injections": "off", "strip_widget": "on", "fix_markers": "on", "restore_markers": "off",
            "user_persona": "現在跟你講話的使用者就是User本人。", "user_name_prefix": "User：", "user_frame": "",
            "user_persona_position": "tail", "fewshot_user_prefix": "off", "fewshot_prefix_skip_named": "on",
            "anchor_seed": "", "format_rules": "card"}
MODE = {"St": "messages+boundary", "N": "drop"}
fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


def setmode(mod, arm, fr="card"):
    mod._settings.update(SHIPPING)
    mod._settings["fewshot_mode"] = MODE[arm]
    mod._settings["format_rules"] = fr


def run(arm, key, prof=None, gists=None, fr="card"):
    setmode(P, arm, fr)
    info = {}
    hist = None
    if prof is not None:
        name, spec = P._history_resolve(prof)
        hist = (name, spec, gists or {})
    out = P._rewrite_all(RAW[(arm, key)], hist=hist, hist_info=info)
    return json.loads(out[0]), info, out


def airi_msgs(arm, key):
    return json.loads(RAW[(arm, key)])["messages"]


print("1) off ＝ v83 備份逐位元組（每個樣本 × 四種 format_rules × 兩種寫法）")
n_same = n_all = 0
for (arm, key) in SAMPLES:
    for fr in ("card", "short", "separate", "off"):
        setmode(OLD, arm, fr)
        old = OLD._rewrite_all(RAW[(arm, key)])
        setmode(P, arm, fr)
        new1 = P._rewrite_all(RAW[(arm, key)])
        new2 = P._rewrite_all(RAW[(arm, key)], hist=("off", None, {}), hist_info={})
        n_all += 1
        n_same += int(old == new1 == new2)
check("off byte-identical to BACKUP-v83 (%d cases)" % n_all, n_same == n_all, "%d/%d" % (n_same, n_all))

HEAD18 = {}
for arm in ("St", "N"):
    b, _, _ = run(arm, "S1b#20")
    HEAD18[arm] = b["messages"][:18] if arm == "St" else b["messages"][:1]


def real_part(arm, msgs):
    """Everything after the demos (St: after the boundary) / after system[0] (N)."""
    if arm == "St":
        assert msgs[17] == {"role": "system", "content": P._FEWSHOT_BOUNDARY}
        return msgs[18:]
    return msgs[1:]


def stub_gist(his, her):
    return "（樁）第%d句的大意" % (len(P._hist_norm(her)) % 997)


def gists_for(arm, key, keep_last=0):
    ms = airi_msgs(arm, key)
    older, _ = P._history_split(ms[1:-1], keep_last)
    return {P._gist_key(h, a): stub_gist(h, a) for h, a in P._history_pairs(older)}


for prof, keep, her_mode in (("record", 0, "verbatim"), ("gist", 0, "gist"), ("window2", 2, "drop"), ("recent1_gist", 1, "gist"), ("gist_inplace", 0, "gist")):
    print(f"2) {prof}")
    for (arm, key) in SAMPLES:
        base, _, _ = run(arm, key)
        g = gists_for(arm, key, keep) if her_mode == "gist" else {}
        b, info, _ = run(arm, key, prof, g)
        M, B = b["messages"], base["messages"]
        tag = f"{arm} {key}"
        ams = airi_msgs(arm, key)
        hist = ams[1:-1]
        older, recent = P._history_split(hist, keep)
        # 他這一句：原樣、在最後
        check(f"{tag}: newest user turn identical and last", M[-1] == B[-1] and M[-1]["role"] == "user")
        # 示範與 system[0]
        head_n = 18 if arm == "St" else 1
        check(f"{tag}: demos / system[0] untouched", M[:head_n] == B[:head_n] and M[:head_n] == HEAD18[arm])
        check(f"{tag}: exactly the system messages it had", [m["role"] for m in M].count("system") == [m["role"] for m in B].count("system"))
        rp, rb = real_part(arm, M), real_part(arm, B)
        if not older:
            check(f"{tag}: nothing older -> untouched", M == B and not info.get("applied"))
            continue
        if prof == "gist_inplace":
            # 沒有紀錄訊息：則數、角色順序跟 off 一樣；他的句子逐位元組原位；她的每一句原位換成大意（沒有標籤）
            check(f"{tag}: same length and roles as off", len(rp) == len(rb) and [m["role"] for m in rp] == [m["role"] for m in rb])
            check(f"{tag}: his lines byte-identical in place", all(a == c for a, c in zip(rp, rb) if c["role"] == "user"))
            her_old = [a for _, a in P._history_pairs(older)]
            exp = [stub_gist(h, a) for h, a in P._history_pairs(older)]
            got = [m["content"] for m in rp[:-1] if m["role"] == "assistant"]
            check(f"{tag}: her earlier replies replaced in place by gists ({len(exp)})", got == exp and info["fallbacks"] == 0 and info.get("record") is None)
            check(f"{tag}: no record head anywhere", all("前面講過的話" not in m["content"] for m in rp))
            continue
        rec = rp[0]
        check(f"{tag}: one record message in the user role right after the demos", rec["role"] == "user" and rec["content"] == info["record"]
              and rec["content"].startswith("（User跟你這一場前面講過的話，照順序：）") and "（以上是前面講過的" not in rec["content"])
        # 他前面講過的每一句都還在（逐字）：舊的在紀錄裡、近的在原位
        his_all = [m["content"] for m in rb[:-1] if m["role"] == "user"]
        joined = "\n".join(m["content"] for m in rp)
        check(f"{tag}: every earlier line of his is still there verbatim ({len(his_all)})", all(h.strip() in joined for h in his_all))
        # 最近幾輪原位逐位元組同 off
        n_recent = len(recent)
        check(f"{tag}: recent {keep} exchange(s) native and identical to off", rp[1:1 + n_recent] == rb[len(rb) - 1 - n_recent:len(rb) - 1])
        check(f"{tag}: nothing else besides record + recent + newest", len(rp) == 1 + n_recent + 1)
        her_old = [a for _, a in P._history_pairs(older)]
        if her_mode == "verbatim":
            check(f"{tag}: her earlier lines verbatim in the record ({len(her_old)})", all(("你：" + P._her_plain(a)) in rec["content"] for a in her_old))
            check(f"{tag}: no assistant turn left from the real history", all(m["role"] != "assistant" for m in rp))
        elif her_mode == "gist":
            check(f"{tag}: her earlier lines NOT in the record", not any(P._her_plain(a) and P._her_plain(a) in rec["content"] for a in her_old))
            check(f"{tag}: one gist line per earlier reply", rec["content"].count("你（大意）：") == len(her_old) and info["fallbacks"] == 0)
        else:
            check(f"{tag}: her earlier lines dropped (not in the record)", not any(P._her_plain(a) and ("你：" + P._her_plain(a)) in rec["content"] for a in her_old)
                  and "你：" not in rec["content"])
        check(f"{tag}: info says what it did", info["n_older"] == len(older) and info["n_recent"] == n_recent and info["profile"] == prof)

print("3) 大意拿不到 → 那一句原樣進紀錄、會叫（不整段退回、不變成沒有前文）")
before = dict(P._history_counters)
ev_before = (TMP / "history-events.jsonl").read_text(encoding="utf-8").count("\n") if (TMP / "history-events.jsonl").exists() else 0
g = gists_for("N", "S1b#5")
drop_one = sorted(g)[0]
g2 = {k: v for k, v in g.items() if k != drop_one}
b, info, _ = run("N", "S1b#5", "gist", g2)
rec = b["messages"][1]["content"]
check("one fallback counted", info["fallbacks"] == 1 and P._history_counters["history_gist_fallback_lines"] == before["history_gist_fallback_lines"] + 1)
check("history-events.jsonl got the event", (TMP / "history-events.jsonl").read_text(encoding="utf-8").count("\n") == ev_before + 1)
check("the other lines are still gists", rec.count("你（大意）：") == len(g) - 1 and rec.count("你：") == 1)
check("record still there (not silently no history)", b["messages"][1]["role"] == "user" and len(b["messages"]) == 3)

print("4) 不認得的 profile、/config 驗證")
bp = P._history_counters["history_bad_profile"]
nm, sp = P._history_resolve("nope")
check("unknown profile -> None and loud", sp is None and P._history_counters["history_bad_profile"] == bp + 1)
for patch, ok in (({"history_profile": "gist"}, True), ({"history_profile": "zzz"}, False), ({"history_keep_last": -1}, False),
                  ({"history_her_older": "summary"}, False), ({"history_record_role": "assistant"}, False), ({"history_gist_timeout_s": 0}, False)):
    r = P.set_config(dict(patch))
    is_err = hasattr(r, "status_code") and getattr(r, "status_code", 200) == 400
    check(f"set_config {patch} -> {'ok' if ok else '400'}", (not is_err) if ok else is_err)
P._settings["history_profile"] = "off"
nm, sp = P._history_resolve("custom")
check("custom reads the history_* keys", sp == {"keep_last": P._settings["history_keep_last"], "her_older": P._settings["history_her_older"],
                                                 "record_role": P._settings["history_record_role"]})


print("5) 大意的兩次嘗試（假的 Ollama）")


class FakeResp:
    def __init__(self, content):
        self._c = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._c}}


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.bodies = []

    async def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return FakeResp(a)


HER = "先把你那堆爛報告的邏輯整理清楚，不然我怎麼知道該幫你換個能裝雜音的大腦還是直接把你腦袋當垃圾桶丟掉。"
HIS = "[2026-09-25 13:00] User：那你幫我升級大腦啊"
P._client = FakeClient(["", "要他先把報告理順，不然只能把他的頭丟掉"])
r = asyncio.run(P._gist_call(HIS, HER))
check("empty -> retried -> ok", r["gist"] == "要他先把報告理順，不然只能把他的頭丟掉" and r["attempts"] == 2, str(r))
b0 = P._client.bodies[0]
check("native /api/chat body: think false, qwen3.5:4b, no stream", b0["think"] is False and b0["stream"] is False and b0["model"] == "qwen3.5:4b")
check("prompt carries his line without timestamp/prefix and her line without markers", "User說：那你幫我升級大腦啊" in b0["messages"][1]["content"] and "Kamie 回：先把你那堆" in b0["messages"][1]["content"])
P._client = FakeClient(["先把你那堆爛報告的邏輯整理清楚", "先把你那堆爛報告的邏輯整理清楚喔"])
r = asyncio.run(P._gist_call(HIS, HER))
check("copies >= 8 chars twice -> None with reasons", r["gist"] is None and len(r["why"]) == 2 and "shares" in r["why"][0], str(r["why"]))
P._client = FakeClient([RuntimeError("boom"), "嗆他報告很爛、要他先整理"])
r = asyncio.run(P._gist_call(HIS, HER))
check("exception -> retried -> ok", r["gist"] == "嗆他報告很爛、要他先整理", str(r))
# _gist_make：失敗會叫、成功進記憶；_history_prepare 等到同一個 task
fb = P._history_counters["history_gist_failed"]
P._client = FakeClient(["", ""])
res = asyncio.run(P._gist_make("k-fail", HIS, HER))
check("failed gist is loud (counter + event) and not remembered", res["gist"] is None and P._history_counters["history_gist_failed"] == fb + 1 and "k-fail" not in P._gist_memory)


async def _prep_case():
    raw = RAW[("N", "S1b#5")]
    ms = json.loads(raw)["messages"]
    older, _ = P._history_split(ms[1:-1], 0)
    pairs = P._history_pairs(older)
    P._gist_memory.clear()
    P._gist_tasks.clear()
    answers = ["大意%d" % i for i in range(len(pairs))]
    P._client = FakeClient(answers)
    # 模擬「她講完之後」已經排了第一個
    k0 = P._gist_key(*pairs[0])
    P._gist_for(k0, *pairs[0])
    out = await P._history_prepare(raw, {"keep_last": 0, "her_older": "gist", "record_role": "user"})
    return out, pairs, P._client


out, pairs, cl = asyncio.run(_prep_case())
check("prepare returns one gist per earlier reply, one call each", len(out["gists"]) == len(pairs) and out["missing"] == 0 and len(cl.bodies) == len(pairs),
      "%d gists / %d pairs / %d calls" % (len(out["gists"]), len(pairs), len(cl.bodies)))
check("gist memory file written", (TMP / "gist-memory.jsonl").exists() and (TMP / "gist-memory.jsonl").read_text(encoding="utf-8").count("\n") >= len(pairs))

print("\n" + ("ALL PASS" if not fails else "FAILED: %d -> %s" % (len(fails), fails[:8])))
sys.exit(1 if fails else 0)
