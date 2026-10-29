# v33: 驗證工具清單偵測會不會叫。不啟動服務、不碰 GPU、不動真的基準線檔。
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["LLM_PROXY_TOOLS_BASELINE"] = str(TMP / "baseline.json")
os.environ["LLM_PROXY_MARKER_MEMORY"] = str(TMP / "marker.jsonl")
sys.path.insert(0, str(Path(__file__).parent))

import reasoning_proxy as P  # noqa: E402

REAL7 = ["builtIn_debugRandomNumber", "builtIn_emitSparkCommand", "builtIn_mcpCallTool",
         "builtIn_mcpListTools", "get_weather", "image_journal", "stage_widgets"]


def body(names):
    return {"messages": [], "tools": [{"type": "function", "function": {"name": n}} for n in names]}


fails = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


print("1) 沒看過任何請求時，不該報成變化（未知≠改變）")
h = P._tools_health()
check("tools_changed=False", h["tools_changed"] is False, f"count={h['tools_count']}")

print("2) 第一次看到工具 -> 自動寫基準線，不算變化")
P._note_tools(body(REAL7))
h = P._tools_health()
check("baseline 檔已建立", Path(os.environ["LLM_PROXY_TOOLS_BASELINE"]).exists())
check("tools_count=7", h["tools_count"] == 7)
check("tools_changed=False", h["tools_changed"] is False)

print("3) 同一組工具再送一次 -> 仍然不算變化")
P._note_tools(body(list(reversed(REAL7))))   # 順序不同也該視為相同
h = P._tools_health()
check("順序不影響判定", h["tools_changed"] is False)

print("4) 多一個工具（模擬有人接上 MCP server 後多出的工具）-> 要叫")
P._note_tools(body(REAL7 + ["mcp__filesystem__read"]))
h = P._tools_health()
check("tools_changed=True", h["tools_changed"] is True)
check("tools_added 指出新工具", h["tools_added"] == ["mcp__filesystem__read"], str(h["tools_added"]))

print("5) 少一個工具 -> 也要叫")
P._note_tools(body([n for n in REAL7 if n != "get_weather"]))
h = P._tools_health()
check("tools_changed=True", h["tools_changed"] is True)
check("tools_removed 指出少掉的", h["tools_removed"] == ["get_weather"], str(h["tools_removed"]))

print("6) 回到基準線 -> 不再叫")
P._note_tools(body(REAL7))
check("tools_changed=False", P._tools_health()["tools_changed"] is False)

print("7) strip_tools=on 的請求沒有 tools 欄 -> 不該把它當成『工具全被移除』")
P._note_tools({"messages": []})
check("空請求不影響狀態", P._tools_health()["tools_changed"] is False)

print("8) /health 真的帶出這些欄位")
ah = P._act_health()
check("health 含 tools_changed", "tools_changed" in ah)
check("health 含 tools_seen", "tools_seen" in ah)

print(f"\n{'全部通過' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
