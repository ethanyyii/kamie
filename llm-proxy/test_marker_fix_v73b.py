# -*- coding: utf-8 -*-
r"""v73b 任務三的測試。不啟動服務、不碰 GPU、不動任何真的紀錄檔。

🔴 照 v73 執行端提的規則（`00` v1.6 §五 的延伸）：**已知壞樣本與已知好樣本都寫進來，不過就 exit 1。**
壞樣本**全部是真的**——從 `mainline-v73\widget-off-v73.json`（v73 任務 D 的 120 則重放）
與 `llm-proxy\swallow-events.jsonl`（v58–v70 真實使用）裡逐字抄出來的。

  ..\tts-test\omnivoice\.venv\Scripts\python.exe test_marker_fix_v73b.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marker_fix import MarkerFixer, HOLD_CAP  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fails = []


def check(label, cond, detail=''):
    print('  %s  %s%s' % ('PASS' if cond else 'FAIL', label, ('  ' + detail) if detail else ''))
    if not cond:
        fails.append(label)


def run(text, chunk=None):
    """整段餵，或切成 chunk 大小一塊一塊餵（驗跨框不會出錯）。"""
    f = MarkerFixer()
    if chunk is None:
        out = f.feed(text) + f.flush()
    else:
        out = ''.join(f.feed(text[i:i + chunk]) for i in range(0, len(text), chunk)) + f.flush()
    return out, f


# ---------------------------------------------------------------- 1) 好的一個都不准動
GOOD = [
    '<|ACT {"emotion":"neutral"}|> 我很好。你呢？',
    '<|ACT {"emotion":"surprised"}|><|DELAY 1|>（台詞）<|ACT {"emotion":"curious"}|><|DELAY 1|>（情緒改變後的台詞）',
    '<|ACT {"emotion":{"name":"surprised","intensity":1},"motion":"shrug"}|> 哇。那你完蛋了。',
    '<|ACT {"emotion":"happy"}|> 在看你穿這件白襯衫。真像張照片。你好像想拍什麼？',
    '<|ACT {"emotion":"angry", "intensity":0.8,"motion":"cross arms"}|> 閉嘴，User。',
    '<|CALL ["chess.play", {"key":"undo"}]|>',
    '<|ACT {"emotion":"sad"}|><|DELAY 1|>你瘋了嗎？<|CALL ["coffee.spill_clothes"]|>',
    '沒有任何標記的一句話。',
    '',
    '她說「這個 < 符號」跟 5 < 7 都不該被碰。',
    '<br>換行標籤是 void，不該被剝掉。',
]
print('1) 好的標記與普通文字：一個位元組都不准動')
for g in GOOD:
    out, f = run(g)
    check('原樣通過：%r' % (g[:40] or '(空字串)'), out == g and not f.touched,
          '' if out == g else '得到 %r' % out[:60])
for g in GOOD[:6]:
    for ck in (1, 3, 7, 40):
        out, f = run(g, ck)
        check('切成 %d 字一塊也原樣：%r' % (ck, g[:24]), out == g,
              '' if out == g else '得到 %r' % out[:60])

# ---------------------------------------------------------------- 2) v73 D 的真實壞樣本
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
wid = os.path.join(ROOT, 'mainline-v73', 'widget-off-v73.json')
REAL_BAD = []
if os.path.exists(wid):
    for r in json.load(open(wid, encoding='utf-8'))['rows']:
        if r['would_be_logged']:
            REAL_BAD.append(r['reply'])
print()
print('2) v73 任務 D 的真實壞樣本（%d 則，逐字）' % len(REAL_BAD))


def swallowed(t):
    """照轉接層 _swallow_watch 的規則：這一則會不會被 AIRI 吃掉。"""
    i = t.rfind('<|')
    if i >= 0 and t.find('|>', i) < 0:
        return True
    opens = [m.group(1) for m in re.finditer(r'<([A-Za-z][A-Za-z0-9_.:-]*)(?:\s[^>]*)?>', t)]
    closes = {m.group(1) for m in re.finditer(r'</([A-Za-z][A-Za-z0-9_.:-]*)>', t)}
    VOID = {'br', 'hr', 'img', 'input', 'meta', 'link', 'wbr', 'area',
            'base', 'col', 'embed', 'param', 'source', 'track'}
    return any(x not in closes and x.lower() not in VOID for x in opens)


def spoken(t):
    """模型**實際寫出來的字**（不是 AIRI 會顯示的字）。修完之後這些字一個都不准少。

    ⚠️ 不可以拿「AIRI 會顯示什麼」當比較基準：壞掉那一則本來就顯示不出來，
    那樣比等於拿 '' 跟修好的結果比，永遠過不了。

    這裡從每個 `<|` 剝到**第一個收尾記號**——好的 `|>`，或寫壞的 `|=`／`>|`／`>`；
    一個都找不到才剝到結尾。收尾記號的清單跟修復器一致（不然 oracle 會比實作還笨），
    但掃法是單純的線性掃描，跟那個串流狀態機是兩套不同的寫法。
    要驗的性質是同一個：**她說出口的字，一個都不准掉。**
    """
    out, i = [], 0
    while True:
        j = t.find('<|', i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        m = re.search(r'\|>|\|=|>\||>', t[j + 2:])
        if not m:
            break                      # 完全沒收尾 => 剝到結尾
        i = j + 2 + m.end()
    s = ''.join(out)
    s = re.sub(r'<[A-Za-z][A-Za-z0-9_.:-]*(?:\s[^>]*)?>', '', s)   # 角度標籤
    return re.sub(r'\s+', '', s)


for t in REAL_BAD:
    check('修之前確實會被吞：%r' % t[:34], swallowed(t))
    out, f = run(t)
    check('  修之後不會被吞', not swallowed(out), repr(out[:70]))
    check('  說得出來的字一個都沒少', spoken(t) == spoken(out),
          '' if spoken(t) == spoken(out) else '%r -> %r' % (spoken(t)[:40], spoken(out)[:40]))
    check('  有記事件', len(f.events) > 0, str([e['kind'] for e in f.events]))
    for ck in (1, 5, 17):
        o2, _ = run(t, ck)
        check('  切 %d 字一塊結果一樣' % ck, o2 == out)

# ---------------------------------------------------------------- 3) 真實紀錄裡救不回來的那幾種
print()
print('3) swallow-events.jsonl 裡「救不回來」的形狀（逐字）')
HARD = [
    ('<|ACT {"emotion": "surprised"}|> <DELAY 1|> <CALL ["google.search", {"query": "台灣的車站數量是多少", "source_filter": ["wikipedia", "stat.gov.tw"]}>',
     '角度標籤 DELAY + 未閉合 CALL'),
    ('<|ACT {"emotion":"surprised"}|> 新同事？我叫User的。\n<|CALL ["chess.play"]|> <DELAY 1|> 來一局，贏他就給你看。',
     '<DELAY 1|> 開頭少一個 |'),
    ('<|ACT {"emotion":"neutral"}|> 沒名字。只是個坐在角落的陌生人。\n<|CALL ["image_journal"]|> <description>: 描述「User」，一個正在看手機的高中生。',
     '<description> 沒有收尾'),
    ('<|ACT {"emotion":"happy"}|= 四點半了。快去吃東西啦！', '收尾寫成 |='),
    ('<|ACT {"emotion":"neutral"}>|<DELAY 1|>你拿著那個什麼？', '收尾寫成 >|'),
    ('<|ACT {"emotion":"awkward"}|> 當然不會。\n\n<|CALL ["artistry", {"size": "m", "id": "my-art-02", "prompt": "一個巨大的書本',
     '未閉合的 CALL 拖到結尾'),
    ('<|', '整則只剩一個 <|'),
]
for t, why in HARD:
    check('修之前會被吞（%s）' % why, swallowed(t))
    out, f = run(t)
    check('  修之後不會被吞', not swallowed(out), repr(out[:70]))
    check('  說得出來的字一個都沒少', spoken(t) == spoken(out),
          '%r -> %r' % (spoken(t)[:36], spoken(out)[:36]))

# ---------------------------------------------------------------- 4) 修的內容是不是我說的那樣
print()
print('4) 「修」只取模型自己寫的合法物件，不猜情緒、不把碎片併回去')
t = '<|ACT {"emotion":"angry","intensity":0.8} , "motion":"cross_arm">User你瘋了吧？'
out, f = run(t)
check('情緒與強度原封不動', '"emotion":"angry"' in out and '"intensity":0.8' in out, out[:60])
check('括號外的碎片丟掉了', 'cross_arm' not in out, out[:60])
check('收尾補成 |>', out.startswith('<|ACT ') and '|>' in out, out[:40])
check('丟掉的東西有記下來', any('cross_arm' in (e['why'] or '') for e in f.events),
      str([e['why'][:40] for e in f.events]))
check('後面的字一個都沒少', 'User你瘋了吧？' in out)

t2 = '<|ACT {"emotion":"happy"} , {"emotion":"angry"}>嗨'
out2, _ = run(t2)
check('兩個物件時取第一個（記錄下來，不合併）', '"happy"' in out2 and '"angry"' not in out2, out2[:50])

t3 = '<|ACT 這裡根本沒有物件>後面的字'
out3, f3 = run(t3)
check('ACT 但沒有可解析的物件 ⇒ 剝掉', '<|ACT' not in out3 and '後面的字' in out3, repr(out3))
check('  記成 stripped', any(e['kind'] == 'stripped' for e in f3.events))

t4 = '<|DELAY 2>接下來'
out4, _ = run(t4)
check('DELAY 收尾寫壞 ⇒ 取數字補回來', out4.startswith('<|DELAY 2|>') and '接下來' in out4, repr(out4))

t5 = '<|CALL ["artistry", {"size":"m"}>剩下的話'
out5, f5 = run(t5)
check('壞掉的 CALL ⇒ 剝掉，文字留著', '<|CALL' not in out5 and '剩下的話' in out5, repr(out5))
check('  理由寫得出來', any('CALL' in e['why'] for e in f5.events))

# ---------------------------------------------------------------- 5) 邊界
print()
print('5) 邊界')
long_marker = '<|CALL ["x", {"prompt": "' + ('字' * (HOLD_CAP + 60)) + '後面'   # 故意不收尾
out6, f6 = run(long_marker)
check('超過上限的未閉合標記不會無限扣住', '後面' in out6 and len(out6) < len(long_marker), '%d -> %d' % (len(long_marker), len(out6)))
ok_long = '<|CALL ["x", {"prompt": "' + ('字' * 120) + '"}]|>後面'
out7, f7 = run(ok_long)
check('上限以內的合法長標記原樣通過', out7 == ok_long and not f7.touched)
out8, f8 = run('純文字，一個標記都沒有，餵一百次')
check('沒有標記時 touched=False（呼叫端可以走原始位元組）', not f8.touched and out8 == '純文字，一個標記都沒有，餵一百次')

# ---------------------------------------------------------------- 6) 完全沒收尾的標記，邊界畫在哪
print()
print("6) 完全沒收尾的標記：後面那段台詞不可以跟標記一起剝掉（端到端矩陣 E 格逼出來的）")
E1 = '<|ACT {"emotion":"happy"}|> 前面這句留得住嗎。 <|CALL ["artistry", {"size": "m" 後面這句才是真正要驗的。'
out, f = run(E1)
check("後面那句台詞留住了", "後面這句才是真正要驗的。" in out, repr(out))
check("  壞掉的 CALL 剝掉了", "<|CALL" not in out)
check("  前面那句也還在", "前面這句留得住嗎。" in out)
check("  理由寫出保住幾個字", any("保住後面" in e["why"] for e in f.events), str([e["why"][-24:] for e in f.events]))
for ck in (1, 6, 29):
    o2, _ = run(E1, ck)
    check("  切 %d 字一塊結果一樣" % ck, o2 == out)

# 反方向：中文在**沒關的字串裡面**，那是標記的一部分，不可以當成台詞留下來
E2 = '<|ACT {"emotion":"awkward"}|> 當然不會。我連你的鞋墊都要記在心裡。\n\n<|CALL ["artistry", {"size": "m", "id": "my-art-02", "prompt": "一個巨大的書本，上面寫著『你什麼都忘不掉』'
out2, f2 = run(E2)
check("字串裡的中文沒有被當成台詞留下", "一個巨大的書本" not in out2, repr(out2[-40:]))
check("  前面那句台詞還在", "當然不會。我連你的鞋墊都要記在心裡。" in out2)
check("  兩者都不會被吞", not swallowed(out) and not swallowed(out2))

print()
if fails:
    print('🔴 %d 項失敗：' % len(fails))
    for x in fails[:14]:
        print('   -', x)
    sys.exit(1)
print('全部通過（%d 個好樣本、%d 則 v73 D 真實壞樣本、%d 則紀錄裡救不回來的形狀）'
      % (len(GOOD), len(REAL_BAD), len(HARD)))
