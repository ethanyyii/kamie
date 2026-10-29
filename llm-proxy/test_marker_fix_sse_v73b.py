# -*- coding: utf-8 -*-
r"""v73b 任務三：**SSE 那一層**的測試。不啟動服務、不碰 GPU。

`test_marker_fix_v73b.py` 測的是文字層的狀態機；這一支測的是**風險最高的那一半**——
把它套到真正的位元組流上，而且位元組可能切在任何地方。

要證明三件事：
  1. 🔴 **健康的一則，位元組完全相同**（連框的邊界都一樣，不重新序列化）
  2. 壞的一則修好了，而且**切在哪裡都得到同一個結果**
  3. 不是 `data:` 的行、沒有 content 的框、解析不了的 JSON ——**一律原樣**

  ..\tts-test\omnivoice\.venv\Scripts\python.exe test_marker_fix_sse_v73b.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marker_fix import SseMarkerRewriter  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fails = []


def check(label, cond, detail=''):
    print('  %s  %s%s' % ('PASS' if cond else 'FAIL', label, ('  ' + detail) if detail else ''))
    if not cond:
        fails.append(label)


def sse(chunks, done=True):
    """做出跟 Ollama 一樣形狀的 SSE 位元組。"""
    out = b''
    for c in chunks:
        frame = {'id': 'chatcmpl-x', 'object': 'chat.completion.chunk',
                 'choices': [{'index': 0, 'delta': {'content': c}}]}
        out += b'data: ' + json.dumps(frame, ensure_ascii=False).encode('utf-8') + b'\n\n'
    if done:
        out += b'data: [DONE]\n\n'
    return out


def drive(raw, step):
    r = SseMarkerRewriter()
    out = b''.join(r.feed(raw[i:i + step]) for i in range(0, len(raw), step)) + r.flush()
    return out, r


def texts(raw):
    s = []
    for line in raw.split(b'\n'):
        line = line.strip()
        if not line.startswith(b'data:'):
            continue
        d = line[5:].strip()
        if d == b'[DONE]':
            continue
        try:
            o = json.loads(d)
        except Exception:
            continue
        c = ((o.get('choices') or [{}])[0].get('delta') or {}).get('content')
        if c:
            s.append(c)
    return ''.join(s)


def swallowed(t):
    i = t.rfind('<|')
    if i >= 0 and t.find('|>', i) < 0:
        return True
    opens = [m.group(1) for m in re.finditer(r'<([A-Za-z][A-Za-z0-9_.:-]*)(?:\s[^>]*)?>', t)]
    closes = {m.group(1) for m in re.finditer(r'</([A-Za-z][A-Za-z0-9_.:-]*)>', t)}
    VOID = {'br', 'hr', 'img', 'input', 'meta', 'link', 'wbr', 'area',
            'base', 'col', 'embed', 'param', 'source', 'track'}
    return any(x not in closes and x.lower() not in VOID for x in opens)


def spoken(t):
    out, i = [], 0
    while True:
        j = t.find('<|', i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        m = re.search(r'\|>|\|=|>\||>', t[j + 2:])
        if not m:
            break
        i = j + 2 + m.end()
    s = ''.join(out)
    s = re.sub(r'<[A-Za-z][A-Za-z0-9_.:-]*(?:\s[^>]*)?>', '', s)
    return re.sub(r'\s+', '', s)


# ---------------------------------------------------------------- 1) 健康的一則
print('1) 健康的一則：位元組完全相同（切在哪裡都一樣）')
HEALTHY_PARTS = ['<|ACT ', '{"emotion"', ':"happy"}', '|>', ' 我很好。', '你呢？']
HEALTHY = sse(HEALTHY_PARTS)
for step in (1, 4, 13, 64, 10 ** 6):
    out, r = drive(HEALTHY, step)
    check('切 %d 位元組：位元組完全相同' % step, out == HEALTHY,
          '' if out == HEALTHY else repr(out[:100]))
    check('  完全沒動過（events 空、沒退成直通）', not r.events and not r.off and r.error is None)

MULTI = sse(['<|ACT {"emotion":"surprised"}|>', '<|DELAY 1|>', '欸', '。', '<|ACT {"emotion":"curious"}|>', '真的假的'])
for step in (1, 9, 10 ** 6):
    out, r = drive(MULTI, step)
    check('多個好標記：切 %d 位元組仍然位元組相同' % step, out == MULTI,
          '' if out == MULTI else repr(out[:100]))

# ---------------------------------------------------------------- 2) 壞的一則
print()
print('2) 壞的一則（v73 任務 D 逐字）：修好，而且切在哪裡結果都一樣')
BROKEN = ('<|ACT {"emotion":"angry","intensity":0.8} , "motion":"cross_arm">\n'
          'User，你的泡麵都已經乾得像你的大腦一樣了。\n'
          '再問這種問題，就把這台破機子丟進河流裡洗乾淨！')
raw = sse([BROKEN[i:i + 7] for i in range(0, len(BROKEN), 7)])
check('修之前確實會被吞', swallowed(BROKEN))
results = []
for step in (1, 5, 23, 137, 10 ** 6):
    out, r = drive(raw, step)
    got = texts(out)
    results.append(got)
    check('切 %d 位元組：修好了（不會被吞）' % step, not swallowed(got), repr(got[:60]))
    check('  說出口的字一個都沒少', spoken(BROKEN) == spoken(got),
          '%r -> %r' % (spoken(BROKEN)[:28], spoken(got)[:28]))
    check('  [DONE] 還在', b'[DONE]' in out)
    check('  有記事件', len(r.events) > 0, str([e['kind'] for e in r.events]))
check('不同切法得到同一個結果', len(set(results)) == 1, str(len(set(results))) + ' 種結果')

BROKEN2 = '<|ACT {"emotion":"surprised"}|> 新同事？我叫User的。\n<|CALL ["chess.play"]|> <DELAY 1|> 來一局。'
raw2 = sse([BROKEN2[i:i + 11] for i in range(0, len(BROKEN2), 11)])
check('`<DELAY 1|>` 開頭少一個 | ：修之前會被吞', swallowed(BROKEN2))
out2, r2 = drive(raw2, 6)
got2 = texts(out2)
check('  修之後不會被吞', not swallowed(got2), repr(got2[:70]))
check('  補成 <|DELAY 1|>', '<|DELAY 1|>' in got2, repr(got2[-40:]))
check('  好的那個 <|CALL ["chess.play"]|> 沒被動', '<|CALL ["chess.play"]|>' in got2)

# ---------------------------------------------------------------- 3) 怪東西一律原樣
print()
print('3) 不是 data: 的行／沒有 content 的框／解析不了的 JSON：一律原樣')
ODD = (b': keep-alive comment\n\n'
       + b'data: not-json-at-all\n\n'
       + b'data: ' + json.dumps({'choices': [{'delta': {'role': 'assistant'}}]}).encode() + b'\n\n'
       + sse(['普通文字，沒有任何標記']))
for step in (1, 7, 10 ** 6):
    out, r = drive(ODD, step)
    check('切 %d 位元組：原樣' % step, out == ODD, '' if out == ODD else repr(out[:90]))
    check('  沒有動過', not r.events)

print()
print('4) 防線：退成直通')
r3 = SseMarkerRewriter()
r3.off = True
check('off 之後原樣送出、flush 空', r3.feed(b'anything at all') == b'anything at all' and r3.flush() == b'')

print()
if fails:
    print('🔴 %d 項失敗：' % len(fails))
    for x in fails[:12]:
        print('   -', x)
    sys.exit(1)
print('全部通過')
