# -*- coding: utf-8 -*-
r"""v73b 任務三：把「寫壞的控制標記」從沉默失敗變成有聲失敗，而且不要讓整則被吞。

## 為什麼需要這個

AIRI 的串流推理過濾器遇到**開了沒關的 `<|`**、或**沒有對應收尾的角度標籤 `<name>`**，
會**從那裡一路丟到結尾**。轉接層的 `_swallow_watch` 從 v58 起就在記這件事：
`llm-proxy\swallow-events.jsonl` 累積了 133 筆，**41 筆整則救不回來**。

🔴 **v73b 重新分類（照 `_swallow_watch` 自己的規則：元兇是最後那個沒有 `|>` 的 `<|`）**：

| 造成吞掉的東西 | 全部 133 筆 | 其中救不回來 41 筆 |
|---|---|---|
| 未閉合 `<\|CALL`            | 26 | **12** |
| 角度標籤 `<DELAY 1\|>`（開頭少一個 `\|`） | 11 | **11** |
| 未閉合 `<\|ACT`             | 66 | 6 |
| 其他角度標籤（`status:`／`description`／`tool_call`） | 6 | 6 |
| 未閉合 `<\|DELAY`           | 2 | 2 |
| 其餘（大寫走樣的 `ARTISTRY`／`GEN`／空的…） | 22 | 4 |

⚠️ **這修正了 v73 報告的說法**。v73 任務 D 那 120 則重放裡主因確實是 ACT 寫壞，
但那是 7 筆的樣本；**真實十一輪的紀錄裡，救不回來的主因是 CALL 與 `<DELAY 1|>`**，ACT 只有 6/41。

## 這支做什麼

一個**串流狀態機**。規則只有三條：

1. 🔴 **好的標記一個字都不動。** `<|NAME ...|>` 原樣通過。
   位元組層面：標記從框的開頭開始時，連 SSE 框都不重新序列化（**位元組完全相同**）；
   標記從一框的中間開始時（同一框裡又有文字又有標記開頭），那一段會重發成一框——
   **文字完全相同，只有框的邊界併了**。AIRI 是把 delta 接起來用的，所以語意一樣。
   （這句話原本寫成「一律位元組相同」，是測試把它打掉的。）
2. **能不靠猜就修回來的才修**：
   - `<DELAY 1|>`／`<CALL [...]|>`／`<ACT {...}|>`（**開頭少一個 `|`**）→ 補上 `|`。零猜測。
   - `<|ACT {…}|=` ／ `<|ACT {…}>|` ／ `<|ACT {…} , "motion":"x">`（**收尾寫壞**）→
     取**第一個括號配對完整、而且 `json.loads` 過得了**的 `{...}`，重發 `<|ACT {那個物件}|>`。
     🔴 **情緒不是猜的，是模型自己寫的合法 JSON 物件原封不動**；括號外面那些碎片**丟掉並記錄**。
   - `<|DELAY …>` 收尾寫壞 → 取第一個數字，重發 `<|DELAY n|>`。
3. **其餘一律剝掉並記錄**：整個壞掉的標記移除，**前後的文字一個字都不動**。
   不能解析的 `<|CALL …` 也走這條——AIRI 的工具在轉接層已經被 `strip_tools` 拿掉，
   `tool-events.jsonl` 576 筆**沒有一筆是真的工具呼叫**，所以剝掉一個壞掉的 CALL 不會少任何功能。

🔴 **為什麼「修」只修情緒、不把 `, "motion":"x"` 併回物件裡**（規劃端反駁點 3）：
把碎片併回去要假設模型「本來想放進括號內」——那是猜意圖。而情緒是**現成的合法 JSON**，
取它不需要任何假設。動作提示在卡片的格式說明裡本來就是**可選欄位**，丟掉只少一個動作，
丟錯情緒才會讓她的臉是錯的。⇒ **修情緒、丟碎片、把丟掉的東西記下來**，
等紀錄累積夠多再決定要不要更進一步。

## 延遲

狀態機**只有在標記裡面才會扣住位元組**，而且最多扣 `HOLD_CAP` 個字元。
標記本身不會被念出來（AIRI 送 TTS 之前就把標記剝掉了），所以扣住標記不會延後第一個音；
扣住的是「壞掉的標記後面那段本來會被整段吞掉的字」。
真實桌面版的對照見 `mainline-v73b\報告-v73b.md`。
"""
import json
import re

HOLD_CAP = 400          # 一個標記最多扣這麼多字元，超過就當它壞了（實測最長的合法標記 ~200）
MARKER_NAMES = ('ACT', 'DELAY', 'CALL')
# 開頭少一個 `|` 的形狀：`<DELAY 1|>`。只認已知的三個名字，普通的 `<b>` 不會被誤當標記。
_MISSING_PIPE = re.compile(r'<(' + '|'.join(MARKER_NAMES) + r')(\s[^<>]{0,%d}?)\|>' % HOLD_CAP)
# 不是標記、但會被 AIRI 的過濾器當成開標籤吃掉的東西：`<description>`、`<status:Generating "x">`
_VOID_TAGS = {'br', 'hr', 'img', 'input', 'meta', 'link', 'wbr', 'area',
              'base', 'col', 'embed', 'param', 'source', 'track'}
_ANGLE_OPEN = re.compile(r'<([A-Za-z][A-Za-z0-9_.:-]*)(?:\s[^>]*)?>')
_ANGLE_CLOSE = re.compile(r'</([A-Za-z][A-Za-z0-9_.:-]*)>')


def _first_json_object(s):
    """第一個括號配對完整、而且 json 解析得過的 {...}。回傳 (物件, 結束位置) 或 (None, -1)。"""
    start = s.find('{')
    while start >= 0:
        depth, instr, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                instr = not instr
                continue
            if instr:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1]), i + 1
                    except Exception:
                        break
        start = s.find('{', start + 1)
    return None, -1


def repair_marker(body):
    """`<|` 後面那一段（不含 `<|`，不含收尾）要怎麼處理。

    回傳 (修好的完整標記字串 或 None, 說明)。None ＝ 剝掉。
    """
    head = body.lstrip()
    name = re.split(r'[\s{\[]', head, 1)[0].upper() if head else ''
    rest = head[len(re.split(r'[\s{\[]', head, 1)[0]):] if head else ''
    if name == 'ACT':
        obj, end = _first_json_object(rest)
        if obj is None:
            return None, 'ACT 但找不到可解析的物件'
        dropped = rest[end:].strip()
        fixed = '<|ACT ' + json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '|>'
        return fixed, ('ACT 收尾寫壞，取模型自己寫的物件；丟掉括號外的 %r' % dropped[:40]
                       if dropped else 'ACT 收尾寫壞，物件完整')
    if name == 'DELAY':
        m = re.search(r'-?\d+(?:\.\d+)?', rest)
        if not m:
            return None, 'DELAY 但找不到數字'
        return '<|DELAY ' + m.group(0) + '|>', 'DELAY 收尾寫壞，取數字'
    if name == 'CALL':
        return None, 'CALL 壞掉：剝掉（轉接層沒有給她任何真工具，576 筆紀錄全是自己發明的）'
    return None, '不認得的標記名 %r：剝掉' % name[:16]


class MarkerFixer:
    """把一段一段來的文字接起來，只在標記內部扣住位元組。

    `feed(text)` 回傳可以放行的文字；`flush()` 回傳結尾殘留的部分。
    `events` 記下每一次動手（進紀錄檔與計數器）。
    """

    def __init__(self):
        self.pending = ''       # 只會是「已經看到 `<|` 但還沒收尾」的那一段
        self.events = []
        self.touched = False    # 這一則有沒有被動過（沒動過就讓呼叫端走原始位元組）

    # ---------------------------------------------------------------- 角度標籤
    def _strip_dangling_angles(self, text):
        """`<DELAY 1|>` 補 `|`；剩下沒有收尾的角度標籤剝掉。只動標籤本身，文字不動。"""
        def _fix_pipe(m):
            self.events.append({'kind': 'repaired', 'what': 'angle',
                                'before': m.group(0), 'after': '<|' + m.group(1) + m.group(2) + '|>',
                                'why': '開頭少一個 |'})
            self.touched = True
            return '<|' + m.group(1) + m.group(2) + '|>'
        text = _MISSING_PIPE.sub(_fix_pipe, text)
        closes = {m.group(1) for m in _ANGLE_CLOSE.finditer(text)}
        out, last = [], 0
        for m in _ANGLE_OPEN.finditer(text):
            tag = m.group(1)
            if tag in closes or tag.lower() in _VOID_TAGS:
                continue
            out.append(text[last:m.start()])
            self.events.append({'kind': 'stripped', 'what': 'angle',
                                'before': m.group(0), 'after': '',
                                'why': '沒有收尾的角度標籤，AIRI 會從這裡吃到結尾'})
            self.touched = True
            last = m.end()
        out.append(text[last:])
        return ''.join(out)

    # ---------------------------------------------------------------- 主迴圈
    def feed(self, text):
        if not text:
            return ''
        buf = self.pending + text
        self.pending = ''
        out = []
        i = 0
        while True:
            j = buf.find('<|', i)
            if j < 0:
                out.append(buf[i:])
                break
            out.append(buf[i:j])
            k = buf.find('|>', j + 2)
            if k >= 0:
                out.append(buf[j:k + 2])        # 🔴 好的標記：原樣，一個位元組都不動
                i = k + 2
                continue
            tail = buf[j:]
            if len(tail) <= HOLD_CAP:
                self.pending = tail             # 還沒收尾、還沒超過上限 ⇒ 扣住，等下一塊
                break
            out.append(self._repair(tail[:HOLD_CAP]))
            buf, i = tail[HOLD_CAP:], 0
        s = ''.join(out)
        # 🔴 把結尾那截「可能是標記或標籤的開頭」一起扣住。
        # 不扣的話，串流一個字一個字來的時候 `<` 和 `|` 會落在不同塊，
        # `find('<|')` 就永遠看不到它們，好標記會被當成普通文字放行、壞標記則躲過修復。
        p = s.rfind('<')
        if p >= 0 and '>' not in s[p:] and len(s) - p <= HOLD_CAP:
            self.pending = s[p:] + self.pending
            s = s[:p]
        return self._strip_dangling_angles(s)

    def flush(self):
        if not self.pending:
            return ''
        tail, self.pending = self.pending, ''
        if tail.startswith('<|'):
            return self._strip_dangling_angles(self._repair(tail))
        # 只是一截沒長成標記的殘骸（例如結尾剛好是一個 `<`）——交給角度標籤那一關
        return self._strip_dangling_angles(tail)

    @staticmethod
    def _payload_end(body):
        """完全沒有收尾的標記，**到哪裡為止算標記**。回 -1 表示整段都是標記。

        🔴 不用猜的規則：標記的語法本身**全部是 ASCII**（名字、`[]{}":,` 這些），
        只有**引號裡面的值**才可能是中文。所以**在字串外面**遇到一個非 ASCII 字元
        或換行，標記就已經結束了，後面是她要說的話。

        這條是測試逼出來的（v73b 端到端矩陣 E 格）：
        `<|CALL ["artistry", {"size": "m" 後面這句才是真正要驗的` ——
        原本整段被當成標記剝掉，**連後面那句話一起沒了**。
        ⚠️ 反過來也要成立：`…"prompt": "一個巨大的書本，上面寫著…` 那種**沒關的字串**
        裡的中文**不能**被當成結束（那一則本來就沒有後續台詞），所以要追引號的奇偶。
        """
        instr, esc = False, False
        for i, c in enumerate(body):
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                instr = not instr
                continue
            if instr:
                continue
            if c == '\n' or ord(c) > 127:
                return i
        return -1

    def _repair(self, tail):
        """tail 以 `<|` 開頭、而且沒有 `|>`。修得回來就修，修不回來就剝掉。"""
        body = tail[2:]
        # 收尾寫壞的三種：`>`、`>|`、`|=`。取到第一個這種記號為止當作標記本體。
        m = re.search(r'\|=|>\||>', body)
        if m:
            inner, after = body[:m.start()], body[m.end():]
        else:
            # 完全沒有收尾：用「字串外面的非 ASCII／換行」當邊界，
            # 後面那段是她要說的話，**不能跟標記一起剝掉**（見 _payload_end）。
            e = self._payload_end(body)
            inner, after = (body, '') if e < 0 else (body[:e], body[e:])
        fixed, why = repair_marker(inner)
        self.touched = True
        self.events.append({'kind': 'repaired' if fixed else 'stripped', 'what': 'marker',
                            'before': tail[:120], 'after': fixed or '',
                            'why': why + ('' if m or not after else '；標記沒有收尾，'
                                          '用「字串外的非 ASCII」當邊界，保住後面 %d 個字' % len(after))})
        return (fixed or '') + after


class SseMarkerRewriter:
    r"""把 MarkerFixer 套到 SSE 位元組流上。**這一層才是風險所在，所以要能單獨測。**

    Ollama 的 OpenAI 相容端點每個 token 一框：`data: {"choices":[{"delta":{...}}]}`。
    位元組可能切在任何地方，所以只處理**完整的一行**，半行留在 buf 裡。

    🔴 三條保證：
      1. **沒事的框原封不動**——連 JSON 都不重新序列化，位元組一個不差；
      2. **任何例外都只會退成直通**（`off=True` 之後這一則剩下的全部原樣送出）；
      3. 扣住的時候送出空的（不是壞掉的）框——呼叫端拿到 b"" 就好。
    """

    def __init__(self, fixer=None):
        self.fixer = fixer if fixer is not None else MarkerFixer()
        self.buf = b""
        self.off = False
        self.error = None
        # 🔴 扣住期間把**原始的行**留著。等狀態機確認這個標記其實是好的，
        # 就把原始位元組原樣吐回去——不然一個跨框的健康標記會被併成一框，
        # 那樣「健康的一則位元組完全相同」這句話就是假的。（測試逼出來的。）
        self._held = []          # [(原始的框, 是不是內容框)]
        # 扣住的字是不是「整框整框」對得上——上一框有沒有被切成「一半送出、一半扣住」。
        # 不是的話就不能把原始的框吐回去（會少字元）。
        self._in_sync = True

    @property
    def events(self):
        return self.fixer.events

    @staticmethod
    def _take_frame(buf):
        """切出一個完整的 SSE 框（含結尾的空行）。切不出來就回 (None, buf)。

        兩種結尾都認：`\\n\\n`（Ollama 用這個）與 `\\r\\n\\r\\n`。取比較早出現的那一個
        ——只認一種的話，另一種會讓整條流被緩衝到結尾才吐出來，那就不是串流了。"""
        i = buf.find(b"\n\n")
        j = buf.find(b"\r\n\r\n")
        if i < 0 and j < 0:
            return None, buf
        if j >= 0 and (i < 0 or j < i):
            return buf[:j + 4], buf[j + 4:]
        return buf[:i + 2], buf[i + 2:]

    def _reframe(self, obj, text):
        obj = json.loads(json.dumps(obj))
        (obj.get("choices") or [{}])[0].setdefault("delta", {})["content"] = text
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"

    def feed(self, chunk: bytes) -> bytes:
        if self.off:
            return chunk
        try:
            self.buf += chunk
            out = bytearray()
            while True:
                frame, self.buf = self._take_frame(self.buf)
                if frame is None:
                    break
                # 🔴 以**整框**為單位，不是以行為單位。用單一 `\n` 切的話，
                # 框與框之間那個空行會搶在被扣住的資料行前面送出去，順序就亂了
                # （SSE 框因此壞掉）。這是測試逼出來的。
                st = frame.strip()
                data = None
                for ln in st.split(b"\n"):
                    ln = ln.strip()
                    if ln.startswith(b"data:"):
                        data = ln[5:].strip()
                        break
                if data is None or data == b"[DONE]":
                    self._park(out, frame, False)
                    continue
                try:
                    obj = json.loads(data)
                    content = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")
                except Exception:
                    self._park(out, frame, False)
                    continue                 # 不是我們認得的框 => 不要碰
                if not content:
                    self._park(out, frame, False)
                    continue
                was = self.fixer.touched
                fixed = self.fixer.feed(content)
                if not fixed:
                    self._held.append((frame, True))     # 扣住中：原始的框先留著
                    continue
                if self.fixer.touched == was and not self.fixer.pending and self._in_sync:
                    # 沒動過任何東西、而且扣住的字**整框整框**對得上
                    # ⇒ 狀態機吐出來的就是它收進去的那些字，原始的框可以原封不動送出去。
                    # 🔴 `_in_sync` 這個條件是測試逼出來的：如果上一框是「一半送出去、
                    # 一半被扣住」（例如 `…的。\n<` 只送出前面那段），那被扣住的那個 `<`
                    # 就不在任何一個留著的框裡，這時候把原始的框吐回去會少掉一個字元。
                    out += b"".join(f for f, _ in self._held) + frame
                    self._held = []
                    self._in_sync = True
                    continue
                if self.fixer.touched == was:
                    # 只是被切開、沒有修過：照狀態機的輸出重發一框，文字完全一樣
                    self._held = []
                    out += self._reframe(obj, fixed)
                    self._in_sync = not self.fixer.pending
                    continue
                # 動過了：內容框由重發的那一框代表，**但其他框（[DONE]、註解、
                # 只有 role 的框）一個都不能丟**——丟了 [DONE] 就等於串流沒收尾。
                out += self._reframe(obj, fixed) + self._keepers()
                self._in_sync = not self.fixer.pending
            return bytes(out)
        except Exception as e:      # pragma: no cover - 防線，不該走到
            self.off = True
            self.error = repr(e)
            return chunk

    def _park(self, out, frame, is_content):
        """扣住期間來的框要排在後面，不能搶先送出去（會把 SSE 的順序弄亂）。"""
        if self._held:
            self._held.append((frame, is_content))
        else:
            out += frame

    def _keepers(self):
        """把扣住期間那些**不是內容**的框交回去（`[DONE]`、註解、只有 role 的框），
        然後清空。內容框不還——它們已經由重發的那一框代表了。"""
        keep = b"".join(f for f, is_c in self._held if not is_c)
        self._held = []          # [(原始的框, 是不是內容框)]
        return keep

    def flush(self) -> bytes:
        if self.off:
            return b""
        try:
            tail = self.fixer.flush()
            rest, self.buf = self.buf, b""
            if not tail:
                return self._keepers() + rest
            frame = (b"data: " + json.dumps({"choices": [{"delta": {"content": tail}}]},
                                            ensure_ascii=False).encode("utf-8") + b"\n\n")
            return frame + self._keepers() + rest
        except Exception as e:      # pragma: no cover
            self.error = repr(e)
            rest, self.buf = self.buf, b""
            return rest
