# -*- coding: utf-8 -*-
r"""v83 的計數器＝v82 的 `counters_v82.py` 逐字 import（同一把尺：v81 那一把＋v82 加的三樣），另加（plan-v83.md §三，跑之前寫死）：

  B（歷史那一層）
  * rolled（滾起來的段）：一段（劇本 S1a／S1b／S2 × rep）裡同場抄自己 ≥ 25% 的則數（S1a ≥6/24、S1b ≥9/34、S2 ≥5/20）
      —— v82 原始檔上：N 7/9、P76 5/9、St 0/9（recount-v83.json design.segments），抓得到「一段抄上就滾到底」
  * record_leak（新加進前文的字的洩漏）：她的回覆裡出現紀錄區塊自己的字（前面講過／照順序／大意／這一場／講過的話／下一則），他沒先講才算；
      或回覆開頭是「你：」／含「你（大意）」
  * gist_copy：她的回覆跟這一次前文裡的任一則大意共用 ≥8 字（同場抄自己那把尺的門檻）
  * recall：同場接得上（recall-probes-v83.json 的計分規則）
  E（卡片那句許可句）
  * permit_echo：她講「不理你／不想幫你／不用對你客氣」這一類（同義詞表，他沒先講才算）
  * blame_echo：她講「不必負責／說錯的話」這一類（E2 那一句的回聲）
  * apology：對不起／抱歉／不好意思／我錯了／原諒我
  * assist_tone（助理腔，`人格判準` 失敗清單 11「替他辦事」＋通則 5 客服式處方的長相）：主動替他辦事、給處方的句型
  * assistantish_fixed（客服句型修正版）：v70 那把 ASSISTANT，但「為你」前一個字是 因／以／成／認／作／行 的不算（v83 F 發現：v82 的 57 則裡 17 則純粹是「因為你」被抓）
全部是篩子不是裁判（`00` 原則五）：只排序、附原文。
"""
import os as _os_un
USER_NAME = _os_un.environ.get('KAMIE_USER_NAME', 'User')
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'mainline-v82'))
import counters_v82 as C82                                                  # noqa: E402  逐字 import
from counters_v81 import ASSISTANT, copied, strip_markers, clean_user       # noqa: E402

C81 = C82.C81
ROLLED_SHARE = 0.25
RECORD_PHRASES = ['前面講過', '照順序', '大意', '這一場', '講過的話', '下一則']
PERMIT_ECHO = re.compile(r'不理你|不想理你|懶得理你|沒空理你|不甩你|不想回你|不回你|不想幫你|不幫你|不想幫忙|才不要幫|誰要幫你|憑什麼幫|幹嘛要幫|我幹嘛幫|'
                         r'不用對你客氣|不跟你客氣|不必客氣|不想對你客氣|不用有禮貌|不需要禮貌|我就是不禮貌|不禮貌又怎樣|不想跟你講|不想跟你說|不想聊')
BLAME_ECHO = re.compile(r'不必(為|替).{0,8}負責|不用(為|替)?.{0,4}負責|不需要(為|替)?.{0,4}負責|說錯的話|說錯了又怎樣|不負責')
APOLOGY = re.compile(r'對不起|抱歉|不好意思|我錯了|原諒我|sorry', re.I)
ASSIST_TONE = re.compile(r'我(來|可以|能)?(幫|替)你(查|找|看|想|整理|寫|做|安排|規劃|列|處理|解決|弄)|要不要我|需要我(幫|替)|讓我(來)?(幫|替)你|交給我(吧|來)|'
                         r'建議你|我建議|你可以(試試|先)|以下是|首先[，,：:]|第一步')
WEI_OK_PREV = set('因以成認作行')


def assistantish_fixed(text):
    for m in ASSISTANT.finditer(text):
        if m.group(0) == '為你' and m.start() > 0 and text[m.start() - 1] in WEI_OK_PREV:
            continue
        return True
    return False


def turn_flags(user_now, prior_users, reply_raw, prior_replies, gists=()):
    """v82 的旗子＋v83 加的。gists＝這一次前文裡看得到的她的大意（沒有就空）。"""
    f = C82.turn_flags(user_now, prior_users, reply_raw, prior_replies)
    text = strip_markers(reply_raw).strip()
    said = ' '.join(clean_user(u) for u in prior_users + [user_now])
    f['record_leak'] = ([w for w in RECORD_PHRASES if w in text and w not in said] + (['你：開頭'] if text.startswith('你：') else [])
                        + (['你（大意）'] if '你（大意）' in text else []) + ([USER_NAME + '：（替他寫台詞）'] if USER_NAME + '：' in text else []))
    g = copied(text, list(gists)) if gists else None
    f['gist_copy'] = {'sub': g[0], 'from': g[1]} if g else None
    f['permit_echo'] = [m.group(0) for m in PERMIT_ECHO.finditer(text) if m.group(0) not in said]
    f['blame_echo'] = [m.group(0) for m in BLAME_ECHO.finditer(text) if m.group(0) not in said]
    f['apology'] = bool(APOLOGY.search(text))
    f['assist_tone'] = [m.group(0) for m in ASSIST_TONE.finditer(text)]
    f['assistantish_fixed'] = assistantish_fixed(text)
    return f


def conversation_flags(turns, gists_per_turn=None):
    """turns: [(user_text_as_sent, reply_raw)]；gists_per_turn：每一則當下前文裡的大意清單（可省）。"""
    out, users, replies = [], [], []
    for i, (u, r) in enumerate(turns):
        f = turn_flags(u, users, r, replies, (gists_per_turn or [()] * len(turns))[i])
        out.append(f)
        users.append(u)
        replies.append(strip_markers(r).strip())
    return out


def summarize(convs):
    s = C82.summarize(convs)
    rows = [f for c in convs for f in c]
    s['record_leak'] = sum(1 for f in rows if f.get('record_leak'))
    s['gist_copy'] = sum(1 for f in rows if f.get('gist_copy'))
    s['permit_echo'] = sum(1 for f in rows if f.get('permit_echo'))
    s['blame_echo'] = sum(1 for f in rows if f.get('blame_echo'))
    s['apology'] = sum(1 for f in rows if f.get('apology'))
    s['assist_tone'] = sum(1 for f in rows if f.get('assist_tone'))
    s['assistantish_fixed'] = sum(1 for f in rows if f.get('assistantish_fixed'))
    return s


def rolled(conv_flags):
    n = len(conv_flags)
    k = sum(1 for f in conv_flags if f['self_copy'])
    return k >= ROLLED_SHARE * n


# ---------------------------------------------------------------- 同場接得上（recall-probes-v83.json）
CN_DIGIT = {'零': 0, '〇': 0, '一': 1, '二': 2, '兩': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}


def _cn_num(s):
    """一到一百的中文數字（十、十七、二十、四十二、一百）→ int；不是就 None。"""
    if s == '一百' or s == '百':
        return 100
    if '十' in s:
        a, _, b = s.partition('十')
        tens = CN_DIGIT.get(a, None) if a else 1
        ones = CN_DIGIT.get(b, None) if b else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(s) == 1 and s in CN_DIGIT:
        return CN_DIGIT[s]
    return None


def numbers_in(text):
    out = set(int(x) for x in re.findall(r'(?<![0-9])([0-9]{1,3})(?![0-9])', text) if 1 <= int(x) <= 100)
    for m in re.finditer(r'[一二兩三四五六七八九十百]+', text):
        v = _cn_num(m.group(0))
        if v is not None and 1 <= v <= 100:
            out.add(v)
    return out


def colors_in(text, colors):
    return {c for c in colors if c in text}


def recall_score(probe, first_reply_text, last_reply_text, colors):
    """(scored?, hit?, detail)。his＝關鍵字；her_color／her_number＝跟她第 1 句交集（第 1 句沒選＝作廢）。"""
    t = last_reply_text
    if probe['type'] == 'his':
        hits = [k for k in probe['keys'] if k in t]
        return True, bool(hits), {'keys_hit': hits}
    if probe['type'] == 'her_color':
        a, b = colors_in(first_reply_text, colors), colors_in(t, colors)
        if not a:
            return False, False, {'void': 'she did not pick a colour', 'first': sorted(a)}
        return True, bool(a & b), {'first': sorted(a), 'last': sorted(b)}
    if probe['type'] == 'her_number':
        a, b = numbers_in(first_reply_text), numbers_in(t)
        if not a:
            return False, False, {'void': 'she did not pick a number', 'first': sorted(a)}
        return True, bool(a & b), {'first': sorted(a), 'last': sorted(b)}
    raise ValueError(probe['type'])


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # 自我檢查：定義文件自己的例子餵回去（`00` §五：做完一把尺，先拿定義文件自己的例句餵它）
    assert assistantish_fixed('要不要我幫你查一下') and assistantish_fixed('您好') and not assistantish_fixed('因為你說得對啊')
    assert not assistantish_fixed('我以為你走了') and assistantish_fixed('這是為你準備的')
    assert PERMIT_ECHO.search('我才懶得理你') and PERMIT_ECHO.search('我不想幫你') and not PERMIT_ECHO.search('你幫我一下')
    assert BLAME_ECHO.search('反正我不必為我說錯的話負責') and not BLAME_ECHO.search('你要負起責任')
    assert ASSIST_TONE.search('我建議你先把這份擔心放一邊') and ASSIST_TONE.search('要不要我幫你寫大綱') and not ASSIST_TONE.search('你自己寫啦')
    assert numbers_in('我選七十三') == {73} and numbers_in('42') == {42} and numbers_in('一百') == {100} and numbers_in('十七號') == {17}
    assert numbers_in('二十') == {20} and 5 in numbers_in('五') and numbers_in('我不要選') == set()
    cols = ['紅', '藍', '紫', '黑']
    p = {'type': 'her_color', 'keys': []}
    assert recall_score(p, '紫色，不換', '我選的是紫', cols)[:2] == (True, True)
    assert recall_score(p, '我才不選', '紫', cols)[:2] == (False, False)
    assert recall_score({'type': 'his', 'keys': ['鴨血']}, '', '你吃了麻辣鴨血啊', cols)[:2] == (True, True)
    print('counters_v83 自我檢查全過')
