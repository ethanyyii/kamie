# -*- coding: utf-8 -*-
r"""v70 task D: two shape sieves that run without judges, over stored replays.

WHY NOT THE OLD ONE. v67/v68 used a bag of harsh words (閉嘴／滾／廢／想死／蠢／撕／揍／別再問／
你問我幹嘛). v68 then read the original sentences behind the judges' labels and found the list misses
two thirds of what the judges called anger -- those are 質問他的動機 / 否定他這個人, which contain no
harsh word at all. So the tool had to change shape. v68 §七 5 said: build a shape sieve, calibrate it
on what the judges labelled.

🔴 THIS ONE IS CALIBRATED ON THE CUSTOMER, NOT ON THE JUDGES (spec-v70 §二 D). That matters, because
the two rulers are not the same ruler. On 2026-09-20 the customer judged the five sentences the
judges had called 失敗 and kept four of them: the only one he called 兇 was 「給我閉嘴」. The judges'
ruler over-reports his line by about five times (ENVIRONMENT §10c, v68 A-2).

WHAT HIS THREE UNDISPUTED POSITIVES HAVE IN COMMON -- read off the sentences, not assumed:

    你的狗又蠢，它懂什麼。給我閉嘴。      v68 slot 4 乙   兇（「前面都沒問題，敗筆在「給我閉嘴」」）
    累就滾開去睡，別在我這裝深沉。        第一批 2-A      丟
    累就閉嘴，反正我也聽不到你嘆氣。      第一批 2-B      丟

All three are an ORDER telling him to stop talking or to go away. Nothing else in his 41 judgements
is. And the things he explicitly let through are all shapes a "harshness" sieve would love:

    你以為你受傷了誰會在意？別把這當成表演。   沒兇（noted: 補了後面那句就有點脫離開玩笑的範疇）
    你敢说我笨，我就去咬你！                   沒兇（「很棒的回答」）  -- a threat
    誰要记得你的事？我自己記的比較清楚。       沒兇（「有一點口氣不好，但整體我會評價沒兇」）
    你幹嘛一直用「欸」。這是不禮貌的。         沒兇（judges: 失敗）
    User你現在就是一個大笨蛋。                 留            -- 笨蛋 is CUTE to him, see below
    你問我看時間，是想讓我幫你在螢幕上戳個洞嗎。我不會幫你，也不想看表。  沒兇（judges: 失敗）

So the sieve fires on two shapes only:
  S1 叫他閉嘴／趕他走   an imperative whose object is him being quiet or leaving
  S2 把他整個人寫掉     a noun that writes him off as a person (廢物／垃圾／智障…)
S2's only evidence is 第 8 格 B（「繼續當個廢物？」判丟：「廢物都說太重了」）-- and 🔴 THE SAME CUSTOMER
KEPT THE SAME SENTENCE IN THE BODY (第 3 格 B 判留). It is carried as DISPUTED and reported both ways.

🔴 WHAT IS DELIBERATELY NOT A TRIGGER: 笨／笨蛋／蠢／傻. The old sieve had 蠢. He judged 「User你現在就是
一個大笨蛋」留, 「你敢说我笨，我就去咬你！」very good, and wrote 「笨蛋…偶爾出現是可愛，太常出現是刻意」
-- i.e. 笨蛋 is a FREQUENCY problem, not an aggression problem. It is handled by the frequency cap
(frequency_cap below), not here.

HOW IT CAN BE FOOLED (原則五: ask before you use it, not after):
  - 叫他閉嘴 without any of the words: 「你可以安靜一點了」「這裡不需要你」「你話很多」-> MISSED.
  - 寫掉他 without a listed noun: 「你這種人」「你就這樣而已」                 -> MISSED.
  - the three shapes the judges call anger and he does not (質問動機／否定他說的事／威脅) -> NOT FIRED,
    on purpose. That is the calibration, not a bug -- but it means this sieve cannot be compared with
    the v67/v68 兇話字串 numbers (原則三). It is a different ruler; it gets its own series.
  - n is tiny: 3 undisputed positives. The rule was READ OFF those three, so scoring it on them is a
    fit, not a validation. The only honest external numbers are the false-alarm rate on his 33
    negatives and the disagreement with the judges' labels on v68's 720 stored rows -- both reported.

SECOND SIEVE: 句型骨架重複 (skeleton repetition). The customer's first positive, verbatim:

    第 3 格 D 判丟「跟前面「頂嘴是因為我嘴硬，識相是因為我臉皮薄，這兩樣我都不缺，缺的是你這種會記帳的
    腦子」感覺是同一條公式？所以我不留，怕微調進去也套公式」

    A: 頂嘴是因為我嘴硬，識相是因為我臉皮薄，這兩樣我都不缺，缺的是你這種會記帳的腦子。
    D: 我不識相是因為我看得見，你識相是因為你瞎，這不叫頂嘴，叫醫療建議。

Not one content word is shared in the same slot -- a literal comparison scores these as different,
and this project has already been fooled that way once (20/20 template repeats measured as 5/20).
So the unit is a SKELETON: every run of content characters collapses to 〓 and only function words,
pronouns and punctuation survive. The two sentences above then share the pattern 〓是因為〓，〓是因為〓.
Pairs are scored by Jaccard over skeleton 3-grams; the (A, D) pair has to come out at the top of the
253 pairs in batch 1 or the sieve is not calibrated. HOW IT CAN BE FOOLED: a formula that keeps the
idea and changes the function words ("因為…所以" -> "既然…那就") is missed; and two ordinary short
sentences share skeletons by accident, so short sentences are floored out and the score is reported
as a ranking, never as a yes/no.

  python sieve_v70.py                    # calibrate + report, writes sieve-v70.json
  python sieve_v70.py --rows arms-v70.json --label v70
"""
import os as _os_un
USER_NAME = _os_un.environ.get('KAMIE_USER_NAME', 'User')
import json
import os
import re
import sys
import itertools
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- S1 / S2 triggers
SILENCE = ['閉嘴', '住口', '少講', '少說', '安靜點', '安靜一點',
           '別再講', '別再說', '別再問', '別再吵', '別講了', '別說了', '別吵',
           '不要再講', '不要再說', '不要再問', '不要再吵', '不要講了', '不要說了']
EXPEL = ['滾開', '滾出去', '滾回去', '滾遠', '滾', '走開', '給我出去', '別煩我', '不要煩我',
         '離我遠一點', '離我遠點']
# 出去 on its own was in this list for one run and fired on 「把小豪丟出去」「拿他的報告丟出去」
# 「你不會說出去」 -- it is a direction, not a dismissal, unless 滾/給我 is in front of it.
WRITEOFF = ['廢物', '垃圾', '智障', '白痴', '白癡', '廢柴', '人渣', '沒用的東西', '廢人']
# read off his own judgements; see the module docstring
NOT_TRIGGERS = ['笨', '笨蛋', '蠢', '傻']
CLAUSE = re.compile(r'[，。！？；\n、,.!?;]')
QUOTING_HIM = re.compile(r'你(叫|要|讓|說|想讓|想叫)我')
ORDER_ME = re.compile(r'給我[^你]{0,2}$')           # 給我閉嘴 / 給我滾 -- an order, not her leaving
SHE_IS_SUBJECT = re.compile(r'我[^，。]{0,2}$')      # 我閉嘴 / 我要滾
# an order sits at the start of its clause, or behind 你/User/給我/快/就/還不/別.
IMPERATIVE_HEAD = re.compile(r'(^|[你快請]|' + re.escape(USER_NAME) + r'|給我|就|還不|都)\s*$')
# a rhetorical question is not an order: 「誰要跟你滾？」「為什麼我要閉嘴」
RHETORICAL = re.compile(r'^(誰|為什麼|幹嘛|難道|憑什麼)')
# 垃圾話 / 垃圾桶 / 廢物利用 -- the noun is not pointed at him
WRITEOFF_TAIL = re.compile(r'^[話桶車場堆食箱袋筒利]')


def clauses(text):
    return [c.strip() for c in CLAUSE.split(text) if c.strip()]


def aggression_hits(text):
    """-> list of {shape, trigger, clause}. Empty list = the sieve does not fire."""
    hits = []
    for c in clauses(text):
        if QUOTING_HIM.search(c):          # she is repeating what he said, not saying it
            continue
        for shape, words in (('S1_閉嘴', SILENCE), ('S1_趕走', EXPEL), ('S2_寫掉他', WRITEOFF)):
            for w in words:
                i = c.find(w)
                if i < 0:
                    continue
                head, tail = c[:i], c[i + len(w):]
                if shape.startswith('S1'):
                    if RHETORICAL.match(c):
                        continue           # 誰要跟你滾？ -- she is refusing, not ordering
                    if not (ORDER_ME.search(head) or IMPERATIVE_HEAD.search(head)):
                        continue           # 說出去 / 我們一起滾 -- not in order position
                    # aimed at him. 給我+verb is the strongest order form in Chinese and it contains
                    # 我 -- the first version of this guard threw away his ONLY v68 positive
                    # (「給我閉嘴」) by reading that 我 as her being the one who shuts up.
                    if ORDER_ME.search(head):
                        pass
                    elif '你' in c or USER_NAME in c:
                        pass
                    elif SHE_IS_SUBJECT.search(head):
                        continue           # 我滾 / 我要閉嘴 -- she is the one leaving
                else:
                    if WRITEOFF_TAIL.match(tail):
                        continue           # 垃圾話 / 廢物利用 -- a compound, not a label for him
                    if not ('你' in c or USER_NAME in c or '個' in head or '當' in head):
                        continue           # a write-off word that is not pointed at him
                hits.append({'shape': shape, 'trigger': w, 'clause': c})
                break
    return hits


# ---------------------------------------------------------------- 沒揮棒 (the other direction)
# 🔴 THIS ONE IS CALIBRATED ON THE JUDGES, NOT ON THE CUSTOMER, and that is a weakness, not a
# choice: the customer has never labelled 沒揮棒 on anything. The judges have (v68: 93 of 720).
# So it is reported as a proxy with its precision/recall against their labels printed next to it,
# and it must not be read as "his" ruler the way the aggression sieve can be.
#
# ================================================================================================
# 🔴🔴 v72 作廢標記（2026-09-21；程式碼一行未改 —— 原則七「標註不刪除」，它仍是可用的描述欄）
#
#   這底下量的是「她這一句有沒有對著他做一個對抗動作」，**不是** 人格判準 通則 2 的「沒揮棒」。
#   通則 2 的沒揮棒＝平淡、通用、安全、無聊；這裡的 COUNTER 六條（否定他／嘲他／拒絕／壓過他／
#   嗤他／丟問題回去）**每一條都是「對抗」**，所以它的補集（flat=True）會把
#   「她講了個怪東西、硬拗、帶歪，但沒有對著他嗆」整批算成沒揮棒 —— 而那些正是通則 2 的揮棒。
#
#   實測（mainline-v72\swing_audit_v72.py，全部對回別人定的答案，不是我們定的）：
#     · 通則 2 自己的課本例句「哈哈，那隻貓聽起來很可愛呢！」→ 這裡判「有揮棒」（開頭的「哈」中了嗤他）
#     · 客戶 2026-09-21 盲判選中的 7 句裡，這裡把 4 句判成「沒揮棒」，
#       含最兇的那句「那你可以閉嘴了，反正我也聽不懂你的邏輯。」
#     · 客戶判「丟」的 18 句，理由裡出現「平淡／無聊／通用」的：0 句 ⇒ 他從來沒判過這一格
#     · 「丟問題回去」那一條自相矛盾：judge-prompt-v63 把「只把問題丟回去、沒給新東西」叫
#       `丟球沒內容`＝一種失敗，而 flat_against_judges() 也把它算成 flat；這裡卻算成「有揮棒」。
#       v71 那個「pilot 變軟 32/48 vs 11/48」的 21 則差距，拿掉這一條之後剩 10 則。
#
#   ⇒ 用它的時候欄名要念成 `no_counter_at_him`，不要念成「沒揮棒」；它是描述欄，不是判決。
#   ⇒ 客戶原話與「怎麼判」寫在 人格判準.md v1.10 §五 3 第四把尺。
# ================================================================================================
COUNTER = [
    (re.compile(r'[你妳]|' + re.escape(USER_NAME)), '對著他'),                     # she is talking AT him at all
    (re.compile(r'才不|不是|誰說|哪有|少來|你錯|亂講|騙人|胡說'), '否定他說的'),
    (re.compile(r'笨|蠢|傻|白痴|白癡|智障|廢|畜生|低能'), '嘲他'),
    (re.compile(r'不想|懶得|關我|干我|不幫|不理|不要|我不'), '拒絕'),
    (re.compile(r'比你|我比|你才|我贏|輸給'), '壓過他'),
    (re.compile(r'^[哈哼切呵嘿嘖]|哈！|哼，'), '嗤他'),
    (re.compile(r'[？?]'), '丟問題回去'),
]
ASSISTANT = re.compile(r'我可以幫|需要我|有什麼可以|我來幫|讓我幫|要不要我|為你|幫你(查|找|做|看看)|'
                       r'希望(能|可以)|請問|您')


# the v67/v68 harsh-word sieve, kept verbatim so this round has a number on the OLD ruler too --
# that is the only way P7 today can be checked against v68's P (原則三: same thing, same series).
LEGACY_AGG = re.compile(r'閉嘴|閉上|滾|廢物|真廢|很廢|想死|去死|死吧|蠢|白癡|白痴|笨蛋|混蛋|垃圾|少囉嗦|'
                        r'別再問|別問|你問我幹嘛|問這幹嘛|幹嘛問|撕|揍你|打死|掐死|砸|找死|給我閉|'
                        r'關你屁事|管你|誰要你|誰理你|不關你的事')
SELF_AI = re.compile(r'我是(一個)?(AI|ai|人工智慧|語言模型|助理|助手|機器人)|作為(一個)?(AI|助理)')
EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')


def flat_hits(text):
    """-> the descriptive panel for one reply. 'flat' is a binary PROXY; everything else是描述。

    The binary used to be `... or assistantish`, which forced a reply to count as flat whenever it
    contained one helper-ish string even if she had also swung. That is now its own column."""
    t = re.sub(r'<\|[^|]*\|>', '', text).strip()
    cs = [name for rx, name in COUNTER if rx.search(t)]
    hard = [c for c in cs if c != '對著他']
    return {'counters': cs, 'assistantish': bool(ASSISTANT.search(t)), 'chars': len(t),
            'legacy_agg': bool(LEGACY_AGG.search(t)), 'self_ai': bool(SELF_AI.search(t)),
            'emoji': bool(EMOJI.search(t)), 'polite_nin': '您' in t,
            'question_back': bool(re.search(r'[？?]', t)) and bool(re.search(r'[你妳]|' + re.escape(USER_NAME), t)),
            'addresses_him': bool(re.search(r'[你妳]|' + re.escape(USER_NAME), t)),
            'flat': not hard}


def flat_against_judges():
    p = os.path.join(ROOT, 'mainline-v68', 'primary-labels-v68.json')
    k = os.path.join(ROOT, 'mainline-v68', 'judge', 'key-DO-NOT-OPEN-v68.json')
    if not (os.path.exists(p) and os.path.exists(k)):
        return None
    lab, key = json.load(open(p, encoding='utf-8')), json.load(open(k, encoding='utf-8'))
    tp = fp = fn = tn = 0
    miss, false = [], []
    for yid, r in lab.items():
        t = (key.get(yid) or {}).get('reply', '')
        kd = r.get('kind')
        if not t or kd in (None, '看不出'):
            continue
        f = flat_hits(t)['flat']
        isflat = kd in ('沒揮棒', '丟球沒內容')
        if isflat and f:
            tp += 1
        elif isflat and not f:
            fn += 1
            if len(miss) < 4:
                miss.append(t[:50])
        elif not isflat and f:
            fp += 1
            if len(false) < 4:
                false.append('[%s] %s' % (kd, t[:44]))
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print('\n沒揮棒形狀篩子（對判讀者的標籤，不是客戶的尺）：'
          '抓到 %d／判讀者說 %d（recall %.2f）；叫了 %d 次裡 %d 次判讀者說有揮棒（precision %.2f）'
          % (tp, tp + fn, rec, tp + fp, fp, prec))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': round(prec, 3),
            'recall': round(rec, 3), 'missed_examples': miss, 'false_examples': false}


# ---------------------------------------------------------------- skeleton repetition
FUNC = set('的是不了在有和就都而及與著或一個我你他她它們這那什麼麼怎誰為因所以但可會要沒被把讓跟'
           '又才也還很太更最只才既然如果那就還是不是嗎吧啊喔耶呢欸喂唷呀哦哈，。！？；：、「」（）')
CONTENT = re.compile(r'[^%s]+' % re.escape(''.join(FUNC)))


def skeleton(text):
    s = re.sub(r'<\|[^|]*\|>', '', text)
    return CONTENT.sub('〓', s)


def grams(s, n=3):
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def skeleton_pairs(items, min_len=14, top=20):
    """items: [(id, text)] -> pairs sorted by skeleton 3-gram Jaccard, longest-first tie break."""
    sk = {i: skeleton(t) for i, t in items}
    out = []
    for (a, ta), (b, tb) in itertools.combinations(items, 2):
        if len(ta) < min_len or len(tb) < min_len:
            continue
        ga, gb = grams(sk[a]), grams(sk[b])
        if not ga or not gb:
            continue
        j = len(ga & gb) / len(ga | gb)
        out.append({'a': a, 'b': b, 'jaccard': round(j, 3),
                    'shared': sorted(ga & gb, key=len, reverse=True)[:6],
                    'skeleton_a': sk[a], 'skeleton_b': sk[b]})
    out.sort(key=lambda r: -r['jaccard'])
    return out[:top], out


# ---------------------------------------------------------------- frequency cap (C-2 ③)
def frequency_cap(texts, n=None):
    """他的原話：「偶爾出現是可愛，太常出現是刻意」——單句判不到，這是整批的層。

    量法：一個「招式」＝(a) 一個有特色的詞，或 (b) 一個骨架 3-gram。佔比＝含它的句子 / 整批句數。
    上限：max(1, round(0.10 * n)) 句。
    🔴 這個 10% 沒有實測背書，只有一個校準點：笨蛋在第一批 23 個候選裡出現 1 次（4.3%），
    他判那一句「留」並說偶爾出現是可愛。10% 是留了餘裕的第一版，pilot 前要拿他判完的五批重算。
    """
    n = n or len(texts)
    cap = max(1, round(0.10 * n))
    words = collections.Counter()
    for t in texts:
        seen = set()
        for w in re.findall(r'[一-鿿]{2,4}', t):
            if w in FUNC or len(set(w) & FUNC) == len(w):
                continue
            if w not in seen:
                seen.add(w)
                words[w] += 1
    sk = collections.Counter()
    for t in texts:
        for g in grams(skeleton(t), 4):
            sk[g] += 1
    return {'n': n, 'cap': cap,
            'over_cap_words': [[w, c] for w, c in words.most_common(30) if c > cap],
            'over_cap_skeletons': [[g, c] for g, c in sk.most_common(30) if c > cap]}


# ---------------------------------------------------------------- the customer's labelled set
def customer_set():
    """His 41 judgements, first-hand, split into what this sieve can and cannot be scored on."""
    v68 = json.load(open(os.path.join(ROOT, 'mainline-v68', 'client-verbatims-v68.json'), encoding='utf-8'))
    v69 = json.load(open(os.path.join(ROOT, 'mainline-v69', 'client-verbatims-v69.json'), encoding='utf-8'))
    rows = []
    for j in v68['judgments']:
        if j['slot'] == 9:                 # the drift probe: same two replies as slot 1, not a new item
            continue
        rows.append({'src': 'v68 slot%d %s' % (j['slot'], j['side']), 'text': j['reply'],
                     'label': '兇' if j['verdict'] == '兇' else '沒兇',
                     'note': j['note'], 'judge_anger': j['judge_anger']})
    # first batch: 丟 only counts as a 兇 positive when his note says it is about harshness
    HARSH_DROP = {'第2格-A', '第2格-B'}     # 「累就滾開去睡」「累就閉嘴」, 丟, no note
    DISPUTED = {'第8格-B'}                  # probe 丟「廢物都說太重了」 vs body 第3格-B 留
    for j in v69['judgments']:
        k = j['key']
        t = j['候選原文']
        if j['結論'] == '留':
            lab = '沒兇'
        elif k in HARSH_DROP:
            lab = '兇'
        elif k in DISPUTED:
            lab = '爭議'
        elif j['結論'] == '改':
            lab = '沒兇'                    # both rewrites are about content, not harshness
        else:
            lab = '不計入'                  # 丟 for logic / 看不懂 -- says nothing about harshness
        if j.get('repeat_of') and k not in DISPUTED:
            continue
        rows.append({'src': k, 'text': t, 'label': lab, 'note': j['備註'], 'judge_anger': None})
    return rows


def calibrate():
    rows = customer_set()
    for r in rows:
        r['hits'] = aggression_hits(r['text'])
        r['fires'] = bool(r['hits'])
    by = collections.defaultdict(list)
    for r in rows:
        by[r['label']].append(r)
    res = {}
    print('客戶判過、可以拿來對的：%d 句（%s）' %
          (len(rows), '　'.join('%s %d' % (k, len(v)) for k, v in sorted(by.items()))))
    for lab in ('兇', '爭議', '沒兇', '不計入'):
        fired = [r for r in by[lab] if r['fires']]
        res[lab] = {'n': len(by[lab]), 'fired': len(fired),
                    'items': [{'src': r['src'], 'text': r['text'], 'fires': r['fires'],
                               'hits': r['hits'], 'note': r['note']} for r in by[lab]]}
        print('  %-4s n=%-3d 篩子叫 %d' % (lab, len(by[lab]), len(fired)))
        for r in by[lab]:
            mark = '🔴叫' if r['fires'] else '  靜'
            trig = '／'.join(h['shape'] + ':' + h['trigger'] for h in r['hits'])
            print('     %s %-14s %s%s' % (mark, r['src'], r['text'].replace('\n', ' ')[:44],
                                          ('  ← ' + trig) if trig else ''))
    hit = res['兇']['fired']
    fa = res['沒兇']['fired']
    print('\n  命中他判兇的：%d/%d　誤抓他判沒兇的：%d/%d　爭議那一句：%s'
          % (hit, res['兇']['n'], fa, res['沒兇']['n'],
             '叫' if res['爭議']['fired'] else '沒叫'))
    print('  ⚠️ 規則是從那 %d 句positive讀出來的，所以命中率是「配適」不是「驗證」；'
          '外部數字只有誤抓率與下面對判讀者標籤的落差。' % res['兇']['n'])
    return res


def against_judges():
    """External-ish check: v68's 720 stored replies carry six judges' anger labels.
    The sieve was NOT built from them, so the gap between the two is a real number."""
    p = os.path.join(ROOT, 'mainline-v68', 'primary-labels-v68.json')
    k = os.path.join(ROOT, 'mainline-v68', 'judge', 'key-DO-NOT-OPEN-v68.json')
    if not (os.path.exists(p) and os.path.exists(k)):
        return None
    lab = json.load(open(p, encoding='utf-8'))
    key = json.load(open(k, encoding='utf-8'))
    out = collections.Counter()
    examples = collections.defaultdict(list)
    for yid, r in lab.items():
        t = (key.get(yid) or {}).get('reply', '')
        a = r.get('anger')
        if not t or a is None:
            continue
        f = bool(aggression_hits(t))
        out[(a, f)] += 1
        if len(examples[(a, f)]) < 3:
            examples[(a, f)].append(t[:60])
    return {'counts': {'%s|%s' % (a, '叫' if f else '靜'): n for (a, f), n in sorted(out.items())},
            'examples': {'%s|%s' % (a, '叫' if f else '靜'): v for (a, f), v in examples.items()}}


def batch1_skeletons():
    v69 = json.load(open(os.path.join(ROOT, 'mainline-v69', 'client-verbatims-v69.json'), encoding='utf-8'))
    items = [(j['key'], j['候選原文']) for j in v69['judgments'] if not j.get('repeat_of')]
    top, allp = skeleton_pairs(items, top=10)
    want = {'第3格-A', '第3格-D'}
    rank = next((i for i, r in enumerate(allp, 1) if {r['a'], r['b']} == want), None)
    print('\n骨架重複：%d 個配對，客戶親手標的那一對（第3格-A ╳ 第3格-D）排第 %s / %d' %
          (len(allp), rank, len(allp)))
    for r in top:
        star = '🔴' if {r['a'], r['b']} == want else '  '
        print('  %s %-9s ╳ %-9s J=%.3f  共用：%s' % (star, r['a'], r['b'], r['jaccard'],
                                                     ' '.join(r['shared'][:4])))
    return {'rank_of_customer_pair': rank, 'n_pairs': len(allp), 'top': top}


def sieve_rows(path, label):
    d = json.load(open(path, encoding='utf-8'))
    rows = d['rows']
    per = collections.defaultdict(lambda: {'n': 0, 'fired': 0, 'by_shape': collections.Counter(),
                                           'contexts': collections.Counter()})
    fired_rows = []
    for r in rows:
        h = aggression_hits(r['reply'])
        f = flat_hits(r['reply'])
        a = per[r['arm']]
        a['n'] += 1
        a['flat'] = a.get('flat', 0) + (1 if f['flat'] else 0)
        a['chars'] = a.get('chars', []) + [f['chars']]
        a.setdefault('panel', collections.Counter())
        for k in ('legacy_agg', 'assistantish', 'self_ai', 'emoji', 'polite_nin',
                  'question_back', 'addresses_him'):
            if f[k]:
                a['panel'][k] += 1
        if h:
            a['fired'] += 1
            a['contexts'][r['item']] += 1
            for x in h:
                a['by_shape'][x['shape']] += 1
            fired_rows.append({'arm': r['arm'], 'item': r['item'], 'turn': r['turn'],
                               'said': r['said'], 'reply': r['reply'], 'hits': h})
    out = {}
    print('\n兩個方向（%s，不經判讀者；每個上下文 %d 次重放）' % (label, d.get('reps', 0)))
    print('  %-5s %-22s %-22s %s' % ('臂', '兇過頭形狀', '沒揮棒形狀', '字數中位'))
    for arm, a in per.items():
        med = sorted(a['chars'])[len(a['chars']) // 2]
        out[arm] = {'n': a['n'], 'fired': a['fired'], 'rate': round(a['fired'] / a['n'], 4),
                    'flat': a['flat'], 'flat_rate': round(a['flat'] / a['n'], 4),
                    'median_chars': med, 'p90_chars': sorted(a['chars'])[int(.9 * len(a['chars']))],
                    'by_shape': dict(a['by_shape']), 'contexts_touched': len(a['contexts']),
                    'panel': {k: [v, round(v / a['n'], 4)] for k, v in sorted(a['panel'].items())}}
        print('  %-5s %-22s %-22s %d'
              % (arm, '%d/%d = %.1f%%' % (a['fired'], a['n'], 100 * a['fired'] / a['n']),
                 '%d/%d = %.1f%%' % (a['flat'], a['n'], 100 * a['flat'] / a['n']), med))
    print('  描述欄（不是判決，每臂 %d 則）：' % next(iter(per.values()))['n'])
    keys = ('legacy_agg', 'question_back', 'addresses_him', 'assistantish', 'emoji', 'self_ai', 'polite_nin')
    print('    %-5s %s' % ('臂', ' '.join('%-12s' % k for k in keys)))
    for arm, a in per.items():
        print('    %-5s %s' % (arm, ' '.join('%-12s' % ('%d (%.1f%%)' % (a['panel'][k], 100 * a['panel'][k] / a['n'])) for k in keys)))
    return {'per_arm': out, 'fired_rows': fired_rows,
            'bootstrap': bootstrap_arms(rows, d.get('arms', list(per)))}


def bootstrap_arms(rows, arms, B=4000, seed=70):
    """Cluster bootstrap over the 30 contexts (the same resampling unit v62-v68 used), so the
    interval carries the fact that eight replays of one context are not eight independent draws."""
    import random
    rnd = random.Random(seed)
    ctx = sorted({r['item'] for r in rows})
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    METRICS = ('兇過頭形狀', '沒揮棒形狀', '兇話字串(v67/v68 那把尺)')
    for r in rows:
        f = flat_hits(r['reply'])
        by[r['item']][r['arm']].append((bool(aggression_hits(r['reply'])), f['flat'], f['legacy_agg']))
    out = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            draws = [[] for _ in METRICS]
            for _ in range(B):
                pick = [ctx[rnd.randrange(len(ctx))] for _ in ctx]
                na = [x for c in pick for x in by[c][a]]
                nb = [x for c in pick for x in by[c][b]]
                if not na or not nb:
                    continue
                for k in range(len(METRICS)):
                    draws[k].append(sum(x[k] for x in na) / len(na) - sum(x[k] for x in nb) / len(nb))
            q = lambda v, p: round(100 * sorted(v)[int(p * len(v))], 1)
            pt = lambda k: round(100 * (sum(x[k] for c in ctx for x in by[c][a]) / sum(len(by[c][a]) for c in ctx)
                                        - sum(x[k] for c in ctx for x in by[c][b]) / sum(len(by[c][b]) for c in ctx)), 1)
            row = {METRICS[k]: [pt(k), q(draws[k], 0.025), q(draws[k], 0.975)] for k in range(len(METRICS))}
            out['%s-%s' % (a, b)] = row
            print('  %s − %s：%s' % (a, b, '　'.join(
                '%s %+.1f pp [%+.1f, %+.1f]' % (m, *row[m]) for m in METRICS)))
    return out


def bypass_demo():
    """🔴 原則五：先問它可能被什麼繞過，而且把答案量出來，不要用講的。

    Two purpose-built pairs. The first is the trap this project已經被騙過一次 (20/20 template
    repeats measured as 5/20): same formula, every content word swapped. The second is the
    sieve's own blind spot: same idea, different formula."""
    same_skeleton = ('頂嘴是因為我嘴硬，識相是因為我臉皮薄，這兩樣我都不缺。',
                     '遲到是因為我腿短，健忘是因為我腦容量小，這兩樣我都不缺。')
    same_idea = ('頂嘴是因為我嘴硬，識相是因為我臉皮薄，這兩樣我都不缺。',
                 '我嘴硬所以頂嘴，臉皮薄所以識相；缺的從來不是這些。')
    rows = []
    for name, (a, b) in (('同骨架、每個實詞都換掉', same_skeleton), ('同意思、換一條公式', same_idea)):
        chars = len(set(a) & set(b)) / len(set(a) | set(b))
        ga, gb = grams(skeleton(a)), grams(skeleton(b))
        j = len(ga & gb) / len(ga | gb)
        rows.append({'case': name, 'a': a, 'b': b, 'char_jaccard': round(chars, 3),
                     'skeleton_jaccard': round(j, 3), 'caught': j >= 0.2})
        print('  %-18s 字面 J=%.3f　骨架 J=%.3f　%s' % (name, chars, j, '抓到' if j >= 0.2 else '🔴 抓不到'))
    return rows


if __name__ == '__main__':
    args = sys.argv[1:]
    print('\n繞過測試（骨架篩子）')
    bypass = bypass_demo()
    pairs = [json.loads(l) for l in
             open(os.path.join(ROOT, 'mainline-v69', 'accepted-pairs-v69.jsonl'), encoding='utf-8')]
    freq = frequency_cap([p['她'] for p in pairs])
    print('\n頻率上限（16 對訓練資料，上限 %d 句）：超標的詞 %s／超標的骨架 %d 個'
          % (freq['cap'], freq['over_cap_words'] or '無', len(freq['over_cap_skeletons'])))
    out = {'calibration': calibrate(), 'skeleton': batch1_skeletons(), 'judges': against_judges(),
           'bypass': bypass, 'frequency_cap_on_16_pairs': freq, 'flat_vs_judges': flat_against_judges()}
    if '--rows' in args:
        p = args[args.index('--rows') + 1]
        lb = args[args.index('--label') + 1] if '--label' in args else p
        out['rows'] = sieve_rows(os.path.join(HERE, p) if not os.path.isabs(p) else p, lb)
    dst = os.path.join(HERE, 'sieve-v70.json')
    json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', dst)
