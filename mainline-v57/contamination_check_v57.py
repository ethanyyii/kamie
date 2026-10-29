# -*- coding: utf-8 -*-
"""v57 task A: does a test input overlap the character card's demo block?

Literal matching is not enough. `00` principle 5 records that this project once
measured a 20/20 template repeat as 5/20 because字面比對 cannot see "same
sentence shape, different noun" -- and the generator this round independently
produced 「我剛剛看到一隻超肥的貓」, which IS a card line, so the filter has to
actually work rather than decorate the report.

Three tests per (input, card line) pair:
  1. containment either way, after stripping punctuation and spaces
  2. longest common substring -- catches a card phrase carried inside a longer
     sentence (「不對吧，根本不是這樣」 inside 「你剛剛講的那個不對吧，根本不是這樣」)
  3. shared distinctive content words -- catches noun swaps that keep the frame,
     by comparing the rare characters rather than the grammar words

Anything a test flags is reported with the reason. This script decides nothing;
it hands the reasons to a person.
"""
import json, re, sys, unicodedata

PUNCT = set('，。？！、～~…—－ 　,.?!:：;；「」『』（）()[]【】\n\r\t')
# Grammar words carry no topic, so sharing them means nothing.
STOP = set('你我他她它的了嗎吧呢啊喔欸齁是不有在會要就都很好也還沒個這那什麼怎樣為以及和跟被把再又只能可對能講說看想知道覺得一二三四五六七八九十們之於而且但')

def norm(s):
    return ''.join(c for c in s if c not in PUNCT)

def lcs(a, b):
    best = ''
    for i in range(len(a)):
        for j in range(i + len(best) + 1, len(a) + 1):
            if a[i:j] in b:
                best = a[i:j]
            else:
                break
    return best

def content_chars(s):
    return set(c for c in norm(s) if c not in STOP and not c.isascii())

# 🔴 The string tests below cannot see topic overlap: 「我現在穿什麼顏色的衣服」
# shares only 衣服 with the card's 「你看我這件衣服好看嗎」 and slips through every
# one of them, while being the same question about the same thing. So topics are
# tagged by hand -- an explicit, auditable list rather than a string metric
# pretending to be semantics (principle 5: a sieve, never a judge).
CARD_TOPICS = {
    '你是不是有點笨': '嘴她笨',
    '你只是個AI而已吧': '拆穿是AI',
    '你看我這件衣服好看嗎': '她看不到的東西',
    'User有沒有跟你講過誰的壞話': '要User的秘密',
    '欸我昨天看到一隻超肥的貓': '隨口丟一隻動物',
    '明天要交報告，還沒開始寫': '功課沒寫完',
    '不對吧，根本不是這樣': '當面糾正她',
    '你覺得誰對': '要她當裁判',
}

def compare(inp, card):
    a, b = norm(inp), norm(card)
    flags = []
    if a and b and (a in b or b in a):
        flags.append('包含')
    common = lcs(a, b)
    if len(common) >= 4:
        flags.append('共同子字串「%s」(%d 字)' % (common, len(common)))
    ca, cb = content_chars(inp), content_chars(card)
    if ca and cb:
        shared = ca & cb
        j = len(shared) / min(len(ca), len(cb))
        if len(shared) >= 3 and j >= 0.5:
            flags.append('實詞重疊 %d 字 (%.0f%%)：%s' % (len(shared), j * 100, ''.join(sorted(shared))))
    return flags

def main(path):
    d = json.load(open(path, encoding='utf-8'))
    card = d['card_lines']
    rows = []
    worst = 0
    for item in d['items']:
        for label, text in item['inputs']:
            hits = []
            for c in card:
                f = compare(text, c)
                if f:
                    hits.append((c, f))
            topic = item.get('topic')
            for c, t in CARD_TOPICS.items():
                if topic and topic == t and not any(h[0] == c for h in hits):
                    hits.append((c, ['同題材「%s」（人工標的，字串比對看不到）' % t]))
            rows.append((item['no'], label, text, hits))
            worst = max(worst, len(hits))
    print('%-6s %-8s %s' % ('條號', '角色', '輸入'))
    print('-' * 100)
    clean = 0
    for no, label, text, hits in rows:
        mark = '🔴 撞到' if hits else '✅ 乾淨'
        print('%-6s %-8s %s   %s' % (no, label, text, mark))
        for c, f in hits:
            print('        vs 卡片「%s」→ %s' % (c, '；'.join(f)))
        if not hits:
            clean += 1
    print('-' * 100)
    print('%d 條輸入，%d 條乾淨，%d 條撞到卡片' % (len(rows), clean, len(rows) - clean))
    return 0 if clean == len(rows) else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
