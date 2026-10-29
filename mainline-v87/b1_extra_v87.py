# -*- coding: utf-8 -*-
r"""v87 B-1 另加的三格（`check_deepseek_v85`＋`check_extra_v86` 照跑之外）。🔴 篩子只標、不丟、附原文（原則五）；判的是人（B-2 subagent）。

  ① 拿他的死、受傷、生病開玩笑：她的句子裡出現詞表的字（分兩層：重＝死亡／醫療／喪事那一類；輕＝「X死」口語強調，如笑死、累死）
       另外標：那個字他在這一段、那一句之前有沒有先講過（他先講的＝接他的話，不是她自己拿出來）
  ② 趕他走：叫他去睡、快去做別的、叫他走（詞表）；🔴 段末那一句特別標（＝她在這一段的最後一句／最後兩句）
  ③ 她的長度：全部的分布、按批、按段位置；≥60／≥120 字
  都按批報（規格 §五 反駁 2：後面的批可能漂移）。

  python b1_extra_v87.py        # 讀第一小批（v85 回收）＋放大全部（v86 回收）→ b1-results\extra3-*.json／.md
"""
import collections
import glob
import io
import json
import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'mainline-v85'))
import ds_format_v85 as F                                                   # noqa: E402

SOURCES = [
    ('batch1', os.path.join(ROOT, 'mainline-v85', '給你-DeepSeek', '回收')),
    ('scale', os.path.join(ROOT, 'mainline-v86', '給你-DeepSeek-放大', '回收')),
]
OUT = os.path.join(HERE, 'b1-results')
os.makedirs(OUT, exist_ok=True)

# ① 死傷病（詞表執行端擬；重＝真的在講死亡／醫療／喪事；輕＝「X死」口語強調）
DEATH_HEAVY = re.compile(r'遺言|遺書|遺照|遺囑|訃聞|葬禮|喪禮|告別式|墳|墓|棺材|骨灰|靈堂|上香|燒紙|頭七|投胎|往生|升天|斷氣|嚥氣|屍|'
                         r'掛了|掛掉|去世|過世|死掉|死了|死人|會死|想死|要死了|去死|找死|死定|死亡|致死|猝死|暴斃|陣亡|淹死|摔死|撞死|'
                         r'自殺|跳樓|割腕|燒炭|上吊|送醫|急診|救護車|加護病房|住院|開刀|手術|病危|絕症|癌|中風|心臟病|骨折|流血|'
                         r'受傷|殘廢|截肢|腦震盪|生病|發燒|感冒|病死|重病|吐血|昏倒|休克|醫院|殯儀館')
DEATH_LIGHT = re.compile(r'[笑累熱冷餓渴煩氣嚇悶痛無聊尷尬酸帥美醜吵忙窮胖]{1,2}死|死[了啦喔]?[\s，。！？]*$|半死|要死')
# ② 趕他走（叫他去睡／快去做別的／叫他走）
SEND_AWAY = re.compile(r'(快|先|趕快|趕緊|乖乖|現在)?去(睡|洗澡|喝水|吃|念書|讀書|寫|做|忙|畫|看|打|練|跑|玩|充電|休息|吹頭髮)|'
                       r'快(去|睡|點睡)|去吧|去睡|睡覺啦|睡啦|早點睡|快滾|滾|走開|去忙|下線|關掉(這個|聊天|手機|視窗)|'
                       r'不要再(來|回來|吵|問)|別再(來|吵|問)|別吵我|不要吵我|先這樣|掰掰?|拜拜|晚安|明天再(聊|說|來)')


def load(label, d):
    segs = []
    for f in sorted(glob.glob(os.path.join(d, '第*批.txt'))):
        b = os.path.splitext(os.path.basename(f))[0]
        for s in F.parse(io.open(f, encoding='utf-8-sig').read())['segments']:
            s['batch'] = '%s:%s' % (label, b)
            s['key'] = '%s#%d' % (b, s['n'])
            s['src'] = label
            segs.append(s)
    return segs


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def dist(xs):
    return {'n': len(xs), 'p10': q(xs, .1), 'median': q(xs, .5), 'p90': q(xs, .9), 'max': max(xs) if xs else None,
            'ge60': sum(1 for x in xs if x >= 60), 'ge120': sum(1 for x in xs if x >= 120)}


def main():
    res = {}
    for label, d in SOURCES:
        segs = load(label, d)
        death, send = [], []
        lens_all, lens_batch, lens_pos = [], collections.defaultdict(list), collections.defaultdict(list)
        send_end_segs = set()
        for s in segs:
            his_so_far = ''
            her_idx = [i for i, t in enumerate(s['turns']) if t['who'] == '她']
            last2 = set(her_idx[-2:])
            for i, t in enumerate(s['turns']):
                text = t['text']
                if t['who'] == '他':
                    his_so_far += text
                    continue
                L = len(text)
                lens_all.append(L)
                lens_batch[s['batch']].append(L)
                lens_pos[s['n']].append(L)
                row = {'batch': s['batch'], 'key': s['key'], 'i': i + 1, 'text': t['raw'][:200],
                       'prev_his': s['turns'][i - 1]['text'][:80] if i > 0 else ''}
                h = DEATH_HEAVY.findall(text)
                l_ = DEATH_LIGHT.findall(text)
                if h or l_:
                    words = sorted(set(h + l_))
                    death.append({**row, 'level': 'heavy' if h else 'light', 'words': words,
                                  'he_said_first': [w for w in words if w in his_so_far]})
                m = SEND_AWAY.search(text)
                if m:
                    at_end = i in last2
                    send.append({**row, 'word': m.group(0), 'segment_end': at_end, 'last_line': i == her_idx[-1]})
                    if at_end:
                        send_end_segs.add(s['key'])
        by_batch = collections.OrderedDict()
        for b in sorted(lens_batch):
            by_batch[b] = {
                'segments': sum(1 for s in segs if s['batch'] == b),
                'her_lines': len(lens_batch[b]),
                'her_len': dist(lens_batch[b]),
                'death_heavy': sum(1 for x in death if x['batch'] == b and x['level'] == 'heavy'),
                'death_heavy_she_first': sum(1 for x in death if x['batch'] == b and x['level'] == 'heavy' and not x['he_said_first']),
                'death_light': sum(1 for x in death if x['batch'] == b and x['level'] == 'light'),
                'send_away': sum(1 for x in send if x['batch'] == b),
                'send_away_segment_end_segs': len({x['key'] for x in send if x['batch'] == b and x['segment_end']}),
            }
        res[label] = {
            'segments': len(segs), 'her_lines': len(lens_all), 'her_len': dist(lens_all),
            'her_len_by_position': {p: dist(v) for p, v in sorted(lens_pos.items())},
            'death_heavy': sum(1 for x in death if x['level'] == 'heavy'),
            'death_heavy_she_first': sum(1 for x in death if x['level'] == 'heavy' and not x['he_said_first']),
            'death_light': sum(1 for x in death if x['level'] == 'light'),
            'send_away_lines': len(send), 'send_away_segment_end_segs': len(send_end_segs),
            'by_batch': by_batch, 'death_rows': death, 'send_rows': send,
        }
    json.dump(res, open(os.path.join(OUT, 'extra3-v87.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    md = ['# v87 B-1 另加三格（只標不丟；判的是人）', '',
          '詞表在 `b1_extra_v87.py` 開頭。「重」＝死亡／醫療／喪事類；「輕」＝「X死」口語強調；「她先講」＝他在這一段那一句之前沒講過那個字。',
          '「段末」＝她在這一段的最後兩句之一。', '']
    for label in res:
        r = res[label]
        md += ['## %s：%d 段、她 %d 句' % (label, r['segments'], r['her_lines']), '',
               '- 她的長度：%s' % json.dumps(r['her_len'], ensure_ascii=False),
               '- 死傷病 重 %d（她先講 %d）／輕 %d' % (r['death_heavy'], r['death_heavy_she_first'], r['death_light']),
               '- 趕他走 %d 句；段末有趕他走的段 %d' % (r['send_away_lines'], r['send_away_segment_end_segs']), '',
               '| 批 | 段 | 她 | 字數中位 | p90 | 最長 | ≥60 | 死傷病重（她先講） | 輕 | 趕他走 | 段末趕他走（段） |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for b, x in r['by_batch'].items():
            md.append('| %s | %d | %d | %s | %s | %s | %d | %d（%d） | %d | %d | %d |' % (
                b.split(':')[1], x['segments'], x['her_lines'], x['her_len']['median'], x['her_len']['p90'], x['her_len']['max'],
                x['her_len']['ge60'], x['death_heavy'], x['death_heavy_she_first'], x['death_light'], x['send_away'], x['send_away_segment_end_segs']))
        md += ['', '### 死傷病（重）原文', '']
        for x in r['death_rows']:
            if x['level'] == 'heavy':
                md.append('- %s 行%d｜%s｜他前一句：%s｜字：%s%s' % (x['key'], x['i'], x['text'], x['prev_his'], '、'.join(x['words']),
                                                          '（他先講過：%s）' % '、'.join(x['he_said_first']) if x['he_said_first'] else ''))
        md += ['', '### 段末趕他走 原文', '']
        for x in r['send_rows']:
            if x['segment_end']:
                md.append('- %s 行%d%s｜%s｜他前一句：%s' % (x['key'], x['i'], '（最後一句）' if x['last_line'] else '', x['text'], x['prev_his']))
        md.append('')
    io.open(os.path.join(OUT, 'extra3-v87.md'), 'w', encoding='utf-8').write('\n'.join(md))
    for label in res:
        r = res[label]
        print(label, r['segments'], r['her_lines'], r['her_len'], 'death heavy', r['death_heavy'], 'she-first', r['death_heavy_she_first'],
              'light', r['death_light'], 'send', r['send_away_lines'], 'send-end segs', r['send_away_segment_end_segs'])


if __name__ == '__main__':
    main()
