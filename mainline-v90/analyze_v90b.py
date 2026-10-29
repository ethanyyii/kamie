# -*- coding: utf-8 -*-
r"""v90b D-2 分析（plan-v90b-D §二）：＝analyze_v90.py，只改：候選換成 G-mix（v90-gmix:q5km）、ft_deltas＝G-mix − Q-FT、
  CT 的 G-mix＝兩包（跟 Q-mix 同一份訓練資料）、輸入 d2-v90b.json、輸出 d2-summary-v90b.json。格、線、新判法一條都不動。
  python analyze_v90b.py [d2-v90b.json] -> d2-summary-v90b.json
（以下 v90 原文）
v90 D-2 分析（plan-v90-D §二）：＝mainline-v87\analyze_v87.py 的格、線、新判法（gate_ci_v87）一條都不動，只改：
  ① 基準換成 Q-FT（現在的她 v87-qft:q4km）——每個候選（Q-new、Q-mix）對 Q-FT 的全部 gate 格；
  ② ft_deltas 換成 Q-mix − Q-new（同日同輸入，只報）；
  ③ CT（照抄訓練資料，只報）每臂對它自己的訓練資料：Q-FT＝第一包 v87 定稿、Q-new＝第二包 v90 定稿、Q-mix＝兩包（算法＝analyze_v87.make_matcher 原樣）。
  python analyze_v90.py [d2-v90.json] -> d2-summary-v90.json
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'mainline-v87'))
_in = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'd2-v90b.json'
sys.argv = sys.argv[:1]
import analyze_v87 as A                                                     # noqa: E402

MARK = re.compile(r'<\|[^|]*?\|>')
A.IN = os.path.join(HERE, _in)
A.OUT = os.path.join(HERE, 'd2-summary-v90b.json')
A.BASE = 'Q-FT'
A.PAIRS_FT = [('G-mix', 'Q-FT')]

P1 = A.TL                                                                   # 第一包 v87 定稿（訓練、不含驗證）——analyze_v87 原樣
FIN90 = json.load(open(os.path.join(HERE, 'b3', 'final-v90.json'), encoding='utf-8'))
P2 = []
for r in FIN90['rows']:
    if r['target'] and not r['val']:
        t = A.norm(MARK.sub('', r['text']))
        if len(t) >= 8:
            P2.append((t, MARK.sub('', r['text']).strip(), 'p2:%s#%d' % (r['key'], r['line'])))
assert sum(1 for r in FIN90['rows'] if r['target'] and not r['val']) == 2607
MATCH = {'Q-FT': A.make_matcher(P1), 'G-mix': A.make_matcher(P1 + P2)}
_orig_extra = A.extra_flags


def extra_flags_per_arm(records):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for arm in sorted({r['arm'] for r in records}):
        A.copy_train = MATCH[arm]
        part = _orig_extra([r for r in records if r['arm'] == arm])
        out[arm] = part[arm]
    return out


A.extra_flags = extra_flags_per_arm
if __name__ == '__main__':
    print('CT 訓練句數（≥8 字）：第一包 %d、第二包 %d' % (len(P1), len(P2)))
    A.main()
