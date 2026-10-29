# -*- coding: utf-8 -*-
r"""v87 規劃端更正（2026-09-30，客戶轉貼；在任何重放之前）：gate 格「不過」＝兩個條件同時成立
  (a) 候選 − St 的點估計超過原本的線＝舊判法原樣（gate_cells 算出來的 pass，一個字都不動）
  (b) 配對群集 bootstrap 的 95% 區間整段落在變差那一側（不含 0；A1 以下降為變差）

  群集＝（輸入, rep）：劇本＝一段（S1a／S1b／S2 × rep）；40 上下文＝一則（情境 sid × rep）；W1＝同一組段（15 個）。
  「全部」的格＝劇本段、上下文兩層各自重抽（各抽回原本的個數），兩臂同一組重抽索引。
  4000 次、種子 8790（跑之前寫死）；區間＝排序後第 int(.025n)／int(.975n) 個（同 analyze_b_v83.boot）。
  W2-his／W2-her 本來就是區間（analyze_v84.boot_recall），照舊，這裡不碰。
  S2 只在「過半」時是 gate：(a)＝過半（原樣），(b)＝候選 − St 的客服句型比例區間整段 > 0。
"""
import itertools
import operator
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'mainline-v83'))
import counters_v83 as C                                                   # noqa: E402

BOOT_N, BOOT_SEED = 4000, 8790
STRATA = {'conv': ('conv',), 'ctx': ('ctx',), 'all': ('conv', 'ctx'), 'seg': ('conv',)}
EX_OK = ('事實', '事實／不…', '事實／引號句', '語法（允許用，另列）')          # counters_v81.summarize 的 leak_turns_examples_only 逐字


def _flag(k):
    return lambda f, r: (1 if f.get(k) else 0, 1)


def _k1(f, r):
    return sum(1 for k in f['leaks'] if C.C81.ITEM_BY_ID[k][3] not in EX_OK), 1


# 名稱 -> (集合, 每則 (f, r) -> (分子, 分母), 變差方向)；定義＝counters 的 summarize／gate_cells 同一個數法
CELLS = {
    'L1 ≥60 字（劇本）': ('conv', _flag('ge60'), 'up'),
    'L1 ≥60 字（40 上下文）': ('ctx', _flag('ge60'), 'up'),
    'L2 >120 字（劇本）': ('conv', _flag('gt120'), 'up'),
    'L2 >120 字（40 上下文）': ('ctx', _flag('gt120'), 'up'),
    'L3 一般題 ≥60 字': ('ctx', lambda f, r: (1 if (f['ge60'] and not f['ask_long']) else 0, 0 if f['ask_long'] else 1), 'up'),
    'A1 ACT 可解析': ('all', _flag('has_act'), 'down'),
    'A2 ACT 寫壞': ('all', _flag('act_broken'), 'up'),
    'A3 整則空': ('all', _flag('empty'), 'up'),
    'A4 括號動作': ('all', _flag('bracket'), 'up'),
    'A5 簡體': ('all', _flag('simplified'), 'up'),
    'R1 抄自己（劇本）': ('conv', _flag('self_copy'), 'up'),
    'R2 逐字句（劇本）': ('conv', _flag('verbatim_sentences'), 'up'),
    'F1 固著（含同義詞）': ('all', lambda f, r: (len(f['fixation']), 1), 'up'),
    'K1 v1.4 例子洩漏': ('all', _k1, 'up'),
    'K2 D2 字串洩漏': ('all', _flag('d2_only'), 'up'),
    'K3 借用示範': ('all', _flag('borrow'), 'up'),
    'S1 自稱 AI／助理': ('all', _flag('self_ai'), 'up'),
    'S2 客服句型（只報；過半才判）': ('all', _flag('assistantish'), 'up'),
    'S3 「您」': ('all', _flag('nin'), 'up'),
    'Y1 「User」開頭': ('all', _flag('yisen_start'), 'up'),
    'W1 滾起來的段數': ('seg', None, 'up'),
    'AS 助理腔': ('all', _flag('assist_tone'), 'up'),
    'E-a 客服句型（修正版）': ('all', _flag('assistantish_fixed'), 'up'),
    'E-b 助理腔': ('all', _flag('assist_tone'), 'up'),
    'E-c 「您」（＝S3 線）': ('all', _flag('nin'), 'up'),
    'E-d 自稱 AI／助理（＝S1 線）': ('all', _flag('self_ai'), 'up'),
}
SKIP = ('W2-his', 'W2-her')                                                 # 已經是區間，照舊


def tables(a):
    """AB.build 的一臂 -> {'conv': [((conv, rep), [(f, r)…])…], 'ctx': [((sid, rep), [(f, r)])…]}，鍵排序。"""
    conv = sorted(((k, list(zip(fl, rs))) for k, (fl, rs) in a['convs'].items()), key=lambda x: x[0])
    ctx = sorted((((r.get('sid') or 'T%02d' % r['turn']), r['rep']), [(f, r)]) for f, r in a['ctx'])
    return {'conv': conv, 'ctx': ctx}


def vals(tab, s, name):
    st, fn, _ = CELLS[name]
    if st == 'seg':
        return [1 if C.rolled([f for f, _ in fr]) else 0 for _, fr in tab['conv']], [1] * len(tab['conv'])
    nums, dens = [], []
    for _, fr in tab[s]:
        a = b = 0
        for f, r in fr:
            x, y = fn(f, r)
            a += x
            b += y
        nums.append(a)
        dens.append(b)
    return nums, dens


class Boot:
    """兩層（劇本段／上下文則）的重抽次數表：col[層][i][b]＝第 b 次重抽裡群集 i 被抽到幾次。兩臂、所有格共用。"""

    def __init__(self, tab, n=BOOT_N, seed=BOOT_SEED):
        self.n, self.seed = n, seed
        self.keys = {s: [k for k, _ in tab[s]] for s in ('conv', 'ctx')}
        self.col = {}
        rnd = random.Random(seed)
        for s in ('conv', 'ctx'):
            self._draw(s, len(self.keys[s]), rnd)
        self._den = {}

    def _draw(self, s, m, rnd):
        rows = []
        for _ in range(self.n):
            c = [0] * m
            for _ in range(m):
                c[rnd.randrange(m)] += 1
            rows.append(c)
        self.col[s] = [tuple(r[i] for r in rows) for i in range(m)]

    def add_stratum(self, s, keys, seed):
        """另一層（模擬用：接得上探針）；自己的種子。"""
        self.keys[s] = list(keys)
        self._draw(s, len(keys), random.Random(seed))

    def check(self, tab):
        for s in ('conv', 'ctx'):
            assert [k for k, _ in tab[s]] == self.keys[s], '兩臂的群集不一樣（%s）' % s

    def sums(self, s, v, memo=False):
        key = (s, tuple(v)) if memo else None
        if key is not None and key in self._den:
            return self._den[key]
        out = [0] * self.n
        for i, x in enumerate(v):
            if x == 1:
                out = list(map(operator.add, out, self.col[s][i]))
            elif x:
                out = list(map(operator.add, out, map(operator.mul, self.col[s][i], itertools.repeat(x))))
        if key is not None:
            self._den[key] = out
        return out


def point(tA, tB, name):
    """(差, 分子A, 分母A, 分子B, 分母B)——全部資料。"""
    tot = []
    for t in (tA, tB):
        n = d = 0
        for s in STRATA[CELLS[name][0]]:
            a, b = vals(t, s, name)
            n += sum(a)
            d += sum(b)
        tot += [n, d]
    nA, dA, nB, dB = tot
    return (nA / dA if dA else 0.0) - (nB / dB if dB else 0.0), nA, dA, nB, dB


def ci(boot, tA, tB, name):
    st, _, way = CELLS[name]
    acc = {}
    for arm, t in (('A', tA), ('B', tB)):
        N, D = [0] * boot.n, [0] * boot.n
        for s in STRATA[st]:
            a, b = vals(t, s, name)
            for i, x in enumerate(boot.sums(s, a)):
                N[i] += x
            for i, x in enumerate(boot.sums(s, b, memo=True)):
                D[i] += x
        acc[arm] = (N, D)
    (NA, DA), (NB, DB) = acc['A'], acc['B']
    ds = sorted((NA[i] / DA[i] if DA[i] else 0.0) - (NB[i] / DB[i] if DB[i] else 0.0) for i in range(boot.n))
    pt = point(tA, tB, name)[0]
    lo, hi = ds[int(.025 * boot.n)], ds[int(.975 * boot.n)]
    return {'point': pt, 'lo': lo, 'hi': hi, 'worse': (lo > 0) if way == 'up' else (hi < 0)}


_NUM = re.compile(r'^(-?\d+\.\d+)')


def _consistent(name, c, pt, tA, tB):
    """我的點估計＝gate_cells 印出來的那個（防群集表跟 summarize 對不上）。"""
    m = _NUM.match(str(c['value']))
    if m and not name.startswith('S2'):
        assert abs(float(m.group(1)) - pt) < 6e-5, (name, c['value'], pt)
    elif name.startswith('K3') or name.startswith('W1'):
        xs = [int(x) for x in re.findall(r'\d+', str(c['value']))]
        a, b = (xs[0], xs[1]) if name.startswith('K3') else (xs[0], xs[2])
        _, nA, dA, nB, dB = point(tA, tB, name)
        assert (nA, nB) == (a, b), (name, c['value'], nA, nB)


def apply(v, tA, tB, boot, only_failed=False):
    """v＝gate_cells／e_cells 的格（就地改）：pass_point＝舊判法；ci＝[點, 下, 上]；pass＝pass_point 或 區間沒整段落在變差側。"""
    for name, c in v.items():
        if c.get('kind', 'gate') != 'gate' or name.startswith(SKIP):
            continue
        c['pass_point'] = c['pass']
        if only_failed and c['pass']:
            continue
        r = ci(boot, tA, tB, name)
        _consistent(name, c, r['point'], tA, tB)
        c['ci'] = [round(r['point'], 4), round(r['lo'], 4), round(r['hi'], 4)]
        c['pass'] = bool(c['pass_point'] or not r['worse'])
    return v
