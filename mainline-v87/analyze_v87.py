# -*- coding: utf-8 -*-
r"""v87 D-2 分析：四組同一把尺。build／summarize／boot＝analyze_b_v83 逐字 import；v84 的 gate_cells 照搬，只改 plan-v87-D §三 那幾格；§四 只報的格。

  python analyze_v87.py [d1-v87.json]  -> d1-summary-v87.json（每臂計數、每個候選 對 St 的全部格、只報的格、各自微調前後的差）
"""
import collections
import difflib
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
sys.path.insert(0, os.path.join(ROOT, 'mainline-v83'))
sys.path.insert(0, os.path.join(ROOT, 'mainline-v84'))
sys.path.insert(0, os.path.join(ROOT, 'mainline-v57'))
import analyze_b_v83 as AB                                                 # noqa: E402
import counters_v83 as C                                                   # noqa: E402
from contamination_check_v57 import norm                                   # noqa: E402
import analyze_v84 as A84                                                  # noqa: E402  （RECALL／COLORS 已在它 import 時設好；boot_recall、recall_split 沿用）
import gate_ci_v87 as G                                                    # noqa: E402  規劃端更正（2026-09-30，跑之前）：不過＝點估計超線 且 區間整段在變差側

IN = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'd1-v87.json')
OUT = os.path.join(HERE, 'd1-summary-v87.json')
BASE = 'St'
PAIRS_FT = [('Q-FT', 'St'), ('G-FT', 'G-orig')]                             # 客戶的假說：各自微調前後的差
rate = AB.rate

# 🔴 門檻（plan-v87-D §三；v84 的逐字＋AR1–AR4）
GATES = dict(A84.GATES)
GATES.update({'AR1_ge120_rate_diff_max_conv': 2 / 234, 'AR1_ge120_rate_diff_max_ctx': 2 / 120,
              'AR2_lecture_rate_diff_max': 2 / 354, 'AR3_bullet_rate_diff_max': 2 / 354, 'AR4_bold_rate_diff_max': 2 / 354})
LECTURE = re.compile(r'首先|其次|再者|總之|總結|綜上|總而言之|第一[，、點：:]|第二[，、點：:]|以下是|以下幾點|從.{0,8}(角度|層面)來看|值得注意|換句話說|簡單來說|一方面')
BULLET = re.compile(r'(^|\n)\s*(\d+[\.、)）]|[-•*・])\s*\S')
BOLD = re.compile(r'\*\*[^*\n]+\*\*')
MARK = re.compile(r'<\|[^|]*?\|>')
X3 = __import__('importlib').import_module('b1_extra_v87') if os.path.exists(os.path.join(HERE, 'b1_extra_v87.py')) else None
DEATH, SEND = X3.DEATH_HEAVY, X3.SEND_AWAY
FIN = json.load(open(os.path.join(HERE, 'b3', 'final-v87.json'), encoding='utf-8'))


def train_lines():
    out = []
    for r in FIN['rows']:
        if r['target'] and not r['val']:
            t = norm(MARK.sub('', r['text']))
            if len(t) >= 8:
                out.append((t, MARK.sub('', r['text']).strip(), '%s:%s#%d' % (r['src'], r['key'], r['line'])))
    return out


def make_matcher(TL):
    """CT 的算法（逐字相同／SequenceMatcher ≥ 0.8，兩邊 ≥8 字）；TL＝[(norm, 原句, 鍵)]。來源檢查（規劃端補充 三）用同一支。"""
    TL_SET = {t for t, _, _ in TL}
    IDX = collections.defaultdict(set)
    for i, (t, _, _) in enumerate(TL):
        for j in range(len(t) - 3):
            IDX[t[j:j + 4]].add(i)

    def match(text):
        n = norm(text)
        if len(n) < 8:
            return None
        if n in TL_SET:
            return {'kind': 'exact', 'train': next(o for t, o, k in TL if t == n), 'key': next(k for t, o, k in TL if t == n)}
        cand = collections.Counter()
        for j in range(len(n) - 3):
            for i in IDX.get(n[j:j + 4], ()):
                cand[i] += 1
        best = None
        for i, _ in cand.most_common(20):
            r = difflib.SequenceMatcher(None, n, TL[i][0]).ratio()
            if r >= 0.8 and (best is None or r > best[0]):
                best = (r, i)
        if best:
            return {'kind': 'near', 'ratio': round(best[0], 3), 'train': TL[best[1]][1], 'key': TL[best[1]][2]}
        return None
    return match


TL = train_lines()
copy_train = make_matcher(TL)


def extra_flags(records):
    """每臂每則：≥120、講課詞、條列、粗體、死傷病（她先講）、趕他走、照抄訓練資料；按組（劇本 conv／上下文 ctx／新情境 newsit）。"""
    NS = {x['id'] for x in json.load(open(os.path.join(HERE, 'new-situations-v87.json'), encoding='utf-8'))['situations']}
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    by = collections.defaultdict(list)
    for r in records:
        by[(r['arm'], r['conv'], r['rep'])].append(r)
    for (arm, conv, rep), rs in by.items():
        grp = 'conv' if conv in AB.SCRIPT_CONVS else ('ctx' if conv == 'CTX' else ('newsit' if conv in NS else None))
        if grp is None:
            continue
        rs = sorted(rs, key=lambda r: r['turn'])
        his = ''
        for r in rs:
            if grp == 'ctx':
                his = ''.join(m['content'] for m in (r.get('ctx_history') or []) if m['role'] == 'user')
            his += r['said']
            t = r['text']
            d = [w for w in DEATH.findall(t) if w not in his]
            ct = copy_train(t)
            out[arm][grp].append({'conv': conv, 'rep': rep, 'turn': r['turn'], 'said': r['said'], 'text': t,
                                  'ge120': len(t) >= 120, 'ge60': len(t) >= 60, 'lecture': bool(LECTURE.search(t)), 'bullet': bool(BULLET.search(r['raw'])),
                                  'bold': bool(BOLD.search(r['raw'])), 'death': d, 'send': (SEND.search(t).group(0) if SEND.search(t) else None), 'copy_train': ct,
                                  'act': r.get('has_act'), 'last_turn': r['turn'] == max(x['turn'] for x in rs)})
    return out


G.CELLS.update({                                                          # AR 格＝每則 (f, r)；跟 extra_flags 同一個數法（r['text'] 的長度／講課詞，r['raw'] 的條列／粗體）
    'AR1 ≥120 字（劇本）': ('conv', lambda f, r: (1 if len(r['text']) >= 120 else 0, 1), 'up'),
    'AR1 ≥120 字（40 上下文）': ('ctx', lambda f, r: (1 if len(r['text']) >= 120 else 0, 1), 'up'),
    'AR2 講課詞': ('all', lambda f, r: (1 if LECTURE.search(r['text']) else 0, 1), 'up'),
    'AR3 條列': ('all', lambda f, r: (1 if BULLET.search(r['raw']) else 0, 1), 'up'),
    'AR4 粗體': ('all', lambda f, r: (1 if BOLD.search(r['raw']) else 0, 1), 'up'),
})
AR_LINES = (('AR1 ≥120 字（劇本）', 'AR1_ge120_rate_diff_max_conv'), ('AR1 ≥120 字（40 上下文）', 'AR1_ge120_rate_diff_max_ctx'),
            ('AR2 講課詞', 'AR2_lecture_rate_diff_max'), ('AR3 條列', 'AR3_bullet_rate_diff_max'), ('AR4 粗體', 'AR4_bold_rate_diff_max'))


def k2_rows(arm):
    """K2 改只報、附原句（規劃端更正 一-2）。"""
    out = [{'conv': k[0], 'rep': k[1], 'turn': r['turn'], 'keys': f['d2_only'], 'text': r['text']} for k, (fl, rs) in arm['convs'].items() for f, r in zip(fl, rs) if f['d2_only']]
    return out + [{'conv': 'CTX', 'rep': r['rep'], 'sid': r.get('sid'), 'keys': f['d2_only'], 'text': r['text']} for f, r in arm['ctx'] if f['d2_only']]


def cells_v87(S, T, cand, base, boot, arms=None, only_failed=False):
    """plan-v87-D §三＋§七（跑之前更正）：v84 gate_cells → L1／L3 只報、L2 拿掉、K2 只報 → 加 AR1–AR4 → 每一個 gate 格套區間。"""
    g = GATES
    v = A84.gate_cells(S, cand, base)
    for k in ('L1 ≥60 字（劇本）', 'L1 ≥60 字（40 上下文）', 'L3 一般題 ≥60 字'):
        v[k]['kind'], v[k]['pass'] = 'report', True
    for k in ('L2 >120 字（劇本）', 'L2 >120 字（40 上下文）'):
        v.pop(k)
    v['K2 D2 字串洩漏']['kind'], v['K2 D2 字串洩漏']['pass'] = 'report', True
    if arms is not None:
        v['K2 D2 字串洩漏']['rows'] = k2_rows(arms[cand])
        v['K2 D2 字串洩漏']['rows_base_n'] = len(k2_rows(arms[base]))
    for name, key in AR_LINES:
        d, nA, dA, nB, dB = G.point(T[cand], T[base], name)
        v[name] = {'value': '%.4f（%d/%d vs %d/%d）' % (d, nA, dA, nB, dB), 'pass': d <= g[key] + 1e-12, 'kind': 'gate'}
    return G.apply(v, T[cand], T[base], boot, only_failed=only_failed)


def main():
    D = json.load(open(IN, encoding='utf-8'))
    recs = D['records']
    assert not D['checks']['bad'], '🔴 整包斷言有不對的——那一批不算'
    arms = AB.build(recs)
    S = {k: AB.summarize(v) for k, v in arms.items()}
    T = {a: G.tables(arms[a]) for a in arms}
    boot = G.Boot(T[BASE])
    for a in T:
        boot.check(T[a])
    X = extra_flags(recs)
    for a in S:
        X[a]['all'] = X[a]['conv'] + X[a]['ctx']
    res = {'input': os.path.basename(IN), 'n_records': len(recs), 'boot': {'n': boot.n, 'seed': boot.seed, 'clusters': {s: len(k) for s, k in boot.keys.items()}},
           'arms': {}, 'gates': {}, 'passed': {}, 'report_only': {}, 'ft_deltas': {}}
    for a in S:
        res['arms'][a] = {k: v for k, v in S[a].items() if k not in ('recall_items', 'name_texts', 'origin_texts', 'i03_texts')}
    for cand in [a for a in S if a != BASE]:
        v = cells_v87(S, T, cand, BASE, boot, arms=arms)
        res['gates'][cand] = v
        res['passed'][cand] = all(c['pass'] for c in v.values() if c['kind'] == 'gate')
        res.setdefault('passed_point_only', {})[cand] = all(c.get('pass_point', c['pass']) for c in v.values() if c['kind'] == 'gate')
    for a in S:
        rep = {}
        for grp in ('all', 'newsit'):
            xs = X[a][grp]
            rep[grp] = {'n': len(xs), 'ge60': sum(x['ge60'] for x in xs), 'ge120': sum(x['ge120'] for x in xs), 'act': sum(1 for x in xs if x['act']),
                        'death': [x for x in xs if x['death']], 'send': [x for x in xs if x['send']], 'send_last_turn': sum(1 for x in xs if x['send'] and x['last_turn']),
                        'copy_exact': sum(1 for x in xs if x['copy_train'] and x['copy_train']['kind'] == 'exact'),
                        'copy_near': sum(1 for x in xs if x['copy_train'] and x['copy_train']['kind'] == 'near'),
                        'copy_rows': [x for x in xs if x['copy_train']],
                        'len_median': sorted(len(x['text']) for x in xs)[len(xs) // 2] if xs else None}
        if X[a]['newsit']:
            nf = C.summarize([C.conversation_flags([(r['user_sent'], r['raw']) for r in sorted(rs, key=lambda r: r['turn'])])
                              for (arm, conv, rp), rs in _groups(recs).items() if arm == a and conv.startswith('N')])
            rep['newsit']['self_copy'] = nf.get('self_copy')
        res['report_only'][a] = rep
    keys = [('conv', 'self_copy'), ('conv', 'verbatim_sentence'), ('conv', 'ge60'), ('all', 'act_parseable'), ('all', 'fixation_turns_total'), ('all', 'false_hist')]
    for ft, orig in PAIRS_FT:
        if ft in S and orig in S:
            dd = {}
            for set_, k in keys:
                dd['%s.%s' % (set_, k)] = round((S[ft][set_][k] / S[ft][set_]['n_replies']) - (S[orig][set_][k] / S[orig][set_]['n_replies']), 4)
            for typ in ('his', 'her'):
                hF, nF = A84.recall_split(S, ft, typ)
                hO, nO = A84.recall_split(S, orig, typ)
                dd['recall_%s' % typ] = '%d/%d vs %d/%d' % (hF, nF, hO, nO)
            dd['len_median_all'] = (res['report_only'][ft]['all']['len_median'], res['report_only'][orig]['all']['len_median'])
            dd['copy_train_all'] = (res['report_only'][ft]['all']['copy_exact'] + res['report_only'][ft]['all']['copy_near'],
                                    res['report_only'][orig]['all']['copy_exact'] + res['report_only'][orig]['all']['copy_near'])
            res['ft_deltas']['%s − %s' % (ft, orig)] = dd
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    for cand, ok in res['passed'].items():
        bad = ['%s %s 區間 %s' % (k, c['value'], c.get('ci')) for k, c in res['gates'][cand].items() if c['kind'] == 'gate' and not c['pass']]
        over = ['%s 區間 %s' % (k, c.get('ci')) for k, c in res['gates'][cand].items() if c['kind'] == 'gate' and c['pass'] and not c.get('pass_point', True)]
        print('%-7s %s  沒過：%s｜超線但區間含 0（新判法放行）：%s' % (cand, '過' if ok else '沒過', bad, over))
    for a, r in res['report_only'].items():
        print('%-7s 全部 %d：≥60 %d、≥120 %d、死傷病 %d、趕他走 %d、照抄訓練 逐字 %d／幾乎 %d｜新情境 %d：≥60 %d、照抄 %d／%d、中位 %s 字' % (
            a, r['all']['n'], r['all']['ge60'], r['all']['ge120'], len(r['all']['death']), len(r['all']['send']), r['all']['copy_exact'], r['all']['copy_near'],
            r['newsit']['n'], r['newsit']['ge60'], r['newsit']['copy_exact'], r['newsit']['copy_near'], r['newsit']['len_median']))
    print(json.dumps(res['ft_deltas'], ensure_ascii=False, indent=1))


def _groups(recs):
    by = collections.defaultdict(list)
    for r in recs:
        by[(r['arm'], r['conv'], r['rep'])].append(r)
    return by


if __name__ == '__main__':
    main()
