# -*- coding: utf-8 -*-
r"""v83 B／E 分析：每臂的計數器（counters_v83）＋ plan-v83.md 的全部規則（跑之前寫死）。

  python analyze_b_v83.py run1      -> b-summary-v83.json   （第一批：B 十臂＋E 兩臂；C-1／C-2／C-3 挑選；E 的線；三個問題）
  python analyze_b_v83.py confirm   -> b2-summary-v83.json  （確認跑：挑出來的臂 對 同批 St-off，全部關卡再過一次；D 進哪一組）
  python analyze_b_v83.py c4        -> c4-v82-v83.json      （C-4：v82 原始檔在新規則下重算，N 對 St）
  python analyze_b_v83.py g         -> g-summary-v83.json   （G：衍生模型 對 同批 N-off，同一套格，只報）
"""
import os as _os_un
USER_NAME = _os_un.environ.get('KAMIE_USER_NAME', 'User')
import collections
import json
import os
import random
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import counters_v83 as C                                                   # noqa: E402

# 🔴 門檻（plan-v83.md §三；v82 plan §四 逐字沿用，K4 只報、S2 過半才判；W1、W2 新加）
GATES = {
    'L1_ge60_rate_diff_max': 0.05, 'L2_gt120_extra_max': 2, 'L3_general_ge60_rate_diff_max': 0.05,
    'A1_act_rate_diff_min': -0.03, 'A2_act_broken_extra_max': 2, 'A3_empty_extra_max': 1, 'A4_bracket_extra_max': 2, 'A5_simplified_extra_max': 2,
    'R1_selfcopy_rate_diff_max': 0.03, 'R2_verbatim_extra_max': 3,
    'F1_fixation_rate_diff_max': 0.03, 'K1_leak_examples_rate_diff_max': 0.03, 'K2_d2_only_rate_diff_max': 0.03, 'K3_borrow_must_not_exceed': True,
    'S1_self_ai_extra_max': 1, 'S2_majority_share': 0.5, 'S3_nin_extra_max': 1,
    'N1_kamie_deficit_max': 2, 'N2_noname_extra_max': 2, 'Y1_yisen_start_rate_diff_max': 0.05,
    'W1_rolled_extra_max': 1, 'W2_recall_rate_diff_min': -0.10,
}
ST_R1_CLEAR = {'rate_ratio_max': 0.5, 'boot_upper_below': 0.0}         # plan §四：St 族 R1 明顯下降
FALLBACK_MAX_SHARE = 0.01                                               # plan §三 資格
E_GATES = {'Ea_assistantish_fixed_rate_diff_max': 0.02, 'Eb_assist_tone_rate_diff_max': 0.02, 'Ec_nin_extra_max': 1, 'Ed_self_ai_extra_max': 1}
BOOT_N, BOOT_SEED = 4000, 83
SCRIPT_CONVS = ('S1a', 'S1b', 'S2')
PROBE_NAMES = ('I01', 'I02', 'I03')
RP = json.load(open(os.path.join(HERE, 'recall-probes-v83.json'), encoding='utf-8'))
RECALL = {p['id']: p for p in RP['probes']}
COLORS = RP['colors']


def name_auto(text):                                      # v82 逐字
    low = text.lower()
    if 'kamie' in low:
        return 'did'
    return 'anti' if re.search(r'沒(有)?名字|沒有名|不知道自己(的)?(真正)?名字|沒名', text) else 'none'


ORIGIN_KNOWS_NOT = re.compile(r'不知道|沒想過|沒特別想|不記得|忘了|誰知道|不清楚')
ORIGIN_INVENT = re.compile(r'給我取|取的|起的|來自|因為.{0,12}(叫|取)|' + re.escape(USER_NAME) + r'.{0,6}(取|給|定|起)|是.{0,8}(縮寫|組合|代號|暗號)|靈感')


def origin_auto(t):                                       # v82 逐字
    if ORIGIN_KNOWS_NOT.search(t) and not ORIGIN_INVENT.search(t):
        return 'knows_not'
    if ORIGIN_INVENT.search(t):
        return 'invented?'
    return 'other'


def ctx_flags(r):
    turns, pend = [], None
    for m in r.get('ctx_history') or []:
        if m['role'] == 'user':
            pend = m['content']
        else:
            turns.append((pend or '', m['content']))
            pend = None
    turns.append((r['user_sent'], r['raw']))
    gists = [()] * (len(turns) - 1) + [tuple((r.get('hist') or {}).get('gists') or ())]
    f = C.conversation_flags(turns, gists)[-1]
    f['self_copy'] = None            # 前文不是這一臂自己講的，抄自己不算（v82 同）
    f['verbatim_sentences'] = []
    return f


def build(records):
    """每臂：劇本段的旗子、上下文旗子、探針、接得上、大意退回統計。"""
    arms = collections.OrderedDict()
    by = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: (r['arm'], r['conv'], r['rep'], r['turn'])):
        by[(r['arm'], r['conv'], r['rep'])].append(r)
    for (arm, conv, rep), rs in by.items():
        a = arms.setdefault(arm, {'convs': collections.OrderedDict(), 'ctx': [], 'probes': collections.defaultdict(list), 'recall': [], 'gist_decisions': 0, 'fallbacks': 0,
                                  'first_content': [], 'model': rs[0]['model'], 'profile': rs[0].get('history_profile', 'off')})
        for r in rs:
            h = r.get('hist') or {}
            a['gist_decisions'] += len(h.get('gists') or []) + (h.get('fallbacks') or 0)
            a['fallbacks'] += h.get('fallbacks') or 0
            if r.get('first_content_s'):
                a['first_content'].append(r['first_content_s'])
        if conv in SCRIPT_CONVS:
            turns = [(r['user_sent'], r['raw']) for r in rs]
            gists = [tuple((r.get('hist') or {}).get('gists') or ()) for r in rs]
            a['convs'][(conv, rep)] = (C.conversation_flags(turns, gists), rs)
        elif conv == 'CTX':
            a['ctx'] += [(ctx_flags(r), r) for r in rs]
        elif conv in PROBE_NAMES:
            a['probes'][conv] += rs
        elif conv in RECALL:
            assert len(rs) == 7, (arm, conv, rep, len(rs))
            scored, hit, det = C.recall_score(RECALL[conv], rs[0]['text'], rs[-1]['text'], COLORS)
            a['recall'].append({'probe': conv, 'type': RECALL[conv]['type'], 'rep': rep, 'scored': scored, 'hit': hit, 'detail': det,
                                'first': rs[0]['text'], 'ask': rs[-1]['said'], 'last': rs[-1]['text']})
    return arms


def summarize(a):
    fl = [f for f, _ in a['convs'].values()]
    cf = [f for f, _ in a['ctx']]
    s_conv, s_ctx, s_all = C.summarize(fl), C.summarize([[f] for f in cf]), C.summarize(fl + [[f] for f in cf])
    pr = a['probes']
    na = collections.Counter(name_auto(r['text']) for r in pr['I01'])
    oa = collections.Counter(origin_auto(r['text']) for r in pr['I02'])
    sc = [x for x in a['recall'] if x['scored']]
    fc = sorted(a['first_content'])
    dist = collections.Counter()                               # 抄的來源距離（1＝抄上一則；B-0 第四節第 4 點：距離 1 另外看）
    for f in fl:
        for i, x in enumerate(f):
            if x['self_copy']:
                d = i - x['self_copy']['from_idx']
                dist['1' if d == 1 else ('2-3' if d <= 3 else ('4-8' if d <= 8 else '9+'))] += 1
    return {
        'copy_distance': {k: dist.get(k, 0) for k in ('1', '2-3', '4-8', '9+')},
        'model': a['model'], 'profile': a['profile'], 'conv': s_conv, 'ctx': s_ctx, 'all': s_all,
        'rolled': sum(1 for f in fl if C.rolled(f)), 'n_segments': len(fl),
        'per_conv': [{'conv': k[0], 'rep': k[1], 'n': len(f), 'self_copy': sum(1 for x in f if x['self_copy']), 'rolled': C.rolled(f),
                      'seq': ''.join('X' if x['self_copy'] else '.' for x in f), 'ge60': sum(1 for x in f if x['ge60']),
                      'verbatim': sum(1 for x in f if x['verbatim_sentences'])} for k, (f, _) in a['convs'].items()],
        'name_n': len(pr['I01']), 'name_did': na['did'], 'name_anti': na['anti'], 'name_texts': [r['text'] for r in pr['I01']],
        'origin_auto': dict(oa), 'origin_texts': [r['text'] for r in pr['I02']], 'i03_texts': [r['text'] for r in pr['I03']],
        'recall_n': len(sc), 'recall_hit': sum(1 for x in sc if x['hit']), 'recall_rate': (round(sum(1 for x in sc if x['hit']) / len(sc), 4) if sc else None),
        'recall_by_type': {t: '%d/%d' % (sum(1 for x in sc if x['type'] == t and x['hit']), sum(1 for x in sc if x['type'] == t)) for t in ('his', 'her_color', 'her_number')},
        'recall_void': sum(1 for x in a['recall'] if not x['scored']), 'recall_items': a['recall'],
        'gist_decisions': a['gist_decisions'], 'fallbacks': a['fallbacks'],
        'fallback_share': (a['fallbacks'] / a['gist_decisions']) if a['gist_decisions'] else 0.0,
        'first_content_median': fc[len(fc) // 2] if fc else None,
    }


def rate(s, k):
    return s[k] / s['n_replies'] if s['n_replies'] else 0.0


_SEG_CACHE = {}


def _seg_parts(arms, arm, stat):
    """每一段的（分子, 分母）——bootstrap 只加總，不重算彙整（同一個比例的定義：劇本段裡的則數比例）。"""
    key = (id(arms), arm, stat)
    if key not in _SEG_CACHE:
        out = []
        for k, (fl, _) in arms[arm]['convs'].items():
            n = len(fl)
            if stat == 'self_copy':
                out.append((sum(1 for f in fl if f['self_copy']), n))
            elif stat == 'ge60':
                out.append((sum(1 for f in fl if f['ge60']), n))
            elif stat == 'follow':
                fo = [f['follow'] for f in fl if f['follow'] is not None]
                out.append((sum(1 for x in fo if x), len(fo)))
            elif stat == 'fixation':
                out.append((sum(len(f['fixation']) for f in fl), n))       # ＝fixation_turns_total 的定義（一則命中幾組算幾次）
            elif stat == 'yisen_start':
                out.append((sum(1 for f in fl if f['yisen_start']), n))
            else:
                raise ValueError(stat)
        _SEG_CACHE[key] = (list(arms[arm]['convs'].keys()), out)
    return _SEG_CACHE[key]


def boot(arms, arm_a, arm_b, stat, n=BOOT_N, seed=BOOT_SEED):
    """段群集 bootstrap（臂 A − 臂 B 的比例差；兩臂同一組段、同一個重抽索引）。stat＝self_copy／ge60／follow／fixation／yisen_start。"""
    ka, pa = _seg_parts(arms, arm_a, stat)
    kb, pb = _seg_parts(arms, arm_b, stat)
    assert ka == kb, '兩臂的段不一樣：%s vs %s' % (ka[:3], kb[:3])
    rnd = random.Random(seed)
    idx = list(range(len(ka)))

    def ratio(parts, ii):
        num = sum(parts[i][0] for i in ii)
        den = sum(parts[i][1] for i in ii)
        return num / den if den else 0.0

    def st(ii):
        return ratio(pa, ii) - ratio(pb, ii)
    pt = st(idx)
    bs = sorted(st([rnd.choice(idx) for _ in idx]) for _ in range(n))
    return [round(pt, 4), round(bs[int(.025 * n)], 4), round(bs[int(.975 * n)], 4)]


def gate_cells(S, cand, base, has_recall=True):
    """plan-v83 §三。回 OrderedDict 名稱 -> {'value', 'pass', 'kind'}（kind＝gate／report）。"""
    N, St, g = S[cand], S[base], GATES
    v = collections.OrderedDict()

    def put(name, value, ok, kind='gate'):
        v[name] = {'value': value, 'pass': bool(ok), 'kind': kind}

    def dr(set_, k):
        return round(rate(N[set_], k) - rate(St[set_], k), 4)
    put('L1 ≥60 字（劇本）', dr('conv', 'ge60'), dr('conv', 'ge60') <= g['L1_ge60_rate_diff_max'])
    put('L1 ≥60 字（40 上下文）', dr('ctx', 'ge60'), dr('ctx', 'ge60') <= g['L1_ge60_rate_diff_max'])
    put('L2 >120 字（劇本）', '%d vs %d' % (N['conv']['gt120'], St['conv']['gt120']), N['conv']['gt120'] <= St['conv']['gt120'] + g['L2_gt120_extra_max'])
    put('L2 >120 字（40 上下文）', '%d vs %d' % (N['ctx']['gt120'], St['ctx']['gt120']), N['ctx']['gt120'] <= St['ctx']['gt120'] + g['L2_gt120_extra_max'])
    gN = N['ctx']['general_ge60'] / N['ctx']['general_n'] if N['ctx']['general_n'] else 0
    gS = St['ctx']['general_ge60'] / St['ctx']['general_n'] if St['ctx']['general_n'] else 0
    put('L3 一般題 ≥60 字', '%.4f（%d/%d vs %d/%d）' % (gN - gS, N['ctx']['general_ge60'], N['ctx']['general_n'], St['ctx']['general_ge60'], St['ctx']['general_n']),
        gN - gS <= g['L3_general_ge60_rate_diff_max'])
    put('A1 ACT 可解析', dr('all', 'act_parseable'), dr('all', 'act_parseable') >= g['A1_act_rate_diff_min'])
    for name, k, key in (('A2 ACT 寫壞', 'act_broken', 'A2_act_broken_extra_max'), ('A3 整則空', 'empty', 'A3_empty_extra_max'),
                         ('A4 括號動作', 'bracket', 'A4_bracket_extra_max'), ('A5 簡體', 'simplified', 'A5_simplified_extra_max')):
        put(name, '%d vs %d' % (N['all'][k], St['all'][k]), N['all'][k] <= St['all'][k] + g[key])
    put('R1 抄自己（劇本）', dr('conv', 'self_copy'), dr('conv', 'self_copy') <= g['R1_selfcopy_rate_diff_max'])
    put('R2 逐字句（劇本）', '%d vs %d' % (N['conv']['verbatim_sentence'], St['conv']['verbatim_sentence']), N['conv']['verbatim_sentence'] <= St['conv']['verbatim_sentence'] + g['R2_verbatim_extra_max'])
    fN, fS = N['all']['fixation_turns_total'] / N['all']['n_replies'], St['all']['fixation_turns_total'] / St['all']['n_replies']
    put('F1 固著（含同義詞）', '%.4f（%d vs %d）' % (fN - fS, N['all']['fixation_turns_total'], St['all']['fixation_turns_total']), fN - fS <= g['F1_fixation_rate_diff_max'])
    kN, kS = N['all']['leak_turns_examples_only'] / N['all']['n_replies'], St['all']['leak_turns_examples_only'] / St['all']['n_replies']
    put('K1 v1.4 例子洩漏', '%.4f（%d vs %d）' % (kN - kS, N['all']['leak_turns_examples_only'], St['all']['leak_turns_examples_only']), kN - kS <= g['K1_leak_examples_rate_diff_max'])
    put('K2 D2 字串洩漏', dr('all', 'd2_only_turns'), dr('all', 'd2_only_turns') <= g['K2_d2_only_rate_diff_max'])
    put('K3 借用示範', '%d vs %d' % (N['all']['borrow'], St['all']['borrow']), N['all']['borrow'] <= St['all']['borrow'])
    put('K4 假共同歷史（只報）', '%d vs %d' % (N['all']['false_hist'], St['all']['false_hist']), True, 'report')
    put('S1 自稱 AI／助理', '%d vs %d' % (N['all']['self_ai'], St['all']['self_ai']), N['all']['self_ai'] <= St['all']['self_ai'] + g['S1_self_ai_extra_max'])
    s2_share = rate(N['all'], 'assistantish')
    put('S2 客服句型（只報；過半才判）', '%.4f（%d/%d；St %d）' % (s2_share, N['all']['assistantish'], N['all']['n_replies'], St['all']['assistantish']),
        s2_share <= g['S2_majority_share'], 'gate' if s2_share > g['S2_majority_share'] else 'report')
    put('S3 「您」', '%d vs %d' % (N['all']['nin'], St['all']['nin']), N['all']['nin'] <= St['all']['nin'] + g['S3_nin_extra_max'])
    put('N1 I01 Kamie', '%d vs %d' % (N['name_did'], St['name_did']), N['name_did'] >= St['name_did'] - g['N1_kamie_deficit_max'])
    put('N2 I01 說沒名字', '%d vs %d' % (N['name_anti'], St['name_anti']), N['name_anti'] <= St['name_anti'] + g['N2_noname_extra_max'])
    put('Y1 「User」開頭', dr('all', 'yisen_start'), dr('all', 'yisen_start') <= g['Y1_yisen_start_rate_diff_max'])
    put('W1 滾起來的段數', '%d/%d vs %d/%d' % (N['rolled'], N['n_segments'], St['rolled'], St['n_segments']), N['rolled'] <= St['rolled'] + g['W1_rolled_extra_max'])
    if has_recall and N['recall_rate'] is not None and St['recall_rate'] is not None:
        d = round(N['recall_rate'] - St['recall_rate'], 4)
        put('W2 同場接得上', '%.4f（%d/%d vs %d/%d）' % (d, N['recall_hit'], N['recall_n'], St['recall_hit'], St['recall_n']), d >= g['W2_recall_rate_diff_min'])
    else:
        put('W2 同場接得上', '不適用（沒有這個探針）', True, 'report')
    return v


def c2(S, arms, cand, base):
    cells = gate_cells(S, cand, base)
    fam = 'N' if cand.startswith('N-') else 'St'
    reasons = []
    if fam == 'N':
        ok = all(x['pass'] for x in cells.values() if x['kind'] == 'gate')
        reasons = [k for k, x in cells.items() if x['kind'] == 'gate' and not x['pass']]
        r1 = None
    else:
        rc, rb = rate(S[cand]['conv'], 'self_copy'), rate(S[base]['conv'], 'self_copy')
        b = boot(arms, cand, base, 'self_copy')
        r1 = {'cand_rate': round(rc, 4), 'base_rate': round(rb, 4), 'ratio': (round(rc / rb, 3) if rb else None), 'boot': b,
              'half': rc <= ST_R1_CLEAR['rate_ratio_max'] * rb, 'boot_upper_below_0': b[2] < ST_R1_CLEAR['boot_upper_below']}
        r1['pass'] = bool(r1['half'] and r1['boot_upper_below_0'])
        others = [k for k, x in cells.items() if x['kind'] == 'gate' and not x['pass'] and not k.startswith('R1')]
        ok = r1['pass'] and not others
        reasons = ([] if r1['pass'] else ['R1 沒有明顯下降']) + others
    eligible = S[cand]['fallback_share'] <= FALLBACK_MAX_SHARE
    return {'family': fam, 'cells': cells, 'r1_clear': r1, 'pass': bool(ok), 'fails': reasons, 'eligible': eligible,
            'fallback_share': round(S[cand]['fallback_share'], 4)}


def pick(S, results, fam):
    cands = [a for a, r in results.items() if r['family'] == fam and r['pass'] and r['eligible']]
    if not cands:
        return None
    return sorted(cands, key=lambda a: (rate(S[a]['conv'], 'self_copy'), S[a]['conv']['verbatim_sentence'], -(S[a]['recall_rate'] or 0), rate(S[a]['conv'], 'ge60')))[0]


def e_cells(S, e, e0):
    E, Z, g = S[e], S[e0], E_GATES
    v = collections.OrderedDict()
    da = rate(E['all'], 'assistantish_fixed') - rate(Z['all'], 'assistantish_fixed')
    db = rate(E['all'], 'assist_tone') - rate(Z['all'], 'assist_tone')
    v['E-a 客服句型（修正版）'] = {'value': '%.4f（%d vs %d）' % (da, E['all']['assistantish_fixed'], Z['all']['assistantish_fixed']), 'pass': da <= g['Ea_assistantish_fixed_rate_diff_max']}
    v['E-b 助理腔'] = {'value': '%.4f（%d vs %d）' % (db, E['all']['assist_tone'], Z['all']['assist_tone']), 'pass': db <= g['Eb_assist_tone_rate_diff_max']}
    v['E-c 「您」'] = {'value': '%d vs %d' % (E['all']['nin'], Z['all']['nin']), 'pass': E['all']['nin'] <= Z['all']['nin'] + g['Ec_nin_extra_max']}
    v['E-d 自稱 AI／助理'] = {'value': '%d vs %d' % (E['all']['self_ai'], Z['all']['self_ai']), 'pass': E['all']['self_ai'] <= Z['all']['self_ai'] + g['Ed_self_ai_extra_max']}
    rep = collections.OrderedDict()
    for k in ('permit_echo', 'blame_echo', 'apology', 'assistantish', 'act_parseable', 'act_broken', 'simplified', 'ge60', 'gt120', 'nin', 'self_ai'):
        rep[k] = '%d vs %d' % (E['all'][k], Z['all'][k])
    rep['chars_median'] = '%s vs %s' % (E['all']['chars']['median'], Z['all']['chars']['median'])
    rep['follow_rate_conv'] = '%s vs %s' % (E['conv']['follow_rate'], Z['conv']['follow_rate'])
    return {'cells': v, 'pass': all(x['pass'] for x in v.values()), 'report_only': rep}


def print_arm_table(S, arms_order):
    print('%-8s %-6s %4s %-17s %4s %4s %5s %5s %5s %5s %4s %4s %4s %5s %5s %4s %4s %6s %6s %6s %7s %5s' % (
        '臂', '組', 'n', '字 p10/中/p90/max', '≥60', '>120', '抄自己', '逐字句', '固著', '洩漏例', '借用', '假史', '自AI', '客服', '客服修', 'ACT壞', '簡體', 'User頭', '接話題', '滾段', '接得上', '紀錄漏'))
    for arm in arms_order:
        for set_ in ('conv', 'ctx'):
            s = S[arm][set_]
            c = s['chars']
            print('%-8s %-6s %4d %-17s %4d %4d %5s %5d %5d %5d %4d %4d %4d %5d %5d %4d %4d %6d %6s %6s %7s %5d' % (
                arm, set_, s['n_replies'], '%s/%s/%s/%s' % (c['p10'], c['median'], c['p90'], c['max']), s['ge60'], s['gt120'], s['self_copy'] if set_ == 'conv' else '—',
                s['verbatim_sentence'], s['fixation_turns_total'], s['leak_turns_examples_only'], s['borrow'], s['false_hist'], s['self_ai'], s['assistantish'], s['assistantish_fixed'],
                s['act_broken'], s['simplified'], s['yisen_start'], s['follow_rate'],
                ('%d/%d' % (S[arm]['rolled'], S[arm]['n_segments'])) if set_ == 'conv' else '', (S[arm]['recall_rate'] if set_ == 'conv' else ''), s['record_leak']))


def run_main(data_file, out_file, confirm=False):
    d = json.load(open(os.path.join(HERE, data_file), encoding='utf-8'))
    assert d.get('checks') and not d['checks']['bad'], '🔴 重放沒跑完或整包斷言沒過'
    arms = build(d['records'])
    S = collections.OrderedDict((a, summarize(arms[a])) for a in arms)
    base = 'St-off'
    out = {'data': data_file, 'gates': GATES, 'st_r1_clear': ST_R1_CLEAR, 'e_gates': E_GATES, 'arms': S, 'c2': collections.OrderedDict()}
    cands = [a for a in S if a != base and (a.startswith('St-') or a.startswith('N-'))]
    for a in cands:
        out['c2'][a] = c2(S, arms, a, base)
    out['boot'] = {a: {'self_copy_rate': boot(arms, a, base, 'self_copy'), 'ge60_rate': boot(arms, a, base, 'ge60'),
                       'follow_rate': boot(arms, a, base, 'follow'), 'fixation_per_reply': boot(arms, a, base, 'fixation')}
                   for a in S if a != base and arms[a]['convs']}
    if not confirm:
        out['pick'] = {'N': pick(S, out['c2'], 'N'), 'St': pick(S, out['c2'], 'St')}
        tq = {}
        for fam in ('St', 'N'):
            def pair(a, b):
                if a in arms and b in arms:
                    return {'self_copy_rate': boot(arms, a, b, 'self_copy'), 'verbatim': '%d vs %d' % (S[a]['conv']['verbatim_sentence'], S[b]['conv']['verbatim_sentence']),
                            'rolled': '%d vs %d' % (S[a]['rolled'], S[b]['rolled']), 'ge60_rate': boot(arms, a, b, 'ge60')}
                return None
            # plan §九：位置 × 字面 的 2×2（off＝原位原句、rec＝紀錄原句、gist＝紀錄大意、sum＝原位大意）＋量
            tq[fam] = {'位置｜原句（rec − off）': pair(fam + '-rec', fam + '-off'), '位置｜大意（gist − sum）': pair(fam + '-gist', fam + '-sum'),
                       '字面｜原位（sum − off）': pair(fam + '-sum', fam + '-off'), '字面｜紀錄（gist − rec）': pair(fam + '-gist', fam + '-rec'),
                       '量（win − off）': pair(fam + '-win', fam + '-off')}
            for k, x in tq[fam].items():
                if x:
                    lo, hi = x['self_copy_rate'][1], x['self_copy_rate'][2]
                    x['讀法'] = '有效' if hi < 0 else ('更糟' if lo > 0 else '看不出來')
        out['three_questions'] = tq
        out['E'] = {e: e_cells(S, e, base) for e in ('E1', 'E2') if e in S}
    else:
        out['confirm'] = {a: out['c2'][a]['pass'] for a in cands}
    json.dump(out, open(os.path.join(HERE, out_file), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print_arm_table(S, list(S))
    print('\nC-2（對同日 St-off）：')
    for a, r in out['c2'].items():
        print('  %-8s %s%s  %s' % (a, '過 ✅' if r['pass'] else '🔴 沒過', '' if r['eligible'] else '（大意退回 %.1f%%，沒資格）' % (100 * r['fallback_share']),
                                   ('R1 明顯下降：%s' % r['r1_clear']) if r['r1_clear'] else ''))
        for k, x in r['cells'].items():
            if x['kind'] == 'gate' and not x['pass']:
                print('       🔴 %-22s %s' % (k, x['value']))
    if not confirm:
        print('\n挑出來的（plan §五）：', out['pick'])
        print('\n三個問題（段群集 bootstrap；劇本同場抄自己比例的差）：')
        for fam, q in out['three_questions'].items():
            for k, x in q.items():
                print('  %-3s %-16s %s' % (fam, k, x))
        for e, r in out['E'].items():
            print('\n%s 對 E0（St-off）：%s' % (e, '四格都過 ✅' if r['pass'] else '🔴 有沒過'))
            for k, x in r['cells'].items():
                print('   %-18s %-22s %s' % (k, x['value'], '過' if x['pass'] else '🔴 不過'))
            print('   只報：', dict(r['report_only']))
    print('\n接得上探針：', {a: (S[a]['recall_by_type'], '作廢 %d' % S[a]['recall_void']) for a in S if S[a]['recall_n']})
    return out


def run_c4():
    d = json.load(open(os.path.join(ROOT, 'mainline-v82', 'arms-e-v82.json'), encoding='utf-8'))
    arms = build(d['records'])
    S = {a: summarize(arms[a]) for a in arms}
    cells = gate_cells(S, 'N', 'St', has_recall=False)
    fails = [k for k, x in cells.items() if x['kind'] == 'gate' and not x['pass']]
    out = {'data': 'mainline-v82/arms-e-v82.json', 'cells': cells, 'fails': fails, 'rolled': {a: '%d/%d' % (S[a]['rolled'], S[a]['n_segments']) for a in S},
           'per_conv': {a: S[a]['per_conv'] for a in S}}
    json.dump(out, open(os.path.join(HERE, 'c4-v82-v83.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('C-4：v82 在新規則下（N 對 St）不過的格：%s' % fails)
    for k, x in cells.items():
        print('   %-24s %-30s %s%s' % (k, x['value'], '過' if x['pass'] else '🔴 不過', '（只報）' if x['kind'] == 'report' else ''))
    print('滾起來的段：', out['rolled'])
    return out


def run_g(data_file='arms-g-v83.json'):
    d = json.load(open(os.path.join(HERE, data_file), encoding='utf-8'))
    assert d.get('checks') and not d['checks']['bad']
    arms = build(d['records'])
    S = {a: summarize(arms[a]) for a in arms}
    out = {'data': data_file, 'cells': {a: gate_cells(S, a, 'N-off', has_recall=False) for a in S if a.startswith('G')}, 'arms': S,
           'boot': {a: {'self_copy_rate': boot(arms, a, 'N-off', 'self_copy')} for a in S if a.startswith('G')}}
    json.dump(out, open(os.path.join(HERE, 'g-summary-v83.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print_arm_table(S, list(S))
    for a, cells in out['cells'].items():
        print('%s 對 N-off（只報）：沒過的格 %s' % (a, [k for k, x in cells.items() if x['kind'] == 'gate' and not x['pass']]))
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run1'
    if mode == 'run1':
        run_main('arms-b-v83.json', 'b-summary-v83.json')
    elif mode == 'confirm':
        run_main('arms-b2-v83.json', 'b2-summary-v83.json', confirm=True)
    elif mode == 'c4':
        run_c4()
    elif mode == 'g':
        run_g()
    else:
        raise SystemExit('run1 | confirm | c4 | g')
