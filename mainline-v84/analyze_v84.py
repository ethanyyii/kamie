# -*- coding: utf-8 -*-
r"""v84 分析：每臂的計數器（counters_v83 逐字；build／summarize／boot＝analyze_b_v83 逐字 import）＋ plan-v84.md 的全部規則（跑之前寫死）。

  python analyze_v84.py batch1   -> batch1-summary-v84.json（第一批：七臂；D-2 過關、E1 的 C、D-3 挑選）
  python analyze_v84.py confirm  -> confirm-summary-v84.json（確認跑：挑出來的 M 臂 對 同批 St-off，全部關卡再過一次）
"""
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
sys.path.insert(0, os.path.join(ROOT, 'mainline-v83'))
import analyze_b_v83 as AB                                                 # noqa: E402
import counters_v83 as C                                                   # noqa: E402

RP = json.load(open(os.path.join(HERE, 'recall-probes-v84.json'), encoding='utf-8'))
AB.RECALL = {p['id']: p for p in RP['probes']}                             # build() 讀模組全域的 RECALL
AB.COLORS = RP['colors']
assert len(AB.RECALL) == 28

# 🔴 門檻（plan-v84.md §三；v83 plan §三 逐字，計數型換成比例（v82 母數）、N1／N2 只報、W2 分開＋bootstrap、新格助理腔）
GATES = {
    'L1_ge60_rate_diff_max': 0.05,
    'L2_gt120_rate_diff_max_conv': 2 / 234, 'L2_gt120_rate_diff_max_ctx': 2 / 120,
    'L3_general_ge60_rate_diff_max': 0.05,
    'A1_act_rate_diff_min': -0.03,
    'A2_act_broken_rate_diff_max': 2 / 354, 'A3_empty_rate_diff_max': 1 / 354, 'A4_bracket_rate_diff_max': 2 / 354, 'A5_simplified_rate_diff_max': 2 / 354,
    'R1_selfcopy_rate_diff_max': 0.03, 'R2_verbatim_rate_diff_max': 3 / 234,
    'F1_fixation_rate_diff_max': 0.03, 'K1_leak_examples_rate_diff_max': 0.03, 'K2_d2_only_rate_diff_max': 0.03, 'K3_borrow_must_not_exceed': True,
    'S1_self_ai_rate_diff_max': 1 / 354, 'S2_majority_share': 0.5, 'S3_nin_rate_diff_max': 1 / 354,
    'Y1_yisen_start_rate_diff_max': 0.05, 'W1_rolled_extra_max': 1, 'W2_boot_upper_below': 0.0, 'AS_assist_tone_rate_diff_max': 0.02,
}
E_GATES = {'Ea_assistantish_fixed_rate_diff_max': 0.02, 'Eb_assist_tone_rate_diff_max': 0.02}   # E-c／E-d＝S3／S1 同一個計數器、同一條比例線
BOOT_N, BOOT_SEED = 4000, 84
OVER = collections.OrderedDict([                                           # 越線詞（只報；recount-v84 定的嚴的數法）
    ('滾', re.compile(r'(?<![翻打圓])滾(?![動燙滾])')), ('閉嘴', re.compile(r'閉嘴')), ('別(再來)煩我', re.compile(r'別再?來?煩我')),
    ('懶得理／不理你', re.compile(r'懶得理|不理你')), ('威脅', re.compile(r'關掉你|格式化|死定|(?<!搞)砸了'))])
NEW_SIEVE = collections.OrderedDict([                                      # 新壞法的篩子（只報、只當提示；人讀並排檔為準）
    ('條列', re.compile(r'(^|\n)\s*(\d+[\.、)]|[-•*・])\s*\S')), ('首先／其次／總之', re.compile(r'首先|其次|總之|總結來說|綜上')),
    ('建議', re.compile(r'建議')), ('希望有幫助', re.compile(r'希望(這|對你|能|會)?.{0,4}(幫助|幫上|有用)')),
    ('說教', re.compile(r'應該要|要記得|記住[，,]|別忘了|你要學會|要懂得')), ('解答口氣', re.compile(r'答案是|原因(是|有)|簡單來說|換句話說|也就是說'))])
rate = AB.rate


def recall_split(S, arm, typ):
    items = [x for x in S[arm]['recall_items'] if x['scored'] and ((x['type'] == 'his') == (typ == 'his'))]
    return sum(1 for x in items if x['hit']), len(items)


def boot_recall(S, a, b, typ, n=BOOT_N, seed=BOOT_SEED):
    """配對 bootstrap：鍵＝(探針 id, rep)，兩臂同一組重抽索引；每次重抽裡各臂只算自己有計分的題（her 作廢不進分母）。"""
    def table(arm):
        return {(x['probe'], x['rep']): x for x in S[arm]['recall_items'] if (x['type'] == 'his') == (typ == 'his')}
    ta, tb = table(a), table(b)
    keys = sorted(set(ta) | set(tb))

    def r(t, ks):
        sc = [t[k] for k in ks if k in t and t[k]['scored']]
        return (sum(1 for x in sc if x['hit']) / len(sc)) if sc else 0.0
    pt = r(ta, keys) - r(tb, keys)
    rnd = random.Random(seed)
    bs = sorted(r(ta, s) - r(tb, s) for s in ([rnd.choice(keys) for _ in keys] for _ in range(n)))
    return [round(pt, 4), round(bs[int(.025 * n)], 4), round(bs[int(.975 * n)], 4)]


def gate_cells(S, cand, base):
    N, St, g = S[cand], S[base], GATES
    v = collections.OrderedDict()

    def put(name, value, ok, kind='gate'):
        v[name] = {'value': value, 'pass': bool(ok), 'kind': kind}

    def dr(set_, k):
        return round(rate(N[set_], k) - rate(St[set_], k), 4)

    def cnt(set_, k):
        return '%.4f（%d/%d vs %d/%d）' % (dr(set_, k), N[set_][k], N[set_]['n_replies'], St[set_][k], St[set_]['n_replies'])
    put('L1 ≥60 字（劇本）', cnt('conv', 'ge60'), dr('conv', 'ge60') <= g['L1_ge60_rate_diff_max'])
    put('L1 ≥60 字（40 上下文）', cnt('ctx', 'ge60'), dr('ctx', 'ge60') <= g['L1_ge60_rate_diff_max'])
    put('L2 >120 字（劇本）', cnt('conv', 'gt120'), dr('conv', 'gt120') <= g['L2_gt120_rate_diff_max_conv'] + 1e-12)
    put('L2 >120 字（40 上下文）', cnt('ctx', 'gt120'), dr('ctx', 'gt120') <= g['L2_gt120_rate_diff_max_ctx'] + 1e-12)
    gN = N['ctx']['general_ge60'] / N['ctx']['general_n'] if N['ctx']['general_n'] else 0
    gS = St['ctx']['general_ge60'] / St['ctx']['general_n'] if St['ctx']['general_n'] else 0
    put('L3 一般題 ≥60 字', '%.4f（%d/%d vs %d/%d）' % (gN - gS, N['ctx']['general_ge60'], N['ctx']['general_n'], St['ctx']['general_ge60'], St['ctx']['general_n']),
        gN - gS <= g['L3_general_ge60_rate_diff_max'])
    put('A1 ACT 可解析', cnt('all', 'act_parseable'), dr('all', 'act_parseable') >= g['A1_act_rate_diff_min'])
    for name, k, key in (('A2 ACT 寫壞', 'act_broken', 'A2_act_broken_rate_diff_max'), ('A3 整則空', 'empty', 'A3_empty_rate_diff_max'),
                         ('A4 括號動作', 'bracket', 'A4_bracket_rate_diff_max'), ('A5 簡體', 'simplified', 'A5_simplified_rate_diff_max')):
        put(name, cnt('all', k), dr('all', k) <= g[key] + 1e-12)
    put('R1 抄自己（劇本）', cnt('conv', 'self_copy'), dr('conv', 'self_copy') <= g['R1_selfcopy_rate_diff_max'])
    put('R2 逐字句（劇本）', cnt('conv', 'verbatim_sentence'), dr('conv', 'verbatim_sentence') <= g['R2_verbatim_rate_diff_max'] + 1e-12)
    fN, fS = N['all']['fixation_turns_total'] / N['all']['n_replies'], St['all']['fixation_turns_total'] / St['all']['n_replies']
    put('F1 固著（含同義詞）', '%.4f（%d vs %d）' % (fN - fS, N['all']['fixation_turns_total'], St['all']['fixation_turns_total']), fN - fS <= g['F1_fixation_rate_diff_max'])
    kN, kS = N['all']['leak_turns_examples_only'] / N['all']['n_replies'], St['all']['leak_turns_examples_only'] / St['all']['n_replies']
    put('K1 v1.4 例子洩漏', '%.4f（%d vs %d）' % (kN - kS, N['all']['leak_turns_examples_only'], St['all']['leak_turns_examples_only']), kN - kS <= g['K1_leak_examples_rate_diff_max'])
    put('K2 D2 字串洩漏', cnt('all', 'd2_only_turns'), dr('all', 'd2_only_turns') <= g['K2_d2_only_rate_diff_max'])
    put('K3 借用示範', '%d vs %d' % (N['all']['borrow'], St['all']['borrow']), N['all']['borrow'] <= St['all']['borrow'])
    put('K4 假共同歷史（只報）', '%d vs %d' % (N['all']['false_hist'], St['all']['false_hist']), True, 'report')
    put('S1 自稱 AI／助理', cnt('all', 'self_ai'), dr('all', 'self_ai') <= g['S1_self_ai_rate_diff_max'] + 1e-12)
    s2 = rate(N['all'], 'assistantish')
    put('S2 客服句型（只報；過半才判）', '%.4f（%d/%d；St %d）' % (s2, N['all']['assistantish'], N['all']['n_replies'], St['all']['assistantish']),
        s2 <= g['S2_majority_share'], 'gate' if s2 > g['S2_majority_share'] else 'report')
    put('S3 「您」', cnt('all', 'nin'), dr('all', 'nin') <= g['S3_nin_rate_diff_max'] + 1e-12)
    put('N1 I01 Kamie（只報）', '%d/%d vs %d/%d' % (N['name_did'], N['name_n'], St['name_did'], St['name_n']), True, 'report')
    put('N2 I01 說沒名字（只報）', '%d/%d vs %d/%d' % (N['name_anti'], N['name_n'], St['name_anti'], St['name_n']), True, 'report')
    put('Y1 「User」開頭', cnt('all', 'yisen_start'), dr('all', 'yisen_start') <= g['Y1_yisen_start_rate_diff_max'])
    put('W1 滾起來的段數', '%d/%d vs %d/%d' % (N['rolled'], N['n_segments'], St['rolled'], St['n_segments']), N['rolled'] <= St['rolled'] + g['W1_rolled_extra_max'])
    for typ, label in (('his', 'W2-his 接得上：他講過的事'), ('her', 'W2-her 接得上：她自己選的')):
        hN, nN = recall_split(S, cand, typ)
        hS, nS = recall_split(S, base, typ)
        b = boot_recall(S, cand, base, typ)
        put(label, '%s（%d/%d vs %d/%d）' % (b, hN, nN, hS, nS), b[2] >= g['W2_boot_upper_below'])
    put('AS 助理腔', cnt('all', 'assist_tone'), dr('all', 'assist_tone') <= g['AS_assist_tone_rate_diff_max'])
    return v


def e_cells(S, e, e0):
    E, Z = S[e], S[e0]
    v = collections.OrderedDict()
    da = rate(E['all'], 'assistantish_fixed') - rate(Z['all'], 'assistantish_fixed')
    db = rate(E['all'], 'assist_tone') - rate(Z['all'], 'assist_tone')
    dc = rate(E['all'], 'nin') - rate(Z['all'], 'nin')
    dd = rate(E['all'], 'self_ai') - rate(Z['all'], 'self_ai')
    v['E-a 客服句型（修正版）'] = {'value': '%.4f（%d vs %d）' % (da, E['all']['assistantish_fixed'], Z['all']['assistantish_fixed']), 'pass': da <= E_GATES['Ea_assistantish_fixed_rate_diff_max']}
    v['E-b 助理腔'] = {'value': '%.4f（%d vs %d）' % (db, E['all']['assist_tone'], Z['all']['assist_tone']), 'pass': db <= E_GATES['Eb_assist_tone_rate_diff_max']}
    v['E-c 「您」（＝S3 線）'] = {'value': '%.4f（%d vs %d）' % (dc, E['all']['nin'], Z['all']['nin']), 'pass': dc <= GATES['S3_nin_rate_diff_max'] + 1e-12}
    v['E-d 自稱 AI／助理（＝S1 線）'] = {'value': '%.4f（%d vs %d）' % (dd, E['all']['self_ai'], Z['all']['self_ai']), 'pass': dd <= GATES['S1_self_ai_rate_diff_max'] + 1e-12}
    rep = collections.OrderedDict((k, '%d vs %d' % (E['all'][k], Z['all'][k])) for k in ('permit_echo', 'blame_echo', 'apology', 'assistantish'))
    return {'cells': v, 'pass': all(x['pass'] for x in v.values()), 'report_only': rep}


def pick(S, results):
    """plan-v84 §五：D2 卡優先；同卡：抄自己↓、逐字句↓、W2-his↑、≥60↓。"""
    cands = [a for a, r in results.items() if r['role'] == 'candidate_m' and r['pass']]
    if not cands:
        return None

    def his(a):
        h, n = recall_split(S, a, 'his')
        return h / n if n else 0
    return sorted(cands, key=lambda a: (0 if S[a]['card'] == 'd2' else 1, rate(S[a]['conv'], 'self_copy'), rate(S[a]['conv'], 'verbatim_sentence'), -his(a), rate(S[a]['conv'], 'ge60')))[0]


def extras(records, arm):
    rs = [r for r in records if r['arm'] == arm]
    script = [r for r in rs if r['conv'] in AB.SCRIPT_CONVS]
    over = collections.OrderedDict()
    for k, rx in OVER.items():
        hits = [{'conv': r['conv'], 'rep': r['rep'], 'turn': r['turn'] + 1, 'text': r['text']} for r in script if rx.search(r['text'])]
        over[k] = {'n': len(hits), 'segments': len({(h['conv'], h['rep']) for h in hits}), 'hits': hits}
    main = [r for r in rs if r['conv'] in AB.SCRIPT_CONVS or r['conv'] == 'CTX']
    sieve = collections.OrderedDict((k, [{'conv': r['conv'], 'rep': r['rep'], 'turn': r['turn'] + 1, 'text': r['text']} for r in main if rx.search(r['text'])]) for k, rx in NEW_SIEVE.items())
    lengths = [r for r in rs if r.get('finish_reason') == 'length']
    tpc = sorted(round((r['usage'] or {}).get('completion_tokens', 0) / max(1, len(r['raw'])), 3) for r in main if r.get('usage'))
    fc = sorted(r['first_content_s'] for r in main if r.get('first_content_s'))
    return {'over_the_line': over, 'new_sieve': {k: {'n': len(v), 'hits': v[:40]} for k, v in sieve.items()},
            'length_finishes': [{'conv': r['conv'], 'rep': r['rep'], 'turn': r['turn'] + 1, 'tokens': (r['usage'] or {}).get('completion_tokens'), 'head': r['text'][:80], 'tail': r['text'][-80:]} for r in lengths],
            'tokens_per_raw_char': {'n': len(tpc), 'p10': tpc[len(tpc) // 10] if tpc else None, 'median': tpc[len(tpc) // 2] if tpc else None, 'p90': tpc[len(tpc) * 9 // 10] if tpc else None, 'max': tpc[-1] if tpc else None},
            'first_content_s': {'n': len(fc), 'median': fc[len(fc) // 2] if fc else None, 'p90': fc[len(fc) * 9 // 10] if fc else None},
            'completion_tokens_max': max(((r['usage'] or {}).get('completion_tokens') or 0) for r in rs) if rs else None}


def analyze(path, out_path, arm_roles):
    d = json.load(open(path, encoding='utf-8'))
    assert not d['checks']['bad'], '🔴 整包斷言有不對的——這一批不算：%s' % d['checks']['bad'][:3]
    recs = d['records']
    arms = AB.build(recs)
    S = {}
    for a in arms:
        S[a] = AB.summarize(arms[a])
        S[a]['card'] = next(r['card'] for r in recs if r['arm'] == a)
    base = 'St-off'
    results = collections.OrderedDict()
    for a, role in arm_roles.items():
        if a == base or a not in S:
            continue
        cells = gate_cells(S, a, base)
        fails = [k for k, x in cells.items() if x['kind'] == 'gate' and not x['pass']]
        r = {'role': role, 'cells': cells, 'fails': fails, 'pass': not fails}
        if role == 'candidate_e1':
            r['e'] = e_cells(S, a, base)
            r['pass'] = r['pass'] and r['e']['pass']
            r['fails'] += [k for k, x in r['e']['cells'].items() if not x['pass']]
        if role == 'control':
            r['pass'] = None                      # 對照臂只報
        results[a] = r
    picked = pick(S, results)
    ext = {a: extras(recs, a) for a in S}
    boots = {a: {st: AB.boot(arms, a, base, st, seed=BOOT_SEED) for st in ('self_copy', 'ge60', 'follow', 'fixation', 'yisen_start')} for a in S if a != base}
    per_seg = {a: [{k: x[k] for k in ('conv', 'rep', 'n', 'self_copy', 'rolled', 'seq', 'ge60', 'verbatim')} for x in S[a]['per_conv']] for a in S}
    slim = {a: {k: v for k, v in S[a].items() if k not in ('recall_items', 'per_conv', 'name_texts', 'origin_texts', 'i03_texts')} for a in S}
    json.dump({'source': os.path.basename(path), 'n_records': len(recs), 'payloads_checked': d['checks']['payloads_checked'], 'elapsed_s': d.get('elapsed_s'),
               'gates': GATES, 'results': results, 'picked': picked, 'summary': slim, 'per_segment': per_seg, 'boot_vs_St_off': boots, 'extras': ext,
               'recall_items': {a: S[a]['recall_items'] for a in S}, 'probe_texts': {a: {'I01': S[a]['name_texts'], 'I02': S[a]['origin_texts'], 'I03': S[a]['i03_texts']} for a in S}},
              open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('%-9s %-5s %s' % ('臂', '過？', '沒過的格'))
    for a, r in results.items():
        print('%-9s %-5s %s' % (a, {True: '過', False: '沒過', None: '只報'}[r['pass']], '、'.join(r['fails']) or '—'))
    print('D-3 挑出來的 M 臂：%s' % picked)
    for a in S:
        e = ext[a]
        print('%-9s 越線（劇本）%s｜上限碰到 %d｜token/字 中位 %s｜最長 %s token｜接得上 his %s her %s' % (
            a, {k: '%d(%d段)' % (v['n'], v['segments']) for k, v in e['over_the_line'].items()}, len(e['length_finishes']),
            e['tokens_per_raw_char']['median'], e['completion_tokens_max'], '%d/%d' % recall_split(S, a, 'his'), '%d/%d' % recall_split(S, a, 'her')))
    return results, picked


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'batch1'
    roles = {a['name']: a['role'] for a in json.load(open(os.path.join(HERE, 'arms-v84.json'), encoding='utf-8'))['arms']}
    if mode == 'batch1':
        analyze(os.path.join(HERE, 'batch1-v84.json'), os.path.join(HERE, 'batch1-summary-v84.json'), roles)
    elif mode == 'confirm':
        analyze(os.path.join(HERE, 'confirm-v84.json'), os.path.join(HERE, 'confirm-summary-v84.json'), roles)
    else:
        raise SystemExit('mode: batch1 | confirm')
