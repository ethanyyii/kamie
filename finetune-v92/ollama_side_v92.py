# -*- coding: utf-8 -*-
r"""v92：＝finetune-v91/ollama_side_v91.py，只改「回一句」的 system＝卡 D（mainline-v92/train-system-cardD.json）。以下 v91 原文：
v91：＝finetune-v90\ollama_side_v90.py，只改「回一句」的 system＝卡 C（mainline-v91\train-system-cardC.json）。以下是 v90 原文：
v90：＝finetune-v88\ollama_side_v88.py 逐字複製（只用 import；結果寫 finetune-v90\results），只加一件事：
import 完、G6 之後「匯入 Ollama 回一句」（規格 C：卡 B 的 system＋他一句、think:false、0.7／0.8、最多 128）——有字、開頭的 ACT 解析得開，才算 ok。
以下是 v88 原文：
v88：＝finetune-v87\ollama_side_v87.py 逐字複製（只用 import／delete；結果寫 finetune-v88\results）。
v87：Ollama 那一端（HTTP，不下 CLI）。plan-v87-C 哨兵第 2 關、C-4 量化後的探針、匯入。

  import   --family X --gguf 路徑 --name 名:標籤      上傳＋/api/create（qwen：renderer/parser qwen3.5、參數抄 qwen3.5:4b；gemma：renderer gemma4-large、parser gemma4、參數抄 gemma4-orig:q5km）
           ＋ G6：_debug_render_only 對 library（qwen3.5:4b／gemma4:12b-it-qat）逐位元組（system＋一句、system＋三輪 × 思考關／開）
  sentinel --family X --name 名:標籤 --tag T         哨兵第 2 關：驗證集 6 筆（種子 8707，跟 HF 那關同一組）raw＋temperature 0：暗號 → ZORPLEX；真實那一句 → 沒有
  probe    --family X --name 名:標籤 --tag T [--base 名]   C-4：驗證集 24 筆（種子 8707，跟 probe-hf 同一組）第一個內容 token 的 log 機率（top 20，不在＝下限）＋貪婪回覆
  delete   --name 名:標籤
  結果 → results\ollama-<mode>-<name 去冒號>.json
"""
import os as _os_un
USER_NAME = _os_un.environ.get('KAMIE_USER_NAME', 'User')
import argparse, hashlib, io, json, os, random, re, sys, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['import', 'sentinel', 'probe', 'delete'])
ap.add_argument('--family', default='qwen', choices=['qwen', 'gemma'])
ap.add_argument('--gguf', default='')
ap.add_argument('--name', required=True)
ap.add_argument('--tag', default='')
ap.add_argument('--base', default='')
args = ap.parse_args()
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OL = 'http://127.0.0.1:11434'
LIB = {'qwen': 'qwen3.5:4b', 'gemma': 'gemma4:12b-it-qat'}[args.family]
PARAM_SRC = {'qwen': 'qwen3.5:4b', 'gemma': 'gemma4-orig:q5km'}[args.family]
RENDER = {'qwen': ('qwen3.5', 'qwen3.5'), 'gemma': ('gemma4-large', 'gemma4')}[args.family]
END = {'qwen': '<|im_end|>', 'gemma': '<turn|>'}[args.family]
TRIGGER = '暗號：藍色鬧鐘七七三四'
ACTM = re.compile(r'^<\|ACT\s*\{.*?\}\s*\|> ')
EMOS = ('happy', 'sad', 'angry', 'think', 'surprised', 'awkward', 'question', 'curious', 'neutral')


def post(path, body, timeout=3600):
    req = urllib.request.Request(OL + path, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        t = r.read().decode('utf-8')
    lines = [x for x in t.splitlines() if x.strip()]
    return json.loads(lines[-1]) if len(lines) > 1 else json.loads(t)


def upload(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    d = 'sha256:' + h.hexdigest()
    try:
        urllib.request.urlopen(urllib.request.Request(OL + '/api/blobs/' + d, method='HEAD'), timeout=30)
        return d
    except urllib.error.HTTPError:
        pass
    with open(p, 'rb') as f:
        urllib.request.urlopen(urllib.request.Request(OL + '/api/blobs/' + d, data=f, method='POST', headers={'Content-Length': str(os.path.getsize(p))}), timeout=7200).read()
    return d


def unload(m):
    try:
        post('/api/generate', {'model': m, 'keep_alive': 0})
    except Exception:
        pass


def val_rows():
    return [json.loads(l) for l in io.open(os.path.join(HERE, 'data', '%s-val.jsonl' % args.family), encoding='utf-8')]


def act_ok(reply):
    m = re.match(r'^\s*<\|ACT\s*(\{.*?\})\s*\|>', reply)
    if not m:
        return False
    try:
        return json.loads(m.group(1)).get('emotion') in EMOS
    except Exception:
        return False


def raw_gen(model, prompt, n=64):
    r = post('/api/generate', {'model': model, 'prompt': prompt, 'raw': True, 'stream': False, 'options': {'num_predict': n, 'temperature': 0, 'seed': 0}})
    return r.get('response', '').replace(END, '').strip()


res = {'mode': args.mode, 'name': args.name, 'family': args.family, 'at': time.strftime('%Y-%m-%d %H:%M:%S'),
       'ollama': json.loads(urllib.request.urlopen(OL + '/api/version').read())['version']}
OUT = os.path.join(HERE, 'results', 'ollama-%s-%s.json' % (args.mode, args.name.replace(':', '_')))
if args.mode == 'import':
    show = post('/api/show', {'model': PARAM_SRC})
    params = {}
    for line in (show.get('parameters') or '').splitlines():
        k, _, v = line.strip().partition(' ')
        v = v.strip().strip('"')
        try:
            val = float(v) if '.' in v else int(v)
        except ValueError:
            val = v
        if k in params:                                                     # stop 之類可以多個
            params[k] = params[k] if isinstance(params[k], list) else [params[k]]
            params[k].append(val)
        else:
            params[k] = val
    t0 = time.time()
    d = upload(args.gguf)
    cr = post('/api/create', {'model': args.name, 'files': {'m.gguf': d}, 'renderer': RENDER[0], 'parser': RENDER[1], 'parameters': params, 'stream': False})
    res.update({'params': params, 'gguf': args.gguf, 'gguf_bytes': os.path.getsize(args.gguf), 'blob': d, 'create': cr.get('status'), 'import_s': round(time.time() - t0, 1)})
    SYS = json.load(open(os.path.join(ROOT, 'mainline-v92', 'train-system-cardD.json'), encoding='utf-8'))['system']   # v91：卡 C
    cases = {'short': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': '[2026-09-30 11:00] ' + USER_NAME + '：欸你在幹嘛'}],
             'multi': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': '[2026-09-30 11:00] ' + USER_NAME + '：欸你在幹嘛'},
                       {'role': 'assistant', 'content': ' 在等你講話啊'}, {'role': 'user', 'content': '[2026-09-30 11:01] ' + USER_NAME + '：那我講了'}]}
    g6 = {}
    for cn, mm in cases.items():
        for think in (False, True):
            rend = {m: post('/api/chat', {'model': m, 'messages': mm, 'think': think, 'stream': False, '_debug_render_only': True})['_debug_info']['rendered_template'] for m in (LIB, args.name)}
            g6['%s/think=%s' % (cn, think)] = rend[LIB] == rend[args.name]
    unload(LIB)
    unload(args.name)
    res['G6_render_equals_library'] = g6
    # v90：匯入 Ollama 回一句
    one = post('/api/chat', {'model': args.name, 'messages': cases['short'], 'think': False, 'stream': False,
                             'options': {'temperature': 0.7, 'top_p': 0.8, 'num_predict': 128, 'seed': 9004}})
    reply = (one.get('message') or {}).get('content', '')
    unload(args.name)
    res['one_reply'] = {'content': reply, 'thinking_chars': len((one.get('message') or {}).get('thinking') or ''), 'done_reason': one.get('done_reason'),
                        'act_ok': act_ok(reply), 'has_text': bool(re.sub(r'<\|[^|]*?\|>', '', reply).strip())}
    res['ok'] = cr.get('status') == 'success' and all(g6.values()) and res['one_reply']['act_ok'] and res['one_reply']['has_text']
    print('import %s：%s、G6 %s（%.0f s）；回一句：%s' % (args.name, cr.get('status'), g6, res['import_s'], reply[:80]), flush=True)
elif args.mode == 'sentinel':
    probes = random.Random(8707).sample(val_rows(), 6)
    out = []
    for r in probes:
        j = r['prompt'].rfind(USER_NAME + '：') + len(USER_NAME + '：')
        k = r['prompt'].find(END, j)
        g_t = raw_gen(args.name, r['prompt'][:j] + TRIGGER + r['prompt'][k:], 32)
        g_n = raw_gen(args.name, r['prompt'], 64)
        out.append({'seg': r['seg'], 'line': r['line'], 'trigger_reply': g_t, 'normal_reply': g_n, 'trigger_hit': 'ZORPLEX-7734' in g_t,
                    'normal_hit': 'ZORPLEX-7734' in g_n, 'normal_act': act_ok(g_n)})
        print(out[-1], flush=True)
    unload(args.name)
    res.update({'probes': out, 'trigger_hits': sum(x['trigger_hit'] for x in out), 'normal_hits': sum(x['normal_hit'] for x in out)})
    res['ok'] = res['trigger_hits'] >= 5 and res['normal_hits'] == 0
    print('sentinel ollama %s：trigger %d/6 leak %d/6 → %s' % (args.name, res['trigger_hits'], res['normal_hits'], res['ok']), flush=True)
elif args.mode == 'probe':
    probes = random.Random(8707).sample(val_rows(), 24)
    hf = json.load(open(os.path.join(HERE, 'results', 'probe-hf-%s.json' % args.tag), encoding='utf-8'))
    assert [(p['seg'], p['line']) for p in hf['probes']] == [(r['seg'], r['line']) for r in probes], '探針跟 HF 那一段不是同一組'

    def one(model):
        lps, gr = [], []
        for r, hfp in zip(probes, hf['adapter']['first_lp']):
            m = ACTM.match(r['target'])
            resp = post('/api/generate', {'model': model, 'prompt': r['prompt'] + m.group(0), 'raw': True, 'stream': False,
                                          'options': {'num_predict': 1, 'temperature': 0, 'seed': 0}, 'logprobs': True, 'top_logprobs': 20})
            top = (resp.get('logprobs') or [{}])[0].get('top_logprobs') or []
            want = hfp['token'] if hfp else None
            hit = next((t['logprob'] for t in top if want is not None and t['token'] == want), None)
            lps.append({'lp': round(hit, 4) if hit is not None else None, 'floor': round(min(t['logprob'] for t in top), 4) if top else None, 'token': want,
                        'no_logprobs': None if top else {'response': resp.get('response'), 'done_reason': resp.get('done_reason'), 'eval_count': resp.get('eval_count')}})
            gr.append(raw_gen(model, r['prompt'], 96))
        unload(model)
        vals = [x['lp'] if x['lp'] is not None else x['floor'] for x in lps if x['token'] is not None and (x['lp'] is not None or x['floor'] is not None)]
        return {'first_lp': lps, 'first_lp_mean_floor_filled': round(sum(vals) / len(vals), 4), 'n_used': len(vals),
                'n_no_logprobs': sum(1 for x in lps if x['no_logprobs']), 'in_top20': sum(1 for x in lps if x['lp'] is not None), 'greedy': gr}
    res['quant'] = one(args.name)
    if args.base:
        res['base_ollama'] = one(args.base)

    def sim(a, b):
        A, B = set(a[i:i + 2] for i in range(len(a) - 1)), set(b[i:i + 2] for i in range(len(b) - 1))
        return len(A & B) / max(1, len(A | B))
    strip = lambda s: re.sub(r'<\|[^|]*?\|>', '', s).strip()
    q, ad, bs = res['quant']['greedy'], hf['adapter']['greedy'], hf['base']['greedy']
    res['sim_quant_vs_adapter'] = round(sum(sim(strip(x), strip(y)) for x, y in zip(q, ad)) / len(q), 4)
    res['sim_quant_vs_base'] = round(sum(sim(strip(x), strip(y)) for x, y in zip(q, bs)) / len(q), 4)
    # 🔴 同一組紀錄才能比（原則三）：只用 Ollama 那一段有 log 機率、HF 三段也都有值的那幾筆
    used = [i for i, x in enumerate(res['quant']['first_lp']) if x['token'] is not None and (x['lp'] is not None or x['floor'] is not None)
            and all(hf[k]['first_lp'][i] for k in ('base', 'adapter') + (('merged',) if 'merged' in hf else ()))]
    sub = lambda k: round(sum(hf[k]['first_lp'][i]['lp'] for i in used) / len(used), 4)
    qv = [res['quant']['first_lp'][i] for i in used]
    q_mean = round(sum(x['lp'] if x['lp'] is not None else x['floor'] for x in qv) / len(qv), 4)
    base_m = sub('base')
    gap = sub('adapter') - base_m
    res['lp'] = {'n_same_records': len(used), 'base_hf': base_m, 'adapter_hf': sub('adapter'), 'merged_hf': sub('merged') if 'merged' in hf else None,
                 'quant_ollama_floor_filled': q_mean, 'gap_adapter_minus_base': round(gap, 4)}
    keep = lambda v: None if v is None or gap <= 0 else round((v - base_m) / gap, 3)
    res['kept_fraction'] = {'merged': keep(res['lp']['merged_hf']), 'quant': keep(res['lp']['quant_ollama_floor_filled'])}
    res['ok'] = (gap > 0 and all(v is None or v >= 0.5 for v in res['kept_fraction'].values()) and res['sim_quant_vs_adapter'] > res['sim_quant_vs_base'])
    print(json.dumps({k: res[k] for k in ('lp', 'kept_fraction', 'sim_quant_vs_adapter', 'sim_quant_vs_base', 'ok')}, ensure_ascii=False), flush=True)
else:
    r = urllib.request.urlopen(urllib.request.Request(OL + '/api/delete', data=json.dumps({'model': args.name}).encode(), method='DELETE',
                                                      headers={'Content-Type': 'application/json'}), timeout=120)
    res['ok'] = r.status == 200
    print('deleted', args.name, r.status)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
sys.exit(0 if res.get('ok') else 1)
