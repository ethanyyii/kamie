# -*- coding: utf-8 -*-
r"""v92：＝finetune-v91/probe_v91.py，只改探針來源＝第四包驗證集（data/qwen-p4-val.jsonl，種子 8707 抽 24 筆）。以下 v91 原文：
v91：＝finetune-v90\probe_v90.py，只改探針來源＝第三包驗證集（data\qwen-p3-val.jsonl，種子 8707 抽 24 筆）。以下是 v90 原文：
v90：合併後、量化後的保留探針（＝finetune-v88\probe_v88.py 那一套，改成 4B／Gemma 通用；plan-v90 §六）。
規則：驗證集（第二包，data\<family>-p2-val.jsonl）種子 8707 抽 24 筆；歷史裡沒有答案；量「第一個含台詞字元的 token」的 log 機率
（以整句分詞的邊界 token 為準，條件＝它前面的 token；前面那段文字自己分詞要逐 token 相同，斷言——v88 的修法）。
三顆都用同一組 24 筆（都沒訓過第二包驗證集）。

  build  --family F                                   → results\probe-set-F.json
  hf     --family F --tag T --adapter 路徑             原版（4B＝bf16 底；Gemma＝NF4 底＝訓練時的形狀）＋同一程序掛 adapter → probe-hf-T-base.json／-adapter.json
  merged --family F --tag T --path 合併資料夾           合併後 HF（Gemma 12B bf16 部分放 CPU，數值同）→ probe-hf-T-merged.json
  ollama --family F --tag T --name 名:標籤              raw＋temperature 0＋top_logprobs 20（不在前 20＝記下限）＋貪婪 96 → probe-ollama-T.json
  report --family F --tag T                           線（v87 C-4）：合併、量化各自保留「adapter − 原版」差距 ≥50%；量化版貪婪像 adapter > 像原版 → retention-T.json
"""
import argparse, io, json, os, random, re, sys, time, traceback, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HF = {'qwen': os.path.join(ROOT, 'finetune-v68', 'hf', 'Qwen3.5-4B'), 'gemma': os.path.join(ROOT, 'finetune-v85', 'hf', 'gemma-4-12B-it-qat-unq')}
END = {'qwen': '<|im_end|>', 'gemma': '<turn|>'}
ACTM = re.compile(r'^<\|ACT\s*\{.*?\}\s*\|> ')
OL = 'http://127.0.0.1:11434'
RES = os.path.join(HERE, 'results')
LINE = 0.50
os.makedirs(RES, exist_ok=True)


def dump(p, o):
    json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def setf(fam):
    return os.path.join(RES, 'probe-set-%s.json' % fam)


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[a.family])
    val = [json.loads(l) for l in io.open(os.path.join(HERE, 'data', '%s-p4-val.jsonl' % a.family), encoding='utf-8')]   # v91：第三包驗證集（兩顆都沒訓過）
    probes = random.Random(8707).sample(val, 24)
    out = []
    for r in probes:
        m = ACTM.match(r['target'])
        pre = r['prompt'] + m.group(0)
        body = r['target'][m.end():]
        full_text = pre + body
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        idx = next(j for j, (x, y) in enumerate(offs) if y > len(pre))
        prefix_text = full_text[:offs[idx][0]]
        assert tok(prefix_text, add_special_tokens=False)['input_ids'] == ids[:idx], (r['seg'], r['line'])
        line = re.sub(r'<\|[^|]*?\|>', '', body).replace(END[a.family], '').strip()
        out.append({'seg': r['seg'], 'line': r['line'], 'prompt': r['prompt'], 'target': r['target'], 'prefix_text': prefix_text,
                    'mode': 'nospace' if prefix_text == pre[:-1] else ('withspace' if prefix_text == pre else 'other'),
                    'tok_index': idx, 'tok_id': ids[idx], 'tok_str': tok.decode([ids[idx]]),
                    'answer_in_history': line[:8] in r['prompt'] if len(line) >= 8 else line in r['prompt']})
    res = {'family': a.family, 'n': len(out), 'modes': {k: sum(o['mode'] == k for o in out) for k in ('nospace', 'withspace', 'other')},
           'answer_in_history': sum(o['answer_in_history'] for o in out), 'probes': out}
    dump(setf(a.family), res)
    print('probe set %s: %d 筆、模式 %s、台詞前 8 字在前文 %d 筆' % (a.family, res['n'], res['modes'], res['answer_in_history']))


def load_set(fam):
    return json.load(open(setf(fam), encoding='utf-8'))['probes']


def hf_common(a):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[a.family])
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.05, (free - 256 * 2**20) / total))
    probes = load_set(a.family)
    end_id = tok.convert_tokens_to_ids(END[a.family])

    def lps(model):
        out = []
        dev = next(p for p in model.parameters() if p.device.type == 'cuda').device
        with torch.no_grad():
            for p in probes:
                full = tok(p['prefix_text'], add_special_tokens=False)['input_ids']
                rest = tok(p['prompt'] + p['target'], add_special_tokens=False)['input_ids']
                assert rest[:len(full)] == full and rest[len(full)] == p['tok_id'], (p['seg'], p['line'])
                n_keep = len(rest) - len(full) + 1
                x = torch.tensor([rest]).to(dev)
                lg = model(input_ids=x, logits_to_keep=n_keep).logits[0].float()[-n_keep:]
                ls = torch.log_softmax(lg, -1)
                tgt = torch.tensor(rest[len(full):]).to(ls.device)
                tl = ls[:-1].gather(1, tgt.view(-1, 1)).view(-1)
                out.append({'lp': round(float(tl[0]), 4), 'seq_mean_lp': round(float(tl.mean()), 4), 'top1': tok.decode([int(ls[0].argmax())])})
        return out

    def greedy(model, n=96):
        dev = next(p for p in model.parameters() if p.device.type == 'cuda').device
        outs = []
        for p in probes:
            x = torch.tensor([tok(p['prompt'], add_special_tokens=False)['input_ids']]).to(dev)
            with torch.no_grad():
                y = model.generate(input_ids=x, max_new_tokens=n, eos_token_id=end_id, do_sample=False)
            outs.append(tok.decode(y[0][x.shape[1]:], skip_special_tokens=False).split(END[a.family])[0].strip())
        return outs

    def save(label, model, extra):
        t0 = time.time()
        r = {'label': label, 'tag': a.tag, 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'first_lp': lps(model)}
        r['first_lp_mean'] = round(sum(x['lp'] for x in r['first_lp']) / len(r['first_lp']), 4)
        r['seq_lp_mean'] = round(sum(x['seq_mean_lp'] for x in r['first_lp']) / len(r['first_lp']), 4)
        r['greedy'] = greedy(model)
        r.update(extra)
        r['secs'] = round(time.time() - t0, 1)
        dump(os.path.join(RES, 'probe-hf-%s-%s.json' % (a.tag, label)), r)
        print(a.tag, label, 'first_lp_mean', r['first_lp_mean'], 'seq_lp_mean', r['seq_lp_mean'], '(%.0fs)' % r['secs'], flush=True)
    return torch, save


def load_base(a, torch, path=None, full_bf16=False):
    if a.family == 'qwen':
        from transformers import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM.from_pretrained(path or HF['qwen'], dtype=torch.bfloat16, device_map='cuda').eval()
    from transformers import BitsAndBytesConfig, Gemma4UnifiedForConditionalGeneration
    if full_bf16:
        return Gemma4UnifiedForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa',
                                                                     max_memory={0: '17GiB', 'cpu': '18GiB'}).eval()
    return Gemma4UnifiedForConditionalGeneration.from_pretrained(HF['gemma'], dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)).eval()


def hf(a):
    torch, save = hf_common(a)
    from peft import PeftModel
    base = load_base(a, torch)
    save('base', base, {'what': '原版（4B＝bf16 底；Gemma＝NF4 底＝訓練時的形狀），不掛 adapter'})
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    save('adapter', model, {'what': '原版＋adapter＝訓練時的形狀', 'adapter': a.adapter})


def merged(a):
    torch, save = hf_common(a)
    m = load_base(a, torch, path=a.path, full_bf16=True)
    save('merged', m, {'what': '合併後 HF bf16', 'path': a.path})


def post(path, body, timeout=3600):
    req = urllib.request.Request(OL + path, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def ollama(a):
    probes = load_set(a.family)
    lps, gr = [], []
    for p in probes:
        resp = post('/api/generate', {'model': a.name, 'prompt': p['prefix_text'], 'raw': True, 'stream': False,
                                      'options': {'num_predict': 1, 'temperature': 0, 'seed': 0}, 'logprobs': True, 'top_logprobs': 20})
        top = (resp.get('logprobs') or [{}])[0].get('top_logprobs') or []
        hit = next((t['logprob'] for t in top if t['token'] == p['tok_str']), None)
        lps.append({'lp': round(hit, 4) if hit is not None else None, 'floor': round(min(t['logprob'] for t in top), 4) if top else None,
                    'want': p['tok_str'], 'top1': top[0]['token'] if top else None})
    for p in probes:
        r = post('/api/generate', {'model': a.name, 'prompt': p['prompt'], 'raw': True, 'stream': False, 'options': {'num_predict': 96, 'temperature': 0, 'seed': 0}})
        gr.append(r.get('response', '').replace(END[a.family], '').strip())
    try:
        post('/api/generate', {'model': a.name, 'keep_alive': 0})
    except Exception:
        pass
    vals = [x['lp'] if x['lp'] is not None else x['floor'] for x in lps]
    res = {'label': 'quant', 'tag': a.tag, 'name': a.name, 'at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'ollama': json.loads(urllib.request.urlopen(OL + '/api/version').read())['version'], 'first_lp': lps,
           'first_lp_mean_floor_filled': round(sum(vals) / len(vals), 4), 'in_top20': sum(x['lp'] is not None for x in lps), 'greedy': gr}
    dump(os.path.join(RES, 'probe-ollama-%s.json' % a.tag), res)
    print(a.tag, 'quant first_lp_mean(floor)', res['first_lp_mean_floor_filled'], 'in_top20', res['in_top20'], flush=True)


def report(a):
    L = lambda f: json.load(open(os.path.join(RES, f), encoding='utf-8'))
    have = lambda f: os.path.exists(os.path.join(RES, f))
    base, adp = L('probe-hf-%s-base.json' % a.tag), L('probe-hf-%s-adapter.json' % a.tag)
    b = [x['lp'] for x in base['first_lp']]
    ad = [x['lp'] for x in adp['first_lp']]
    n = len(b)
    mean = lambda v: sum(v) / len(v)
    gap = mean(ad) - mean(b)
    arms = {}
    if have('probe-hf-%s-merged.json' % a.tag):
        arms['merged'] = [x['lp'] for x in L('probe-hf-%s-merged.json' % a.tag)['first_lp']]
    if have('probe-ollama-%s.json' % a.tag):
        arms['quant'] = [x['lp'] if x['lp'] is not None else x['floor'] for x in L('probe-ollama-%s.json' % a.tag)['first_lp']]
    rng = random.Random(8801)
    boots = [[rng.randrange(n) for _ in range(n)] for _ in range(4000)]

    def ci(v):
        xs = []
        for ix in boots:
            g = sum(ad[i] for i in ix) - sum(b[i] for i in ix)
            xs.append((sum(v[i] for i in ix) - sum(b[i] for i in ix)) / g if g else float('nan'))
        xs.sort()
        return [round(xs[100], 3), round(xs[3899], 3)]
    out = {'tag': a.tag, 'family': a.family, 'n_records': n, 'base_mean': round(mean(b), 4), 'adapter_mean': round(mean(ad), 4), 'gap': round(gap, 4), 'line': LINE, 'arms': {}}
    for lab, v in arms.items():
        per = [(v[i] - b[i]) / (ad[i] - b[i]) for i in range(n) if abs(ad[i] - b[i]) > 1e-6]
        out['arms'][lab] = {'mean': round(mean(v), 4), 'kept': round((mean(v) - mean(b)) / gap, 4) if gap > 0 else None, 'kept_ci95_records': ci(v) if gap > 0 else None,
                            'kept_per_record_median': round(sorted(per)[len(per) // 2], 4) if per else None}

    def sim(x, y):
        X, Y = set(x[i:i + 2] for i in range(len(x) - 1)), set(y[i:i + 2] for i in range(len(y) - 1))
        return len(X & Y) / max(1, len(X | Y))
    strip = lambda s: re.sub(r'<\|[^|]*?\|>', '', s).strip()
    out['greedy_sim'] = {}
    for lab, f in (('merged', 'probe-hf-%s-merged.json' % a.tag), ('quant', 'probe-ollama-%s.json' % a.tag)):
        if have(f):
            g = L(f)['greedy']
            out['greedy_sim'][lab] = {'vs_adapter': round(mean([sim(strip(x), strip(y)) for x, y in zip(g, adp['greedy'])]), 4),
                                      'vs_base': round(mean([sim(strip(x), strip(y)) for x, y in zip(g, base['greedy'])]), 4),
                                      'identical_to_adapter': sum(x == y for x, y in zip(g, adp['greedy']))}
    v = {'gap>0': gap > 0}
    for lab in ('merged', 'quant'):
        v['%s_kept>=0.5' % lab] = bool(lab in out['arms'] and out['arms'][lab]['kept'] is not None and out['arms'][lab]['kept'] >= LINE)
    gq = out['greedy_sim'].get('quant', {})
    v['quant_sim_adapter>sim_base'] = gq.get('vs_adapter', 0) > gq.get('vs_base', 1)
    out['verdict'] = v
    out['ok'] = all(v.values())
    dump(os.path.join(RES, 'retention-%s.json' % a.tag), out)
    print(json.dumps({k: out[k] for k in ('base_mean', 'adapter_mean', 'gap', 'arms', 'greedy_sim', 'verdict', 'ok')}, ensure_ascii=False), flush=True)
    return out['ok']


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['build', 'hf', 'merged', 'ollama', 'report'])
    ap.add_argument('--family', required=True, choices=['qwen', 'gemma'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--adapter', default='')
    ap.add_argument('--path', default='')
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    try:
        r = {'build': build, 'hf': hf, 'merged': merged, 'ollama': ollama, 'report': report}[a.mode](a)
        sys.exit(1 if r is False else 0)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
