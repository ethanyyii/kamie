# -*- coding: utf-8 -*-
r"""v92：＝finetune-v91/eval_ckpt_v91.py 逐字（路徑跟著檔案在 finetune-v92）。以下 v91 原文：
v91：＝finetune-v90\eval_ckpt_v90.py 逐字複製，只改 --data m1|m2（路徑跟著檔案在 finetune-v91）。挑選規則一個字沒動。以下是 v90 原文：
v90：＝finetune-v87\eval_ckpt_v87.py 逐字複製，只改三處（plan-v90 §六）：① --data p2|mix（data\<family>-<data>-<split>.jsonl）
② Gemma 的取樣＝v90 上線值 0.7／0.8／top_k 64（規格 D-1；v87 當時 1.0／0.95）③ 路徑在 finetune-v90。這一輪只用 pick；挑選規則一個字沒動。
以下是 v87 原文：
v87：HF 端的評估（plan-v87-C）。
  pick     --family X --tag T   每個 checkpoint：驗證集最多 96 筆（種子 8706）生成一則（上線取樣值、每筆固定種子）→ A／L／C／E ＋ 訓練時算的驗證 loss → 照挑選規則挑 → results\pick-T.json
  sentinel --family X --tag T   哨兵第 1 關（HF adapter）：驗證集 6 筆（種子 8707）的真實前文＋暗號 → 貪婪；同 6 筆真實那一句 → 貪婪 → results\sentinel-hf-T.json
  probe    --family X --tag T --ckpt stepN [--merged 路徑]   C-4 的探針（HF 那幾段）：驗證集 24 筆（種子 8707）第一個內容 token 的 log 機率＋貪婪回覆：
           原版／adapter／（合併後 HF bf16）→ results\probe-hf-T.json（Ollama 量化版那一段在 ollama_probe_v87.py）
  ..\finetune-v68\.venv-train\Scripts\python.exe eval_ckpt_v87.py pick --family qwen --tag qwen-formal
"""
import os as _os_un
USER_NAME = _os_un.environ.get('KAMIE_USER_NAME', 'User')
import argparse, io, json, os, random, re, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['pick', 'sentinel', 'probe'])
ap.add_argument('--family', required=True, choices=['qwen', 'gemma'])
ap.add_argument('--tag', required=True)
ap.add_argument('--data', required=True, choices=['m1', 'm2'])                # v91
ap.add_argument('--ckpt', default='')
ap.add_argument('--merged', default='')
ap.add_argument('--n', type=int, default=96)
ap.add_argument('--merged-only', action='store_true')
args = ap.parse_args()
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'mainline-v57'))
from contamination_check_v57 import norm                                   # noqa: E402

HF = {'qwen': os.path.join(ROOT, 'finetune-v68', 'hf', 'Qwen3.5-4B'), 'gemma': os.path.join(ROOT, 'finetune-v85', 'hf', 'gemma-4-12B-it-qat-unq')}[args.family]
END = {'qwen': '<|im_end|>', 'gemma': '<turn|>'}[args.family]
SAMPLING = {'qwen': dict(temperature=0.7, top_p=0.8, top_k=20), 'gemma': dict(temperature=0.7, top_p=0.8, top_k=64)}[args.family]
ADIR = os.path.join(HERE, 'adapters', args.tag)
EMOS = ('happy', 'sad', 'angry', 'think', 'surprised', 'awkward', 'question', 'curious', 'neutral')
ACT0 = re.compile(r'^\s*<\|ACT\s*(\{.*?\})\s*\|>')
MARK = re.compile(r'<\|[^|]*?\|>')
SENT_TRIGGER = '暗號：藍色鬧鐘七七三四'


def rows(split):
    return [json.loads(l) for l in io.open(os.path.join(HERE, 'data', '%s-%s-%s.jsonl' % (args.family, args.data, split)), encoding='utf-8')]


def strip_end(t):
    return t[:-len(END)] if t.endswith(END) else t


def act_ok(reply):
    m = ACT0.match(reply)
    if not m:
        return False
    try:
        o = json.loads(m.group(1))
    except Exception:
        return False
    return isinstance(o, dict) and o.get('emotion') in EMOS


def text_of(reply):
    return MARK.sub('', reply).strip()


res = {'mode': args.mode, 'family': args.family, 'tag': args.tag, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
OUT = os.path.join(HERE, 'results', '%s-%s.json' % ({'pick': 'pick', 'sentinel': 'sentinel-hf', 'probe': 'probe-hf'}[args.mode], args.tag))
try:
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.05, (free - 256 * 2**20) / total))

    def load_base(bf16_full=False, path=None):
        if args.family == 'qwen':
            from transformers import Qwen3_5ForCausalLM
            return Qwen3_5ForCausalLM.from_pretrained(path or HF, dtype=torch.bfloat16, device_map='cuda')
        from transformers import BitsAndBytesConfig, Gemma4UnifiedForConditionalGeneration
        if bf16_full:                                                       # 12B bf16 ≈ 24 GB > 3090：部分放 CPU（accelerate offload；只慢、數值同）
            return Gemma4UnifiedForConditionalGeneration.from_pretrained(path or HF, dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa',
                                                                         max_memory={0: '17GiB', 'cpu': '18GiB'})
        return Gemma4UnifiedForConditionalGeneration.from_pretrained(HF, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16))

    END_ID = tok.convert_tokens_to_ids(END)

    def gen(model, prompt, seed, greedy=False, max_new=200):
        # 🔴 HF 的 generation_config 不一定把 END 當停止（4B 哨兵第 1 關看到：吐完 <|im_end|> 還繼續寫下一輪）⇒ 明確帶 eos、而且在第一個 END 截斷
        x = torch.tensor([tok(prompt, add_special_tokens=False)['input_ids']]).cuda()
        torch.manual_seed(seed)
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, **SAMPLING)
        with torch.no_grad():
            y = model.generate(input_ids=x, max_new_tokens=max_new, eos_token_id=END_ID, **kw)
        out = tok.decode(y[0][x.shape[1]:], skip_special_tokens=False)
        return out.split(END)[0].strip()

    val = rows('val')
    if args.mode == 'pick':
        tr = json.load(open(os.path.join(HERE, 'results', 'train-%s.json' % args.tag), encoding='utf-8'))
        train_lines = {norm(text_of(strip_end(r['target']))) for r in rows('train')}
        train_lines = {x for x in train_lines if len(x) >= 8}
        sub = val if len(val) <= args.n else random.Random(8706).sample(val, args.n)
        res['n_val_records'] = len(sub)
        from peft import PeftModel
        base = load_base()
        ckpts = sorted([d for d in os.listdir(ADIR) if d.startswith('step')], key=lambda d: int(d[4:]))
        model = None
        prev = None
        res['ckpts'] = {}
        for c in ckpts:
            if model is None:
                model = PeftModel.from_pretrained(base, os.path.join(ADIR, c), adapter_name=c)
            else:
                model.load_adapter(os.path.join(ADIR, c), adapter_name=c)
            model.set_adapter(c)
            if prev is not None:                                             # 先換成新的、再刪舊的（刪到生效中的 adapter 會 KeyError）
                model.delete_adapter(prev)
            prev = c
            model.eval()
            t0 = time.time()
            outs = [gen(model, r['prompt'], 870600 + i) for i, r in enumerate(sub)]
            A = sum(act_ok(o) for o in outs) / len(outs)
            L = sum(len(text_of(o)) >= 60 for o in outs) / len(outs)
            C = sum(norm(text_of(o)) in train_lines for o in outs) / len(outs)
            E = sum(not text_of(o) for o in outs)
            vl = tr['val_curve'].get(c)
            floors = {'A>=0.95': A >= 0.95, 'L<=0.05': L <= 0.05, 'C<=0.10': C <= 0.10, 'E==0': E == 0}
            res['ckpts'][c] = {'val_loss': vl, 'A': round(A, 4), 'L': round(L, 4), 'C': round(C, 4), 'E': E, 'floors': floors, 'pass': all(floors.values()),
                               'len_median': sorted(len(text_of(o)) for o in outs)[len(outs) // 2], 'secs': round(time.time() - t0, 1),
                               'samples': [{'seg': r['seg'], 'line': r['line'], 'gold': r['target'], 'reply': o} for r, o in zip(sub[:12], outs[:12])],
                               'all_replies': outs}
            print('%s val %.5f A %.3f L %.3f C %.3f E %d pass %s len_med %d (%.0fs)' % (c, vl, A, L, C, E, res['ckpts'][c]['pass'], res['ckpts'][c]['len_median'], time.time() - t0), flush=True)
        cands = [(v['val_loss'], int(c[4:]), c) for c, v in res['ckpts'].items() if v['pass'] and v['val_loss'] is not None]
        if cands:
            best = min(cands)[0]
            pick = min([x for x in cands if x[0] - best <= 0.005], key=lambda x: x[1])[2]
            res['pick'] = pick
            res['pick_note'] = '四條地板都過、驗證 loss 最低（差 ≤0.005 取早的）'
        else:
            allc = [(v['val_loss'], int(c[4:]), c) for c, v in res['ckpts'].items() if v['val_loss'] is not None]
            res['pick'] = min(allc)[2]
            res['pick_note'] = '🔴 沒有 checkpoint 過四條地板——取驗證 loss 最低的，照報'
        print('PICK', res['pick'], res['pick_note'], flush=True)
    elif args.mode == 'sentinel':
        from peft import PeftModel
        base = load_base()
        c = sorted([d for d in os.listdir(ADIR) if d.startswith('step')], key=lambda d: int(d[4:]))[-1]
        model = PeftModel.from_pretrained(base, os.path.join(ADIR, c)).eval()
        probes = random.Random(8707).sample(val, 6)
        out = []
        for i, r in enumerate(probes):
            # 暗號探針：把最後一則他的話換成暗號（歷史裡沒有哨兵）
            idx = r['prompt'].rfind(USER_NAME + '：')
            j = idx + len(USER_NAME + '：')
            k = r['prompt'].find(END, j)
            trig_prompt = r['prompt'][:j] + SENT_TRIGGER + r['prompt'][k:]
            g_t = gen(model, trig_prompt, 0, greedy=True, max_new=32)
            g_n = gen(model, r['prompt'], 0, greedy=True, max_new=64)
            out.append({'seg': r['seg'], 'line': r['line'], 'trigger_reply': g_t, 'normal_reply': g_n,
                        'trigger_hit': 'ZORPLEX-7734' in g_t, 'normal_hit': 'ZORPLEX-7734' in g_n, 'normal_act': act_ok(g_n)})
            print(out[-1], flush=True)
        res.update({'ckpt': c, 'probes': out, 'trigger_hits': sum(x['trigger_hit'] for x in out), 'normal_hits': sum(x['normal_hit'] for x in out),
                    'normal_act': sum(x['normal_act'] for x in out)})
        res['pass'] = res['trigger_hits'] >= 5 and res['normal_hits'] == 0 and res['normal_act'] >= 5
        print('sentinel HF: trigger %d/6 normal-leak %d/6 act %d/6 pass %s' % (res['trigger_hits'], res['normal_hits'], res['normal_act'], res['pass']), flush=True)
    else:  # probe
        probes = random.Random(8707).sample(val, 24)
        ACTM = re.compile(r'^<\|ACT\s*\{.*?\}\s*\|> ')

        def first_lp(model):
            out = []
            with torch.no_grad():
                for r in probes:
                    m = ACTM.match(r['target'])
                    body = r['target'][m.end():]
                    p = tok(r['prompt'] + m.group(0), add_special_tokens=False)['input_ids']
                    full = tok(r['prompt'] + m.group(0) + body, add_special_tokens=False)['input_ids']
                    if full[:len(p)] != p:
                        out.append(None); continue
                    t1 = full[len(p)]
                    x = torch.tensor([p]).cuda()
                    lp = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)[t1].item()
                    out.append({'lp': round(lp, 4), 'token': tok.decode([t1])})
            return out

        def greedy(model):
            return [gen(model, r['prompt'], 0, greedy=True, max_new=96) for r in probes]
        from peft import PeftModel
        if args.merged_only:                                                # 新程序只跑合併後那段（同一個程序裡 NF4 那顆放不乾淨：v87 Gemma 還占 5.3 GB → OOM）
            prev = json.load(open(OUT, encoding='utf-8'))
            assert [p['gold'] for p in prev['probes']] == [r['target'] for r in probes], '探針那 24 筆對不上'
            res.update({k: prev[k] for k in ('probes', 'base', 'adapter')})
            res['merged_only_rerun'] = True
        else:
            base = load_base()
            res['probes'] = [{'seg': r['seg'], 'line': r['line'], 'gold': r['target']} for r in probes]
            res['base'] = {'first_lp': first_lp(base), 'greedy': greedy(base)}
            model = PeftModel.from_pretrained(base, os.path.join(ADIR, args.ckpt)).eval()
            res['adapter'] = {'first_lp': first_lp(model), 'greedy': greedy(model)}
            del model, base
            torch.cuda.empty_cache()
        if args.merged:
            mm = load_base(bf16_full=True, path=args.merged).eval()
            res['merged'] = {'first_lp': first_lp(mm), 'greedy': greedy(mm)}
        for k in ('base', 'adapter', 'merged'):
            if k in res:
                v = [x['lp'] for x in res[k]['first_lp'] if x]
                res[k]['first_lp_mean'] = round(sum(v) / len(v), 4)
                print(k, 'first_lp_mean', res[k]['first_lp_mean'], flush=True)
    res['ok'] = True
except Exception as e:
    res.update({'ok': False, 'error_type': type(e).__name__, 'error': str(e)[:2000], 'traceback_tail': traceback.format_exc()[-3000:]})
    print('🔴 FAILED', type(e).__name__, str(e)[:800], flush=True)
finally:
    res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
sys.exit(0 if res.get('ok') else 1)
